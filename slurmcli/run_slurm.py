import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import sys, threading, atexit
from collections import defaultdict

# --- minimal background tailer ---
_tail_thread = None
_tail_stop = threading.Event()
_tail_seen = defaultdict(int)  # source -> last line index

from dask import delayed
from dask.distributed import Client, progress

from .launch_slurm import SCHEDULER_HOST, PORT_SLURM_SCHEDULER

def _tail_logs_loop(client, interval: float, include_scheduler: bool, n_lines: int):
    sys.stdout.write(f"📜 Tailing Dask logs every {interval:.1f}s\n"); sys.stdout.flush()
    while not _tail_stop.wait(interval):
        try:
            # workers
            for addr, lines in client.get_worker_logs(n=n_lines).items():
                key = f"worker::{addr}"
                start = _tail_seen[key]
                if start < len(lines):
                    sys.stdout.write("".join(lines[start:]))
                    _tail_seen[key] = len(lines)
            # scheduler
            if include_scheduler:
                sch = client.get_scheduler_logs(n=n_lines)
                key = "scheduler"
                start = _tail_seen[key]
                if start < len(sch):
                    sys.stdout.write("".join(sch[start:]))
                    _tail_seen[key] = len(sch)
            sys.stdout.flush()
        except Exception:
            # ignore transient failures (worker restarts, etc.)
            pass
    sys.stdout.write("📜 Log tailer stopped\n"); sys.stdout.flush()

def _start_log_tailer(client, interval=1.0, include_scheduler=True, n_lines=2000):
    global _tail_thread
    if _tail_thread and _tail_thread.is_alive():
        return
    _tail_stop.clear()
    _tail_thread = threading.Thread(
        target=_tail_logs_loop,
        args=(client, interval, include_scheduler, n_lines),
        name="dask-log-tailer",
        daemon=True,
    )
    _tail_thread.start()
    atexit.register(stop_log_tailing)

def stop_log_tailing():
    _tail_stop.set()
    if _tail_thread and _tail_thread.is_alive():
        _tail_thread.join(timeout=2)


SCHEDULER_ADDRESS_FILE = Path().home() / ".dask_scheduler_address"
DEFAULT_SCHEDULER_ADDRESS = f"tcp://{SCHEDULER_HOST}:{PORT_SLURM_SCHEDULER}"


def _load_saved_scheduler_address() -> Optional[str]:
    """Return scheduler address stored on disk, if available."""
    try:
        raw = SCHEDULER_ADDRESS_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(f"⚠️ Unable to read {SCHEDULER_ADDRESS_FILE}: {exc}")
        return None
    if not raw:
        return None
    return raw if "://" in raw else f"tcp://{raw}"


def _connect_to_scheduler() -> Tuple[Client, str]:
    """Connect to a running scheduler, preferring the last recorded address."""
    attempts = []
    saved_addr = _load_saved_scheduler_address()
    if saved_addr:
        attempts.append(("saved", saved_addr))
    attempts.append(("configured", DEFAULT_SCHEDULER_ADDRESS))

    last_error: Optional[Exception] = None
    for source, address in attempts:
        try:
            client = Client(address)
            if source == "saved":
                print(f"ℹ️ Using scheduler address from {SCHEDULER_ADDRESS_FILE}: {address}")
            else:
                print(f"ℹ️ Using configured scheduler address: {address}")
            return client, address
        except Exception as exc:
            print(f"⚠️ Failed to connect to scheduler via {source} address ({address}): {exc}")
            last_error = exc

    raise RuntimeError("Unable to connect to Dask scheduler") from last_error


# --- change: enable tailing by default via get_client() ---
def get_client(
    *,
    tail_logs: bool = True,
    log_interval: float = 1.0,
    include_scheduler_logs: bool = True,
    max_log_lines: int = 2000,
) -> Client:
    client, _ = _connect_to_scheduler()
    print("🚀 Connected to scheduler:", client.scheduler.address)
    print("Client status:", client.status)
    if tail_logs:
        _start_log_tailer(
            client,
            interval=log_interval,
            include_scheduler=include_scheduler_logs,
            n_lines=max_log_lines,
        )
    return client

def close_workers():
    client, address_used = _connect_to_scheduler()
    workers = list(client.scheduler_info()["workers"])

    print(f"ℹ️ Retiring workers connected to {address_used}")
    client.retire_workers(workers=workers, close_workers=True, remove=True)
    client.close()


def main() -> None:
    import jax, jax.numpy as jnp
    import numpy as np


    client = get_client()

    # Get worker info
    sched_info = client.scheduler_info()
    workers = sched_info.get("workers", {})
    n_workers = len(workers)
    print(f"Workers connected: {n_workers}")

    if n_workers > 0:
        total_cores = sum(w.get("nthreads", 0) for w in workers.values())
        total_mem_bytes = sum(w.get("memory_limit", 0) for w in workers.values())
        total_mem_gb = total_mem_bytes / (1024 ** 3)
        print(f"Total cores: {total_cores}, Total memory: {total_mem_gb:.2f}GB")

    from tfl import get_device

    # small jitted kernel: returns a DeviceArray int
    @partial(jax.jit, static_argnums=(1,))
    def _count_inside(key, n):
        MAYBE_GPU = get_device('gpu')[0]
        xy = jax.random.uniform(key, shape=(n, 2))
        xy = jax.device_put(xy, MAYBE_GPU)
        r2 = jnp.sum(xy * xy, axis=1)
        return jnp.sum(r2 <= 1.0)

    def estimate_pi_task(seed_int: int, n_samples: int):
        # build PRNGKey on worker, call jitted kernel, return Python ints
        key = jax.random.PRNGKey(int(seed_int))
        inside = _count_inside(key, int(n_samples))
        return int(inside), int(n_samples)

    # --- prepare tasks as before ---
    MASTER_SEED = 42
    NUM_TASKS = 100
    TOTAL_SAMPLES = 200_000
    rng = np.random.default_rng(MASTER_SEED)
    task_seeds = rng.integers(0, 2**31 - 1, size=NUM_TASKS, dtype=np.int64)

    base = TOTAL_SAMPLES // NUM_TASKS
    rem = TOTAL_SAMPLES - base * NUM_TASKS
    samples = [base + (1 if i < rem else 0) for i in range(NUM_TASKS)]

    tasks = [delayed(estimate_pi_task)(int(s), int(n)) for s, n in zip(task_seeds, samples)]

    t0 = time.time()
    futures = client.compute(tasks)
    progress(futures)
    results = client.gather(futures)
    elapsed = time.time() - t0

    total_inside = sum(r[0] for r in results)
    total_samples = sum(r[1] for r in results)
    pi_est = 4.0 * total_inside / total_samples

    print(f"Done in {elapsed:.2f}s — samples={total_samples:,}, inside={total_inside:,}")
    print(f"π ≈ {pi_est:.12f}   jnp.pi={float(jnp.pi):.12f}   error={abs(pi_est - float(jnp.pi)):.12f}")


if __name__ == "__main__":
    main()
    
