import time
from functools import partial

import numpy as np
import jax, jax.numpy as jnp
from dask import delayed
from dask.distributed import Client, progress

from .launch_slurm import SCHEDULER_HOST, PORT_SLURM_SCHEDULER


def get_client() -> Client:
    client = Client(f"tcp://{SCHEDULER_HOST}:{PORT_SLURM_SCHEDULER}")
    
    # Print connection info
    print("🚀 Connected to scheduler:", client.scheduler.address)
    print("Client status:", client.status)
    return client


def main() -> None:

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
    
