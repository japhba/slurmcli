import atexit
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
import weakref
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import signal

from dask import delayed
from dask.distributed import Client, progress
from distributed.diagnostics.progressbar import ProgressBar
from distributed.utils import LoopRunner
import numpy as np

from .launch_slurm import (
    CORES,
    MEMORY_GB_DEFAULT,
    NUM_JOBS,
    PORT_SLURM_SCHEDULER,
    PROCESSES_PER_WORKER,
    SCHEDULER_HOST,
    VENV_ACTIVATE,
    WALLTIME,
    get_cluster,
)

logger = logging.getLogger("slurmcli.run_slurm")
_ACTIVE_CLIENT: Optional[Client] = None
_ACTIVE_CLUSTER: Any = None
_SIGNALS_INSTALLED = False
_DEBUG_ENABLED = os.getenv("SLURMCLI_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

# --- minimal background tailer ---
_tail_thread = None
_tail_stop = threading.Event()
_tail_seen = defaultdict(int)  # source -> last line index

def _normalize_log_lines(raw) -> list[str]:
    lines: list[str] = []
    def _add_line(text: str) -> None:
        lines.append(text if text.endswith("\n") else text + "\n")

    if isinstance(raw, dict):
        # flatten nanny/worker log buckets
        items = []
        for key, val in raw.items():
            items.append((key, val))
        raw_iter = items
    else:
        raw_iter = raw or []

    for entry in raw_iter:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], (list, tuple, dict)):
            # e.g. ("worker", [...]) or ("nanny", {...})
            prefix, payload = entry
            nested = payload
            if isinstance(nested, dict):
                for sub_key, sub_val in nested.items():
                    sub_lines = _normalize_log_lines(sub_val)
                    for line in sub_lines:
                        _add_line(f"[{prefix}/{sub_key}] {line.rstrip()}")
            else:
                sub_lines = _normalize_log_lines(nested)
                for line in sub_lines:
                    _add_line(f"[{prefix}] {line.rstrip()}")
            continue

        if isinstance(entry, str):
            _add_line(entry)
        elif isinstance(entry, bytes):
            _add_line(entry.decode(errors="replace"))
        elif isinstance(entry, tuple) and len(entry) >= 2:
            prefix = entry[0]
            payload = entry[1] if entry[1] is not None else ""
            if isinstance(payload, bytes):
                payload = payload.decode(errors="replace")
            _add_line(f"[{prefix}] {payload}")
        else:
            _add_line(str(entry))
    return lines


def _tail_logs_loop(client, interval: float, include_scheduler: bool, n_lines: int):
    sys.stdout.write(f"📜 Tailing Dask logs every {interval:.1f}s\n"); sys.stdout.flush()
    while not _tail_stop.wait(interval):
        try:
            # workers
            for addr, raw_lines in client.get_worker_logs(n=n_lines, nanny=True).items():
                lines = _normalize_log_lines(raw_lines)
                key = f"worker::{addr}"
                start = _tail_seen[key]
                if start < len(lines):
                    sys.stdout.write("".join(lines[start:]))
                    _tail_seen[key] = len(lines)
            # scheduler
            if include_scheduler:
                sch = _normalize_log_lines(client.get_scheduler_logs(n=n_lines))
                key = "scheduler"
                start = _tail_seen[key]
                if start < len(sch):
                    sys.stdout.write("".join(sch[start:]))
                    _tail_seen[key] = len(sch)
            sys.stdout.flush()
        except Exception as exc:
            logger.debug("Log tailing iteration failed: %s", exc, exc_info=False)
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
DEFAULT_MEMORY = f"{MEMORY_GB_DEFAULT}GB"


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


def _connect_to_scheduler(timeout: str = "10s") -> Tuple[Client, str]:
    """Connect to a running scheduler, preferring the last recorded address."""
    attempts = []
    saved_addr = _load_saved_scheduler_address()
    if saved_addr:
        attempts.append(("saved", saved_addr))
    attempts.append(("configured", DEFAULT_SCHEDULER_ADDRESS))

    last_error: Optional[Exception] = None
    for source, address in attempts:
        try:
            client = Client(address, timeout=timeout)
            if source == "saved":
                print(f"ℹ️ Using scheduler address from {SCHEDULER_ADDRESS_FILE}: {address}")
            else:
                print(f"ℹ️ Using configured scheduler address: {address}")
            try:
                SCHEDULER_ADDRESS_FILE.write_text(client.scheduler.address)
            except Exception:
                pass
            return client, address
        except Exception as exc:
            print(f"⚠️ Failed to connect to scheduler via {source} address ({address}): {exc}")
            last_error = exc

    raise RuntimeError("Unable to connect to Dask scheduler") from last_error


def _register_cluster_cleanup(client: Client, cluster: Any) -> None:
    global _ACTIVE_CLIENT, _ACTIVE_CLUSTER, _SIGNALS_INSTALLED
    _ACTIVE_CLIENT = client
    _ACTIVE_CLUSTER = cluster

    def _cleanup():
        stop_log_tailing()
        try:
            client.close()
        except Exception:
            pass
        try:
            if cluster is not None:
                cluster.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    weakref.finalize(client, _cleanup)
    if not _SIGNALS_INSTALLED:
        try:
            def handler(signum, frame):
                _cleanup()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal.signal(sig, handler)
                except Exception:
                    pass
            _SIGNALS_INSTALLED = True
        except Exception:
            pass


def _dashboard_url(client: Client, cluster: Any) -> Optional[str]:
    """Best-effort dashboard URL for logging."""
    if cluster is not None:
        link = getattr(cluster, "dashboard_link", None)
        if link:
            return str(link)
    try:
        info = client.scheduler_info()
        services = info.get("services", {}) if isinstance(info, dict) else {}
        dash_port = services.get("dashboard") or services.get("bokeh")
        if dash_port:
            host = client.scheduler.address.split("://")[-1].split(":")[0]
            return f"http://{host}:{dash_port}"
    except Exception:
        return None
    return None


def _log_worker_allocations(client: Client) -> None:
    try:
        info = client.scheduler_info()
    except Exception as exc:
        logger.info("Worker allocation: unable to query scheduler info (%s)", exc)
        return
    workers = info.get("workers", {}) if isinstance(info, dict) else {}
    if not workers:
        logger.info("Worker allocation: no workers connected")
        return
    logger.info("Worker allocation: %d worker(s)", len(workers))
    for addr, meta in workers.items():
        host = addr.split("://")[-1].split(":")[0]
        name = meta.get("name") or meta.get("id") or "unknown"
        pid = meta.get("pid")
        threads = meta.get("nthreads", 0)
        mem_bytes = meta.get("memory_limit", 0) or 0
        mem_gb = mem_bytes / (1024 ** 3) if mem_bytes else 0.0
        resources = meta.get("resources", {}) or {}
        local_dir = meta.get("local_directory")
        services = meta.get("services", {}) or {}
        gpu_total = 0.0
        for key, val in resources.items():
            if "gpu" in str(key).lower():
                try:
                    gpu_total += float(val or 0)
                except (TypeError, ValueError):
                    pass
        logger.info(
            "Worker %s name=%s pid=%s host=%s threads=%s mem=%.2fGB gpus=%s resources=%s local_dir=%s services=%s",
            addr,
            name,
            pid,
            host,
            threads,
            mem_gb,
            gpu_total if gpu_total else 0,
            resources,
            local_dir,
            services,
        )


class _TqdmProgressBar(ProgressBar):
    __loop = None

    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        complete=True,
        loop=None,
        desc: str = "slurmcli.vmap",
        start: bool = True,
    ):
        self._loop_runner = loop_runner = LoopRunner(loop=loop)
        super().__init__(keys, scheduler, interval, complete)
        try:
            from tqdm.std import tqdm
        except Exception as exc:
            raise RuntimeError("tqdm (std backend) is required for slurmcli.vmap progress output") from exc
        self._tqdm = tqdm(total=0, desc=desc, leave=True, dynamic_ncols=True)
        self._last_done = 0

        if start:
            loop_runner.run_sync(self.listen)

    def _draw_bar(self, remaining, all, **kwargs):
        total = int(all or 0)
        if self._tqdm.total != total:
            self._tqdm.total = total
            self._tqdm.refresh()
        done = total - int(remaining or 0)
        delta = done - self._last_done
        if delta > 0:
            self._tqdm.update(delta)
            self._last_done = done

    def _draw_stop(self, status=None, **kwargs):
        if self._tqdm.total and self._last_done < self._tqdm.total:
            self._tqdm.update(self._tqdm.total - self._last_done)
            self._last_done = self._tqdm.total
        if status == "error":
            self._tqdm.set_postfix_str("error", refresh=False)
        self._tqdm.close()



def _wait_for_workers(client: Client, attempts: int, delay: float) -> bool:
    """Return True if any workers appear within the probe window."""
    for i in range(max(1, attempts)):
        try:
            workers = client.scheduler_info().get("workers", {})
            if workers:
                return True
        except Exception as exc:
            logger.debug("Worker probe %d failed: %s", i + 1, exc, exc_info=False)
        time.sleep(delay)
    return False


def _infer_cluster_type(
    *,
    local: bool,
    partition: Optional[str],
    gres: Optional[str],
    num_gpus: int,
) -> str:
    if local:
        return "local"
    text = " ".join([partition or "", gres or ""]).lower()
    if "gpu" in text or num_gpus > 0:
        return "gpu"
    return "cpu"


def _cluster_type_from_workers(workers: dict) -> Optional[str]:
    if not workers:
        return None
    any_gpu = False
    hosts: set[str] = set()
    for addr, meta in workers.items():
        resources = meta.get("resources", {}) or {}
        for key, val in resources.items():
            if "gpu" in str(key).lower() and float(val or 0) > 0:
                any_gpu = True
                break
        host = addr.split("://")[-1].split(":")[0]
        hosts.add(host)
    if any_gpu:
        return "gpu"
    local_hosts = {"127.0.0.1", "localhost"}
    if hosts and hosts.issubset(local_hosts):
        return "local"
    return "cpu"


def get_client(
    *,
    local: bool = False,
    partition: Optional[str] = None,
    walltime: Optional[str] = None,
    num_jobs: Optional[int] = None,
    processes: Optional[int] = None,
    cores_per_process: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    memory: Optional[str] = None,
    nodelist: Optional[str] = None,
    num_gpus: int = 0,
    gres: Optional[str] = None,
    account: str = "gcnu-ac",
    job_name: str = "tfl",
    log_dir: str = "dask_logs",
    venv_activate: Optional[str] = VENV_ACTIVATE,
    verbose: int = 0,
    tail_logs: bool = True,
    log_interval: float = 1.0,
    include_scheduler_logs: bool = True,
    max_log_lines: int = 2000,
    shutdown_on_exit: bool = True,
    return_cluster: bool = False,
    log_worker_details: bool = True,
    worker_probe_attempts: int = 3,
    worker_probe_delay: float = 2.0,
    ensure_gc_cleanup: bool = True,
    launch_if_no_workers: bool = True,
    cluster_type: Optional[str] = None,
    connect_timeout: str = "5s",
    local_processes: Optional[bool] = None,
    reuse_existing: bool = True,
) -> Union[Client, Tuple[Client, Any]]:
    """Connect to an existing scheduler or launch one if none is reachable.

    The SLURM arguments mirror the interactive CLI: partition, walltime,
    num_jobs, processes, cores_per_process / threads_per_worker, memory,
    nodelist, num_gpus/gres, account, job_name, log_dir, and worker venv
    activation. The function reuses any reachable scheduler first. By default it
    will auto-launch if a scheduler exists but has zero workers (set
    `launch_if_no_workers=False` to skip). Use `local=True` to prefer a
    LocalCluster. `shutdown_on_exit` now defaults to True so clusters are torn
    down on kernel restart; disable if you want them to live beyond the calling
    process. Set `cluster_type` to `local`, `gpu`, or `cpu` to only reuse
    clusters whose workers match that type (default inferred from args).
    """
    cluster = None
    client = None
    need_launch = False
    target_type = (cluster_type or _infer_cluster_type(local=local, partition=partition, gres=gres, num_gpus=int(num_gpus or 0))).lower()

    # Try to reuse an existing scheduler first
    if reuse_existing:
        try:
            client, addr = _connect_to_scheduler(timeout=connect_timeout)
            has_workers = _wait_for_workers(client, worker_probe_attempts, worker_probe_delay)
            n_workers = len(client.scheduler_info().get("workers", {}))
            existing_type = _cluster_type_from_workers(client.scheduler_info().get("workers", {}))
            type_matches = (existing_type == target_type) if existing_type else False

            if has_workers and type_matches:
                logger.info(
                    "Reusing existing scheduler at %s with %d %s worker(s)",
                    addr,
                    n_workers,
                    target_type,
                )
            elif has_workers and not type_matches:
                logger.info(
                    "Existing scheduler at %s has %d worker(s) of type %s; expected %s. Launching a new %s cluster.",
                    addr,
                    n_workers,
                    existing_type or "unknown",
                    target_type,
                    "local" if local else "SLURM",
                )
                try:
                    client.close()
                except Exception:
                    pass
                client = None
                need_launch = True
            elif not has_workers and launch_if_no_workers:
                logger.info(
                    "Existing scheduler at %s has no workers after %d probe(s); launching a %s cluster.",
                    addr,
                    worker_probe_attempts,
                    "local" if local else "SLURM",
                )
                try:
                    client.close()
                except Exception:
                    pass
                client = None
                need_launch = True
            else:
                logger.info(
                    "Existing scheduler at %s has no workers (target type %s). Not launching because launch_if_no_workers=False.",
                    addr,
                    target_type,
                )
        except Exception as exc:
            logger.info(
                "No existing scheduler reachable (%s); launching a %s cluster.",
                type(exc).__name__,
                "local" if local else "SLURM",
            )
            need_launch = True
    else:
        need_launch = True

    if need_launch:
        jobs = int(num_jobs or NUM_JOBS)
        proc = int(processes or PROCESSES_PER_WORKER)
        cores = int(cores_per_process or CORES)
        threads = int(threads_per_worker or cores)
        mem = memory or DEFAULT_MEMORY
        wall = walltime or WALLTIME
        effective_gres = gres
        if not effective_gres and num_gpus:
            effective_gres = f"gpu:{int(num_gpus)}"

        cluster, client = get_cluster(
            local=local,
            queue=partition,
            account=account,
            num_jobs=jobs,
            processes=proc,
            threads_per_worker=threads,
            cores=cores,
            memory=mem,
            walltime=wall,
            log_dir=log_dir,
            job_name=job_name,
            venv_activate=venv_activate,
            nodelist=nodelist,
            gres=effective_gres,
            verbose=verbose,
            local_processes=local_processes,
        )
        try:
            SCHEDULER_ADDRESS_FILE.write_text(client.scheduler.address)
            logger.info("Scheduler address saved to %s", SCHEDULER_ADDRESS_FILE)
        except Exception:
            pass
        if shutdown_on_exit:
            _register_cluster_cleanup(client, cluster)
        elif ensure_gc_cleanup:
            weakref.finalize(client, client.close)
            if cluster is not None:
                weakref.finalize(cluster, cluster.close)
    else:
        print("🚀 Connected to scheduler:", client.scheduler.address)
        print("Client status:", client.status)

    if tail_logs:
        _start_log_tailer(
            client,
            interval=log_interval,
            include_scheduler=include_scheduler_logs,
            n_lines=max_log_lines,
        )

    dash = _dashboard_url(client, cluster)
    if dash:
        logger.info("Dashboard: %s", dash)
    if log_worker_details:
        _log_worker_allocations(client)

    if return_cluster:
        return client, cluster
    return client

def close_workers():
    client, address_used = _connect_to_scheduler()
    workers = list(client.scheduler_info()["workers"])

    print(f"ℹ️ Retiring workers connected to {address_used}")
    client.retire_workers(workers=workers, close_workers=True, remove=True)
    client.close()


def _normalize_in_axes(in_axes: Any, nargs: int) -> list[Optional[int]]:
    if isinstance(in_axes, (list, tuple)):
        axes = list(in_axes)
        if len(axes) != nargs:
            raise ValueError(f"in_axes must have length {nargs}, got {len(axes)}")
        return [None if ax is None else int(ax) for ax in axes]
    return [None if in_axes is None else int(in_axes) for _ in range(nargs)]


def _axis_size(arg: Any, axis: int) -> int:
    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported by slurmcli.vmap")
    if hasattr(arg, "shape") and arg.shape is not None:
        return int(arg.shape[axis])
    return len(arg)


def _slice_arg(arg: Any, axis: Optional[int], index: int) -> Any:
    if axis is None:
        return arg
    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported by slurmcli.vmap")
    return arg[index]


def _stack_results(results: Sequence[Any], out_axes: Optional[int]) -> Any:
    if out_axes is None:
        return list(results)
    if out_axes != 0:
        raise NotImplementedError("Only out_axes=0 is supported by slurmcli.vmap")
    if not results:
        return np.asarray([])
    first = results[0]
    if isinstance(first, (list, tuple)):
        num_fields = len(first)
        grouped = [[res[i] for res in results] for i in range(num_fields)]
        stacked = [np.stack(group, axis=0) for group in grouped]
        return type(first)(stacked)
    return np.stack(results, axis=0)


def vmap(
    fun,
    in_axes: Any = 0,
    out_axes: Any = 0,
    *,
    axis_name: Optional[str] = None,
    axis_size: Optional[int] = None,
    client: Optional[Client] = None,
    show_progress: bool = True,
    progress_mode: str = "tqdm",
    cache_check: Any = "auto",
    return_info: bool = False,
    disconnect: bool = True,
    **client_kwargs,
):
    """Distributed vmap similar to jax.vmap, executing slices via Dask.

    Accepts all get_client kwargs via **client_kwargs (e.g. local, num_jobs).
    Only axis=0 is supported for now. Returns the gathered results (stacked).
    If cache_check="auto" (default), vmap uses joblib-style cache checks when
    the function exposes check_call_in_cache; when all calls return True, vmap
    runs locally instead of launching Dask jobs.
    """
    if axis_name is not None:
        raise NotImplementedError("axis_name is not supported by slurmcli.vmap")
    if axis_size is not None:
        raise NotImplementedError("axis_size is not supported by slurmcli.vmap")

    def _mapped(*args: Any):
        if not args:
            raise ValueError("vmap-mapped function requires at least one argument")
        axes = _normalize_in_axes(in_axes, len(args))
        mapped_sizes = [(_axis_size(arg, ax) if ax is not None else None) for arg, ax in zip(args, axes)]
        inferred = [size for size in mapped_sizes if size is not None]
        if not inferred:
            raise ValueError("vmap requires at least one argument with in_axes != None")
        size = inferred[0]
        for other in inferred[1:]:
            if other != size:
                raise ValueError(f"vmap arguments have mismatched axis sizes: {inferred}")

        use_local = False
        cache_probe = None
        if cache_check in ("auto", True):
            cache_probe = getattr(fun, "check_call_in_cache", None)
        elif callable(cache_check):
            cache_probe = cache_check


        if cache_probe is not None:
            try:
                all_cached = True
                for i in range(size):
                    slice_args = [_slice_arg(arg, ax, i) for arg, ax in zip(args, axes)]
                    if not cache_probe(*slice_args):
                        all_cached = False
                        break
                use_local = all_cached
            except Exception as exc:
                logger.debug("Cache check failed; using Dask (%s)", exc, exc_info=False)

        if use_local:
            t0 = time.time()
            iterator = range(size)
            if show_progress:
                try:
                    from tqdm.std import tqdm
                except Exception as exc:
                    raise RuntimeError("tqdm (std backend) is required for slurmcli.vmap progress output") from exc
                iterator = tqdm(iterator, total=size, desc="slurmcli.vmap")
            if _DEBUG_ENABLED:
                logger.info("vmap: running locally for %d task(s)", size)
            results = [
                fun(*[_slice_arg(arg, ax, i) for arg, ax in zip(args, axes)])
                for i in iterator
            ]
            elapsed = time.time() - t0
            info = {
                "n_tasks": size,
                "elapsed": elapsed,
                "scheduler_address": None,
                "cache_only": True,
            }
            output = _stack_results(results, out_axes)
            return (output, info) if return_info else output

        owns_client = client is None
        cluster = None
        active_client = client
        if active_client is None:
            kwargs = dict(client_kwargs)
            kwargs.pop("return_cluster", None)
            if "tail_logs" not in kwargs:
                kwargs["tail_logs"] = False
            if kwargs.get("local") and "reuse_existing" not in kwargs:
                # Avoid hanging on stale remote scheduler addresses when running locally.
                kwargs["reuse_existing"] = False
            if _DEBUG_ENABLED:
                logger.info("vmap: acquiring client with %s", kwargs)
            active_client, cluster = get_client(return_cluster=True, **kwargs)

        t0 = time.time()
        tasks = [
            delayed(fun)(*[_slice_arg(arg, ax, i) for arg, ax in zip(args, axes)])
            for i in range(size)
        ]
        if _DEBUG_ENABLED:
            logger.info("vmap: submitting %d task(s) to Dask", len(tasks))
        futures = active_client.compute(tasks)
        if show_progress:
            mode = (progress_mode or "tqdm").lower()
            if mode not in ("tqdm", "auto"):
                raise ValueError("Only progress_mode='tqdm' is supported")
            _TqdmProgressBar(
                futures,
                scheduler=active_client.scheduler.address if active_client else None,
                interval="100ms",
                complete=True,
            )
        results = active_client.gather(futures)
        if _DEBUG_ENABLED:
            logger.info("vmap: gathered %d result(s)", len(results))
        elapsed = time.time() - t0

        scheduler = getattr(active_client, "scheduler", None) if active_client else None
        info = {
            "n_tasks": size,
            "elapsed": elapsed,
            "scheduler_address": scheduler.address if scheduler else None,
        }

        output = _stack_results(results, out_axes)

        if owns_client and disconnect:
            try:
                active_client.close()
            except Exception:
                pass
            try:
                if cluster is not None:
                    cluster.close()
            except Exception:
                pass

        if return_info:
            if not (owns_client and disconnect):
                info["client"] = active_client
                info["cluster"] = cluster
            return output, info
        return output

    return _mapped


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
    
