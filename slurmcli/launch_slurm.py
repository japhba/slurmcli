#!/usr/bin/env python3

"""
slurmcli (shell-commands edition)
---------------------------------

Back to the previous approach: we use Slurm CLI tools (`sinfo`, `squeue`,
`scontrol`) for cluster introspection. Dask launch is still handled via
`dask_jobqueue.SLURMCluster`. Optional per-worker Jupyter servers are
unchanged.

Verbosity:
  default   : concise
  -v        : more detail (per-node lines when small buckets)
  -vv       : include admin/hidden partitions (via `sinfo -a`) and list job names per node
  -vvv      : also show user lists per node
  -vvvv     : include reserved resources for each job

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import select
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Third-party (kept for parity with original script)
import numpy as np  # noqa: F401
import jax, jax.numpy as jnp  # noqa: F401
from dask import delayed  # noqa: F401
from dask.distributed import Client, LocalCluster
from distributed.utils import is_kernel
from dask_jobqueue import SLURMCluster

from slurmcli.cluster_status import (
    _format_job_time_info,
    aggregate_gpu_counts,
    format_gpu_counts,
    format_gpu_labels,
    get_detailed_node_info,
    get_partitions,
)

# --- Optional color output ----------------------------------------------------
try:
    from colorama import Fore, Style, init as _colorama_init

    _colorama_init()
    COLOR_ENABLED = True
except Exception:  # optional
    COLOR_ENABLED = False

    class _DummyColors:
        def __getattr__(self, item):
            return ""

    Fore = Style = _DummyColors()


def colorize(text: str, *styles: str) -> str:
    if COLOR_ENABLED and styles:
        return "".join(styles) + text + Style.RESET_ALL
    return text


def emphasize(text: str, color: Optional[str] = None) -> str:
    if COLOR_ENABLED:
        styles: List[str] = [Style.BRIGHT]
        if color:
            styles.append(color)
        return colorize(text, *styles)
    return f"**{text}**"


# --- Config ------------------------------------------------------------------
CONFIG_FILE = Path.home() / ".slurmcli"
logger = logging.getLogger("slurmcli")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

CONFIG_FIELDS: List[Dict[str, Any]] = [
    {"key": "mail", "env": "SLURMCLI_MAIL", "prompt": "Notification email address", "type": str, "default": ""},
    {"key": "scheduler_host", "env": "SLURMCLI_SCHEDULER_HOST", "prompt": "Scheduler hostname/IP (for reconnecting)", "type": str, "default": "192.168.240.53"},
    {"key": "scheduler_port", "env": "SLURMCLI_SCHEDULER_PORT", "prompt": "Dask scheduler port", "type": int, "default": 8786},
    {"key": "dashboard_port", "env": "SLURMCLI_DASHBOARD_PORT", "prompt": "Dask dashboard port", "type": int, "default": 8787},
    {"key": "jupyter_port", "env": "SLURMCLI_JUPYTER_PORT", "prompt": "Base Jupyter port", "type": int, "default": 11833},
    {"key": "venv_activate", "env": "SLURMCLI_VENV_ACTIVATE", "prompt": "Path to Python virtualenv to activate in workers", "type": str, "default": "/nfs/nhome/live/jbauer/recurrent_feature/.venv"},
]


def _cast_value(value: Any, typ: type) -> Any:
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    return str(value)


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_config_file(path: Path, data: Dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        print(f"⚠️  Unable to persist credentials to {path}: {exc}")


def load_credentials() -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    file_cache = _load_config_file(CONFIG_FILE)
    missing: List[Dict[str, Any]] = []

    for field in CONFIG_FIELDS:
        key, env_name, typ = field["key"], field["env"], field["type"]
        env_val = os.getenv(env_name)
        if env_val is not None:
            try:
                config[key] = _cast_value(env_val, typ)
                continue
            except ValueError:
                print(f"⚠️  Invalid value in environment variable {env_name}; ignoring.")

        if key in file_cache:
            try:
                config[key] = _cast_value(file_cache[key], typ)
                continue
            except ValueError:
                pass
        missing.append(field)

    if missing:
        print("\n🗝️  slurmcli needs a few credentials to continue.")
        for field in missing:
            key, typ = field["key"], field["type"]
            default = file_cache.get(key, field.get("default"))
            prompt = field["prompt"]
            if default is not None:
                prompt += f" [{default}]"
            prompt += ": "

            while True:
                user_input = input(prompt).strip()
                if user_input == "" and default is not None:
                    user_input = default
                try:
                    config[key] = _cast_value(user_input, typ)
                    break
                except (TypeError, ValueError):
                    print("   Invalid value, please try again.")

            file_cache[key] = config[key]
        _save_config_file(CONFIG_FILE, file_cache)

    logger.info("Loaded slurmcli configuration: %s", {k: config.get(k) for k in sorted(config)})
    return config


CONFIG = load_credentials()
MAIL = CONFIG.get("mail")
SCHEDULER_HOST = CONFIG.get("scheduler_host")
PORT_SLURM_SCHEDULER = int(CONFIG.get("scheduler_port"))
PORT_SLURM_DASHBOARD = int(CONFIG.get("dashboard_port"))
JUPYTER_PORT = int(CONFIG.get("jupyter_port"))
VENV_ACTIVATE = CONFIG.get("venv_activate")
DEFAULT_JUPYTER_RUNTIME_BASE = "/tmp/jrt"
SCHEDULER_ADDRESS_FILE = Path().home() / ".dask_scheduler_address"

LAST_LAUNCH_FILE = Path.home() / ".slurmcli_last_launch.json"
SAVED_CONFIGS_FILE = Path.home() / ".slurmcli_saved_configs.json"
LAUNCH_CONFIG_VERSION = 1
REQUIRED_LAUNCH_FIELDS = {
    "walltime",
    "num_jobs",
    "processes",
    "cores_per_process",
    "threads_per_worker",
    "memory",
    "launch_kernels",
}
PERSISTED_LAUNCH_FIELDS = [
    "partition",
    "walltime",
    "num_jobs",
    "processes",
    "cores_per_process",
    "threads_per_worker",
    "memory",
    "nodelist",
    "num_gpus",
    "gres",
    "launch_kernels",
]


def _normalize_launch_config(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not REQUIRED_LAUNCH_FIELDS.issubset(data):
        return None
    normalized = dict(data)
    for key in ("num_jobs", "processes", "cores_per_process", "threads_per_worker", "num_gpus"):
        if normalized.get(key) is not None:
            try:
                normalized[key] = int(normalized[key])
            except (TypeError, ValueError):
                return None
    normalized["launch_kernels"] = bool(normalized.get("launch_kernels", False))
    return normalized


def load_last_launch_config() -> Optional[Dict[str, Any]]:
    try:
        raw = LAST_LAUNCH_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(f"⚠️ Unable to read last launch configuration ({exc})")
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"⚠️ Could not parse {LAST_LAUNCH_FILE}: {exc}")
        return None

    return _normalize_launch_config(data)


def save_last_launch_config(config: Dict[str, Any]) -> None:
    payload = {k: config.get(k) for k in PERSISTED_LAUNCH_FIELDS}
    payload["config_version"] = LAUNCH_CONFIG_VERSION
    payload["saved_at"] = time.time()
    try:
        LAST_LAUNCH_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"⚠️ Unable to write {LAST_LAUNCH_FILE}: {exc}")


def _load_named_configs_raw() -> Dict[str, Any]:
    try:
        raw = SAVED_CONFIGS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        print(f"⚠️ Unable to read {SAVED_CONFIGS_FILE}: {exc}")
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        print(f"⚠️ Could not parse {SAVED_CONFIGS_FILE}: {exc}")
        return {}


def load_named_launch_config(name: str) -> Optional[Dict[str, Any]]:
    configs = _load_named_configs_raw()
    cfg = configs.get(name)
    if not cfg:
        return None
    normalized = _normalize_launch_config(cfg)
    if normalized is None:
        return None
    normalized["saved_at"] = cfg.get("saved_at")
    normalized["__config_name__"] = name
    return normalized


def save_named_launch_config(name: str, config: Dict[str, Any]) -> None:
    payload = {k: config.get(k) for k in PERSISTED_LAUNCH_FIELDS}
    payload["config_version"] = LAUNCH_CONFIG_VERSION
    payload["saved_at"] = time.time()
    configs = _load_named_configs_raw()
    configs[name] = payload
    try:
        SAVED_CONFIGS_FILE.write_text(json.dumps(configs, indent=2), encoding="utf-8")
        print(f"💾 Saved configuration as '{name}'")
    except OSError as exc:
        print(f"⚠️ Unable to write {SAVED_CONFIGS_FILE}: {exc}")


def format_saved_timestamp(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return None


def print_launch_summary(config: Dict[str, Any], *, title: str = "📋 LAUNCH SUMMARY") -> None:
    partition_display = config.get("partition") or "SLURM default (no explicit queue)"
    walltime = config.get("walltime", "unknown")
    num_jobs = int(config.get("num_jobs") or 0)
    processes = int(config.get("processes") or 1)
    cores_per_process = int(config.get("cores_per_process") or 1)
    threads_per_worker = int(config.get("threads_per_worker") or cores_per_process or 1)
    total_cores_for_job = threads_per_worker * processes
    memory = config.get("memory", "n/a")
    total_workers = num_jobs * processes
    nodelist = config.get("nodelist") or "Any available"
    num_gpus = int(config.get("num_gpus") or 0)
    launch_kernels = bool(config.get("launch_kernels"))

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"  Partition / Queue: {partition_display}")
    print(f"  Walltime:          {walltime}")
    print(f"  SLURM Jobs:        {num_jobs}")
    print(f"  Processes per job: {processes}")
    print(f"  Cores per process: {cores_per_process}")
    print(f"  Threads per process (actual): {threads_per_worker}")
    print(f"  Total cores per job: {total_cores_for_job}")
    print(f"  Memory per job:    {memory}")
    print(f"  Total workers:     {total_workers}")
    print(f"  Specific node:     {nodelist}")
    print(f"  GPUs per process:  {num_gpus}")
    print(f"  Launch kernels:    {'Yes' if launch_kernels else 'No'}")
    print("=" * 50)


def _timed_input(prompt: str, timeout: float = 5.0) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except Exception:
        ready = []
    if ready:
        line = sys.stdin.readline()
        return line.strip()
    sys.stdout.write("\n")
    return ""


def maybe_save_named_config(config: Dict[str, Any]) -> None:
    print("\n💾 Save this configuration for later? (type a name within 5s, or press Enter to skip)")
    name = _timed_input("   Name: ")
    if name:
        save_named_launch_config(name, config)


# --- Helper parsers for CLI output -------------------------------------------

def is_port_available(port: Optional[int]) -> bool:
    """Return True if the given TCP port can be bound on the current host."""
    if port is None or port <= 0:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", int(port)))
        except OSError:
            return False
    return True


def _run_squeue(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=8)
        return result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _get_squeue_rows(partition: Optional[str] = None) -> List[Dict[str, str]]:
    base = ["squeue", "-h", "-o", "%i|%P|%T|%Q|%R|%u|%j"]
    if partition:
        base.extend(["-p", partition])

    rows_text = _run_squeue(base + ["--sort=-P"])
    if rows_text is None:
        rows_text = _run_squeue(base)
    if not rows_text:
        return []

    rows: List[Dict[str, str]] = []
    for line in rows_text.strip().splitlines():
        parts = line.split("|", 6)
        if len(parts) < 7:
            continue
        job_id, part, state, priority, reason, user, name = parts[:7]
        rows.append(
            {
                "job_id": job_id.strip(),
                "partition": part.strip(),
                "state": state.strip(),
                "priority": priority.strip(),
                "reason": reason.strip(),
                "user": user.strip(),
                "name": name.strip(),
            }
        )
    return rows


def _collect_slurm_job_ids(cluster: Any, timeout: float = 10.0, poll: float = 0.5) -> List[str]:
    deadline = time.time() + max(0.1, timeout)
    job_ids: set[str] = set()

    while time.time() <= deadline:
        workers = getattr(cluster, "workers", {}) or {}
        for worker in workers.values():
            job_id = getattr(worker, "job_id", None)
            if job_id:
                job_ids.add(str(job_id))
        if job_ids:
            break
        time.sleep(poll)

    return sorted(job_ids)


def _find_jobs_by_name(job_name: Optional[str], partition: Optional[str]) -> List[str]:
    if not job_name:
        return []
    user = os.getenv("USER")
    rows = _get_squeue_rows(partition=partition)
    matches: List[str] = []
    for row in rows:
        if user and row.get("user") != user:
            continue
        name = row.get("name") or ""
        if name == job_name or name.startswith(f"{job_name}-"):
            matches.append(row.get("job_id", ""))
    return [job_id for job_id in matches if job_id]


def _is_pending_state(state: str) -> bool:
    text = state.strip().upper()
    return text.startswith("PEND") or text.startswith("CONFIG")


def _report_queue_ahead(
    job_ids: List[str],
    *,
    partition: Optional[str],
    job_name: Optional[str],
    max_rows: int = 20,
) -> List[str]:
    rows = _get_squeue_rows(partition=partition)
    if not rows:
        print("ℹ️ Unable to query squeue for queue status.")
        return job_ids

    job_ids_set = set(job_ids)
    if not job_ids_set:
        job_ids = _find_jobs_by_name(job_name, partition)
        job_ids_set = set(job_ids)

    if not job_ids_set:
        queue_label = f"partition '{partition}'" if partition else "the default partition"
        print(f"ℹ️ Submitted jobs not visible yet; showing top pending jobs in {queue_label}.")
        pending = [row for row in rows if _is_pending_state(row["state"])]
        for row in pending[:max_rows]:
            print(
                f"   {row['job_id']} {row['user']} {row['name']} "
                f"{row['state']} prio={row['priority']} reason={row['reason']}"
            )
        return job_ids

    indices = [idx for idx, row in enumerate(rows) if row.get("job_id") in job_ids_set]
    if not indices:
        print("ℹ️ Submitted jobs not visible yet; waiting for them to appear in squeue.")
        return job_ids

    first_idx = min(indices)
    ahead = [row for row in rows[:first_idx] if _is_pending_state(row["state"])]
    queue_label = f"partition '{partition}'" if partition else "the default partition"
    print(emphasize(f"📋 Jobs ahead of us in {queue_label}: {len(ahead)}"))
    for row in ahead[:max_rows]:
        print(
            f"   {row['job_id']} {row['user']} {row['name']} "
            f"{row['state']} prio={row['priority']} reason={row['reason']}"
        )

    return job_ids


def _wait_for_workers_with_queue(
    client: Client,
    *,
    cluster: Any,
    total_workers_expected: int,
    partition: Optional[str],
    job_name: Optional[str],
    poll_interval: float = 30.0,
    max_rows: int = 20,
) -> None:
    job_ids = _collect_slurm_job_ids(cluster)
    job_ids = _report_queue_ahead(
        job_ids,
        partition=partition,
        job_name=job_name,
        max_rows=max_rows,
    )

    while True:
        try:
            client.wait_for_workers(n_workers=total_workers_expected, timeout=poll_interval)
            return
        except TimeoutError:
            job_ids = _report_queue_ahead(
                job_ids,
                partition=partition,
                job_name=job_name,
                max_rows=max_rows,
            )
        except Exception as exc:
            print(f"⚠️ Error while waiting for workers ({type(exc).__name__}: {exc})")
            return


# --- Dask cluster launch ------------------------------------------------------
WALLTIME = "01:00:00"
MEMORY_GB_DEFAULT = 32
PROCESSES_PER_WORKER = 1
CORES = 1
NUM_JOBS = 4


def get_cluster(
    *,
    local: bool = True,
    n_workers: Optional[int] = None,
    num_jobs: int = NUM_JOBS,
    processes: int = PROCESSES_PER_WORKER,
    threads_per_worker: Optional[int] = None,
    cores: int = CORES,
    memory: str = f"{MEMORY_GB_DEFAULT}GB",
    queue: Optional[str] = None,
    account: str = "gcnu-ac",
    walltime: str = WALLTIME,
    log_dir: str = "dask_logs",
    job_name: str = "tfl",
    venv_activate: Optional[str] = VENV_ACTIVATE,
    nodelist: Optional[str] = None,
    gres: Optional[str] = None,
    verbose: int = 0,
    local_processes: Optional[bool] = None,
    queue_poll_interval: float = 30.0,
    queue_report_max: int = 20,
):
    if threads_per_worker is None:
        threads_per_worker = cores

    threads_per_worker = int(max(1, threads_per_worker))
    processes = int(max(1, processes))
    num_jobs = int(max(1, num_jobs))
    if not local and n_workers is not None:
        num_jobs = int(max(1, (int(n_workers) + processes - 1) // processes))
        logger.warning(
            "n_workers is a convenience for SLURM and maps to num_jobs=%d (processes=%d).",
            num_jobs,
            processes,
        )
    cores_for_job = threads_per_worker * processes

    dashboard_address = f":{PORT_SLURM_DASHBOARD}" if PORT_SLURM_DASHBOARD else None
    scheduler_port_configured = PORT_SLURM_SCHEDULER
    scheduler_port = scheduler_port_configured
    if scheduler_port and scheduler_port > 0 and not is_port_available(scheduler_port):
        print(f"⚠️ Dask scheduler port {scheduler_port} already in use; requesting an ephemeral port instead.")
        scheduler_port = 0

    scheduler_options: Dict[str, Any] = {"port": scheduler_port if scheduler_port is not None else 0}
    if dashboard_address:
        scheduler_options["dashboard_address"] = dashboard_address

    if local:
        print("🚀 Starting a local Dask cluster…")
        if n_workers is not None:
            n_workers_local = int(max(1, n_workers))
        else:
            n_workers_local = num_jobs * processes
        if local_processes is None:
            use_processes = not is_kernel()
        else:
            use_processes = bool(local_processes)
        cluster = LocalCluster(
            n_workers=n_workers_local,
            threads_per_worker=threads_per_worker,
            memory_limit=memory,
            processes=use_processes,
            scheduler_port=scheduler_port if scheduler_port is not None else 0,
            dashboard_address=dashboard_address,
        )
        client = Client(cluster)
    else:
        queue_label = f"the '{queue}' SLURM partition" if queue else "the default SLURM partition"
        print(f"🚀 Submitting {num_jobs} jobs to {queue_label}…")
        os.makedirs(log_dir, exist_ok=True)
        prologue: List[str] = []
        if venv_activate:
            prologue.append(f"source {venv_activate}/bin/activate")

        job_extra_directives: List[str] = []
        worker_extra_args: List[str] = []
        if nodelist:
            job_extra_directives.append(f"--nodelist={nodelist}")
        if gres and 'gpu' in (gres or ""):
            try:
                num_gpus = int(gres.split(':')[-1])
                if num_gpus > 0:
                    job_extra_directives.append(f"--gpus-per-task={num_gpus}")
                    job_extra_directives.append(f"--gpu-bind=single:{num_gpus}")
                    worker_extra_args.extend(["--resources", f"GPU={num_gpus}"])
                    print(f"✅ Configuring workers with --gpus-per-task={num_gpus}")
            except Exception:
                if gres:
                    job_extra_directives.append(f"--gres={gres}")

        cluster = SLURMCluster(
            queue=queue,
            account=account or "",
            processes=processes,
            cores=cores_for_job,
            memory=memory,
            walltime=walltime,
            job_name=job_name,
            log_directory=log_dir,
            job_script_prologue=prologue,
            job_extra_directives=job_extra_directives,
            worker_extra_args=worker_extra_args,
            scheduler_options=scheduler_options,
        )
        cluster.scale(n=num_jobs)
        client = Client(cluster)

    if n_workers is not None:
        total_workers_expected = int(max(1, n_workers))
    else:
        total_workers_expected = num_jobs * processes
    print("Cluster dashboard:", getattr(cluster, "dashboard_link", "n/a"))
    print("Scheduler address:", client.scheduler.address)
    print(f"\n⏳ Waiting for {total_workers_expected} workers to connect…")
    try:
        if local:
            client.wait_for_workers(n_workers=total_workers_expected, timeout=300)
        else:
            _wait_for_workers_with_queue(
                client,
                cluster=cluster,
                total_workers_expected=total_workers_expected,
                partition=queue,
                job_name=job_name,
                poll_interval=queue_poll_interval,
                max_rows=queue_report_max,
            )
    except Exception as exc:
        print(f"⚠️ Timed out waiting for {total_workers_expected} workers ({type(exc).__name__}: {exc})")

    sched_info = client.scheduler_info()
    workers = sched_info.get("workers", {})
    n_workers = len(workers)
    total_cores = sum(w.get("nthreads", 0) for w in workers.values())
    total_mem_bytes = sum(w.get("memory_limit", 0) for w in workers.values())
    total_mem_gb = total_mem_bytes / (1024 ** 3) if total_mem_bytes else 0.0
    print(f"\n✅ Workers connected: {n_workers}")
    if n_workers < total_workers_expected:
        print(f"⚠️ Requested {total_workers_expected} worker(s), but only {n_workers} connected.")
    print(f"   Summary: {n_workers} worker(s) — total cores={total_cores}, total mem≈{total_mem_gb:.2f}GB\n")

    if verbose > 0:
        print_worker_details(workers, client)

    return cluster, client


def print_worker_details(workers: Dict[str, Any], client: Client) -> None:
    def _probe_worker():
        import os, platform
        info = {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"),
        }
        try:
            import jax
            info["jax_devices"] = [str(d) for d in jax.devices()]
        except Exception as e:
            info["jax_devices"] = f"jax not available ({type(e).__name__})"
        return info

    probe_results = client.run(_probe_worker)

    print("--- Per-worker details ---")
    header = f"{'addr':<40}  {'host':<15}  {'pid':<6}  {'nthreads':<8}  {'mem_GB':<7}  {'CUDA_VISIBLE':<12}  {'jax_devices'}"
    print(header)
    print("-" * len(header))
    for addr, meta in workers.items():
        probe = probe_results.get(addr, {})
        hostname = probe.get("hostname", meta.get("host", addr.split(":")[0]))
        pid_probe = probe.get("pid", meta.get("pid", ""))
        nthreads = meta.get("nthreads", "")
        mem_gb = meta.get("memory_limit", 0) / (1024 ** 3) if meta.get("memory_limit", 0) else 0.0
        cuda_visible = probe.get("cuda_visible", "N/A")
        jax_devices = probe.get("jax_devices", "")
        print(f"{addr:<40}  {hostname:<15}  {pid_probe:<6}  {nthreads:<8}  {mem_gb:<7.2f}  {cuda_visible:<12}  {jax_devices}")
    print("-" * len(header) + "\n")


# --- Jupyter on workers -------------------------------------------------------

def start_jupyter_server_on_worker(
    port: Optional[int] = JUPYTER_PORT,
    runtime_dir_base: str = DEFAULT_JUPYTER_RUNTIME_BASE,
    timeout: float = 45.0,
    jupyter_exe: Optional[str] = None,
    token: Optional[str] = None,
    allow_origin: str = "*",
):
    import glob
    import json
    import os
    import re
    import select
    import socket
    import subprocess
    import sys
    import time
    import uuid
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    try:
        from dask.distributed import get_worker
        worker = get_worker()
        worker_meta = {"name": worker.name, "address": worker.address}
    except Exception:
        worker = None
        worker_meta = {"name": None, "address": None}

    token = token or uuid.uuid4().hex
    runtime_dir = os.path.join(runtime_dir_base, f"worker-{os.getpid()}")
    os.makedirs(runtime_dir, exist_ok=True)

    cmd = [jupyter_exe, "server"] if jupyter_exe else [sys.executable, "-m", "jupyter", "server"]
    env = os.environ.copy(); env.setdefault("JUPYTER_RUNTIME_DIR", runtime_dir)
    port_arg = "--port=0" if port is None else f"--port={int(port)}"

    cmd += [
        "--no-browser", "--ip=0.0.0.0", port_arg,
        "--ServerApp.port_retries=0", f"--ServerApp.token={token}",
        "--ServerApp.allow_remote_access=True", f"--ServerApp.allow_origin={allow_origin}",
        f"--ServerApp.runtime_dir={runtime_dir}",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", env=env)

    url = None; parsed_url = None; captured: List[str] = []
    url_re = re.compile(r"(https?://[^\s,;]+)")
    fd = proc.stdout.fileno() if proc.stdout is not None else None
    t0 = time.time()

    def _worker_host_ip() -> str:
        try:
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception:
            return "127.0.0.1"

    while True:
        line = ""
        if fd is not None:
            rlist, _, _ = select.select([fd], [], [], 0.5)
            if rlist:
                line = proc.stdout.readline()
        if line:
            line = line.rstrip("\n"); captured.append(line)
            m = url_re.search(line)
            if m:
                candidate = m.group(1)
                parsed = urlparse(candidate)
                qs = dict(parse_qsl(parsed.query, keep_blank_values=True)); qs["token"] = token
                host_ip = _worker_host_ip(); port_part = f":{parsed.port}" if parsed.port else ""
                netloc = f"{host_ip}{port_part}" if parsed.hostname in {"127.0.0.1", "localhost"} else parsed.netloc
                new_query = urlencode(qs)
                parsed = parsed._replace(netloc=netloc, query=new_query)
                url = urlunparse(parsed); parsed_url = parsed
                break
        if proc.poll() is not None:
            if proc.stdout:
                remainder = proc.stdout.read()
                if remainder: captured.append(remainder)
            break
        if (time.time() - t0) > timeout:
            break

    if not url:
        files = list(Path(runtime_dir).glob("nbserver-*.json"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            try:
                data = json.loads(files[0].read_text("utf-8"))
                candidate = data.get("url") or data.get("base_url")
                if candidate:
                    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
                    parsed = urlparse(candidate)
                    host_ip = _worker_host_ip()
                    port_part = f":{parsed.port}" if parsed.port else ""
                    netloc = f"{host_ip}{port_part}" if parsed.hostname in {"127.0.0.1", "localhost"} else parsed.netloc
                    qs = dict(parse_qsl(parsed.query, keep_blank_values=True)); qs.setdefault("token", data.get("token", token))
                    parsed = parsed._replace(netloc=netloc, query=urlencode(qs))
                    url = urlunparse(parsed); parsed_url = parsed
            except Exception as exc:
                captured.append(f"[runtime-json-error] {exc}")

    if url:
        return {
            "status": "ok", "pid": proc.pid, "url": url, "token": token,
            "host": (parsed_url.hostname if parsed_url else None), "port": (parsed_url.port if parsed_url else None),
            "worker": worker_meta, "runtime_dir": runtime_dir, "cmd": cmd, "log_tail": captured[-20:],
        }

    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    return {"status": "error", "error": "no-url", "pid": proc.pid, "worker": worker_meta, "log_tail": captured[-40:]}


def stop_jupyter_server_on_worker(pid: int, sig: int = signal.SIGTERM, wait: float = 5.0):
    try:
        os.kill(int(pid), sig)
    except ProcessLookupError:
        return {"status": "not-found", "pid": pid}
    except PermissionError as exc:
        return {"status": "error", "pid": pid, "error": str(exc)}

    if wait and wait > 0:
        deadline = time.time() + float(wait)
        while time.time() < deadline:
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return {"status": "terminated", "pid": pid}
            time.sleep(0.2)
        return {"status": "alive", "pid": pid}
    return {"status": "signaled", "pid": pid}


def execute_launch(
    config: Dict[str, Any],
    verbosity: int,
    *,
    require_confirmation: bool = True,
    prompt_to_save: bool = False,
) -> None:
    launch_cfg = dict(config)

    num_jobs = max(1, int(launch_cfg.get("num_jobs") or 1))
    processes = max(1, int(launch_cfg.get("processes") or 1))
    cores_per_process = max(1, int(launch_cfg.get("cores_per_process") or 1))
    threads_per_worker = int(launch_cfg.get("threads_per_worker") or cores_per_process or 1)
    threads_per_worker = max(1, threads_per_worker)
    walltime = launch_cfg.get("walltime") or WALLTIME
    memory = launch_cfg.get("memory") or f"{MEMORY_GB_DEFAULT}GB"
    selected_partition = launch_cfg.get("partition")
    nodelist = launch_cfg.get("nodelist") or None
    num_gpus = int(launch_cfg.get("num_gpus") or 0)
    gres = launch_cfg.get("gres")
    if not gres and num_gpus > 0:
        gres = f"gpu:{num_gpus}"
    launch_kernels = bool(launch_cfg.get("launch_kernels"))

    launch_cfg.update(
        {
            "num_jobs": num_jobs,
            "processes": processes,
            "cores_per_process": cores_per_process,
            "threads_per_worker": threads_per_worker,
            "walltime": walltime,
            "memory": memory,
            "partition": selected_partition,
            "nodelist": nodelist,
            "num_gpus": num_gpus,
            "gres": gres,
            "launch_kernels": launch_kernels,
        }
    )

    print_launch_summary(launch_cfg)
    if require_confirmation:
        if input("Proceed with launch? (y/n): ").lower() != 'y':
            print("🚫 Launch aborted.")
            return

    if prompt_to_save:
        maybe_save_named_config(launch_cfg)

    save_last_launch_config(launch_cfg)

    cluster = client = None
    started_jupyter_servers: List[Dict[str, Any]] = []
    try:
        cluster, client = get_cluster(
            local=False,
            queue=selected_partition,
            num_jobs=num_jobs,
            processes=processes,
            threads_per_worker=threads_per_worker,
            cores=cores_per_process,
            memory=memory,
            walltime=walltime,
            nodelist=nodelist,
            gres=gres,
            verbose=verbosity,
        )
        print("\n✅ Dask cluster is running!")
        print(f"   Reconnect with: Client('{client.scheduler.address}')")
        try:
            SCHEDULER_ADDRESS_FILE.write_text(client.scheduler.address)
            print(f"   Scheduler address saved to {SCHEDULER_ADDRESS_FILE}")
        except Exception:
            pass

        if launch_kernels:
            print("\n🔌 Starting a Jupyter server on each worker via Dask…")
            worker_infos = client.scheduler_info().get("workers", {})
            worker_addresses = list(worker_infos.keys())
            if not worker_addresses:
                print("⚠️  No workers available to launch Jupyter kernels.")
            else:
                session_tag = uuid.uuid4().hex[:8]
                for idx, worker_addr in enumerate(worker_addresses):
                    worker_port = JUPYTER_PORT + idx if JUPYTER_PORT is not None else None
                    runtime_base = os.path.join(DEFAULT_JUPYTER_RUNTIME_BASE, f"session-{session_tag}", f"worker-{idx}")
                    future = client.submit(
                        start_jupyter_server_on_worker,
                        port=worker_port,
                        runtime_dir_base=runtime_base,
                        timeout=60.0,
                        pure=False,
                        workers=[worker_addr],
                        allow_other_workers=False,
                    )
                    try:
                        info = future.result(timeout=180)
                    except Exception as exc:
                        print(f"❌ Worker {worker_addr}: exception while starting Jupyter -> {exc}")
                        continue

                    if info.get("status") == "ok" and info.get("url"):
                        host_display = info.get("host") or worker_addr
                        print(f"✅ Worker {worker_addr} ({host_display})")
                        print(f"   PID: {info.get('pid')} | URL: {info.get('url')}")
                        started_jupyter_servers.append({"worker": worker_addr, "pid": info.get("pid")})
                    else:
                        print(f"❌ Worker {worker_addr}: {info.get('error', 'failed to start Jupyter')}")
                        log_tail = info.get("log_tail") or []
                        if log_tail:
                            print("   Log tail:")
                            for line in log_tail[-10:]:
                                print(f"   {line}")

        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupt received. Shutting down…")
    except Exception as e:
        print(f"\n💥 Error during cluster setup: {e}")
    finally:
        print("\n🛑 Shutting down cluster and client…")
        if client and started_jupyter_servers:
            print("🧹 Stopping remote Jupyter servers…")
            for entry in started_jupyter_servers:
                worker, pid = entry.get("worker"), entry.get("pid")
                try:
                    fut = client.submit(stop_jupyter_server_on_worker, pid, pure=False, workers=[worker], allow_other_workers=False)
                    res = fut.result(timeout=15)
                    status = res.get("status")
                except Exception as exc:
                    status = f"error: {exc}"
                print(f"   Worker {worker or 'unknown'} pid {pid}: {status}")
        if client:
            try:
                client.close()
            except Exception:
                pass
        if cluster:
            try:
                cluster.close()
            except Exception:
                pass
        print("✅ Cleanup complete.")


# --- Interactive UI -----------------------------------------------------------

def prompt_for_value(prompt_text: str, default: Any, type_converter=str):
    while True:
        value = input(f"➡️  {prompt_text} (default: {default}): ") or str(default)
        try:
            return type_converter(value)
        except ValueError:
            print(f"❌ Invalid input. Please enter a value of type {type_converter.__name__}.")


def main_interactive(verbosity: int) -> None:
    show_admin = verbosity >= 2
    include_jobs = verbosity >= 2
    include_job_resources = verbosity >= 4

    last_launch_config = load_last_launch_config()
    if last_launch_config:
        saved_at = format_saved_timestamp(last_launch_config.get("saved_at"))
        print("\n🕘 Previous launch configuration detected.")
        if saved_at:
            print(f"   Saved at: {saved_at}")
        print_launch_summary(last_launch_config, title="📁 LAST CONFIGURATION")
        reuse = input("↩️  Re-launch the previous configuration? (y/n): ").strip().lower()
        if reuse == 'y':
            print("\n🔁 Re-launching previous configuration…")
            execute_launch(last_launch_config, verbosity, require_confirmation=False)
            return

    print("🔎 Checking for available SLURM resources…")
    parts = get_partitions(show_all=show_admin)
    if not parts:
        print("\n😔 No suitable partitions found. Try again later or check permissions.")
        return

    print("\n✅ Partitions:\n")
    print("=" * 90)

    # Collect detailed node info once
    all_nodes = sorted({n for p in parts.values() for n in p["nodes"]})
    node_details_all = get_detailed_node_info(
        all_nodes,
        include_jobs=include_jobs,
        include_job_resources=include_job_resources,
    )

    part_index: Dict[int, str] = {}
    partition_by_name: Dict[str, Dict[str, Any]] = {}
    for idx, pdata in sorted(parts.items()):
        pname = pdata['partition']
        nodes = pdata['nodes']
        cpu_info = pdata['cpu_info']
        header = colorize(f"[{idx}] Partition: {pname}", Style.BRIGHT, Fore.CYAN)
        print(header)
        if cpu_info:
            avail, total, alloc = cpu_info['idle'], cpu_info['total'], cpu_info['allocated']
            line = f"    CPUs: {avail}/{total} available ({alloc} in use)"
            print(colorize(line, Style.BRIGHT, Fore.WHITE) if COLOR_ENABLED else line)

        # Summarize by config (RAM/CPU/GPU)
        configs: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        for n in nodes:
            nd = node_details_all.get(n)
            if nd is None:
                continue
            mem_gb = int((nd['memory_gb'] or 0))
            cpus = int(nd['cpus_total'] or 0)
            gpus = int(nd['gpus_total'] or 0)
            configs[(mem_gb, cpus, gpus)].append(n)

        for (mem_gb, cpus, gpus), bucket_nodes in sorted(configs.items(), key=lambda x: (x[0][0], x[0][1]), reverse=True):
            infos = [node_details_all[n] for n in bucket_nodes]
            available_count = sum(1 for i in infos if i['state'] in {'idle', 'mixed', 'mix'})
            gpu_in_use = sum(i['gpus_in_use'] for i in infos)
            gpu_total = sum(i['gpus_total'] for i in infos)
            gpu_counts = aggregate_gpu_counts(infos) if gpus else []
            gpu_type_desc = format_gpu_counts(gpu_counts) if gpu_counts else ""
            users_per_node = [len(i.get('users', [])) for i in infos]
            min_users = min(users_per_node) if users_per_node else 0
            max_users = max(users_per_node) if users_per_node else 0
            bullet = colorize("•", Fore.YELLOW, Style.BRIGHT) if COLOR_ENABLED else "•"
            detail_parts = [
                f"{available_count} available",
                f"GPUs in use: {gpu_in_use}/{gpu_total}",
            ]
            if gpus:
                gpu_label = gpu_type_desc or "unknown"
                detail_parts.append(f"GPU type: {emphasize(gpu_label, Fore.MAGENTA)}")
            if users_per_node:
                detail_parts.append(f"users/node: {min_users}-{max_users}")
            summary = (
                f"      {bullet} {len(bucket_nodes)} nodes: {mem_gb}GB RAM, {cpus} CPUs, {gpus} GPUs "
                f"({'; '.join(detail_parts)})"
            )
            print(summary)

            if verbosity >= 1 and len(bucket_nodes) <= 10:
                for node in sorted(bucket_nodes):
                    i = node_details_all[node]
                    state_good = i['state'] in {'idle', 'mixed', 'mix'}
                    state_icon = colorize("✓", Fore.GREEN) if state_good else colorize("✗", Fore.RED)
                    if not COLOR_ENABLED:
                        state_icon = "✓" if state_good else "✗"
                    mem_str = f"{int(i['memory_gb'] or 0)}GB" if i['memory_gb'] else "N/A"
                    cpu_str = f"{i['cpus_alloc'] or 0}/{i['cpus_total'] or 'N/A'}"
                    gpu_usage = f"{i['gpus_in_use']}/{i['gpus_total']}"
                    gpu_label_text = format_gpu_labels(i.get('gpu_labels', []))
                    if not gpu_label_text and i['gpus_total']:
                        gpu_label_text = "unknown"
                    if gpu_label_text:
                        gpu_usage = f"{gpu_usage} ({emphasize(gpu_label_text, Fore.MAGENTA)})"
                    print(f"        {state_icon} {node:<20} ({i['state']:<6}, RAM: {mem_str}, CPUs: {cpu_str}, GPUs: {gpu_usage})")
                    detail_indent = " " * 12
                    if verbosity >= 2:
                        job_records = [rec for rec in (i.get('job_records') or []) if isinstance(rec, dict)]
                        if job_records:
                            if verbosity >= 4:
                                print(f"{detail_indent}jobs:")
                                for rec in job_records:
                                    name = rec.get('name') or "-"
                                    jid = rec.get('id')
                                    label = f"{name} (#{jid})" if jid else name
                                    time_info = _format_job_time_info(rec.get("elapsed"), rec.get("time_left"))
                                    resource_summary = (rec.get('resources') or {}).get('summary')
                                    detail_bits: List[str] = []
                                    if resource_summary:
                                        detail_bits.append(resource_summary)
                                    if time_info:
                                        detail_bits.append(f"time {time_info}")
                                    if detail_bits:
                                        print(f"{detail_indent}  - {label} | {' | '.join(detail_bits)}")
                                    else:
                                        print(f"{detail_indent}  - {label}")
                            else:
                                job_summaries: List[str] = []
                                for rec in job_records:
                                    name = rec.get('name') or "-"
                                    jid = rec.get('id')
                                    label = f"{name} (#{jid})" if jid else name
                                    time_info = _format_job_time_info(rec.get("elapsed"), rec.get("time_left"))
                                    if time_info:
                                        label = f"{label} [{time_info}]"
                                    job_summaries.append(label)
                                if job_summaries:
                                    print(f"{detail_indent}jobs: {', '.join(job_summaries)}")
                    if verbosity >= 3:
                        users_list = i.get('users', [])
                        if users_list:
                            print(f"{detail_indent}users: {', '.join(users_list)}")

        print("-" * 90)
        part_index[idx] = pname
        partition_by_name[pname] = pdata

    print("\n[0] Use default cluster partition (no explicit selection)")

    # 1) Select partition
    selected_partition: Optional[str] = None
    while True:
        raw_choice = input("\n➡️  Enter the number of your choice (press Enter for default): ").strip()
        if raw_choice in {"", "0"}:
            selected_partition = None
            break
        try:
            choice = int(raw_choice)
            if choice in part_index:
                selected_partition = part_index[choice]
                break
            else:
                print(f"❌ Invalid choice. Please select a number from 1 to {len(part_index)}, or press Enter for the default.")
        except ValueError:
            print("❌ Please enter a valid number or press Enter for the default.")

    if selected_partition:
        print(f"\n✅ You selected partition: '{selected_partition}'")
    else:
        print("\n✅ Using SLURM's default partition (no explicit queue).")

    launch_kernels_choice = input("\n🧪 Start a Jupyter server on each worker after launch? (y/n): ").strip().lower()
    launch_kernels = launch_kernels_choice == 'y'
    if launch_kernels:
        print(
            "\nℹ️  Defaults updated for interactive Jupyter use:\n"
            "   • SLURM jobs: 1\n   • CPU cores per process: 16\n   • No additional thread limits\n"
        )

    # 2) Parameters
    walltime = prompt_for_value("Enter walltime [HH:MM:SS]", WALLTIME)
    default_num_jobs = 1 if launch_kernels else NUM_JOBS
    default_processes = PROCESSES_PER_WORKER
    default_cores_per_process = 16 if launch_kernels else CORES

    num_jobs = prompt_for_value("Enter number of SLURM jobs", default_num_jobs, int)
    processes = prompt_for_value("Enter processes per job", default_processes, int)
    cores_per_process = prompt_for_value("Enter CPU cores per process", default_cores_per_process, int)

    print(
        "\n🧵 Threads limit: Optionally limit threads per Dask worker.\n"
        "Enter 0 for no limit (uses 'cores per process')."
    )
    threads_limit = prompt_for_value("Limit threads per worker (0 for no limit)", 0, int)
    memory_gb = prompt_for_value("Enter memory per job in GB (e.g., 32)", MEMORY_GB_DEFAULT, int)
    memory = f"{memory_gb}GB"

    threads_per_worker = min(cores_per_process, threads_limit) if threads_limit and threads_limit > 0 else cores_per_process
    total_cores_for_job = threads_per_worker * processes

    # 3) Node selection within chosen partition
    nodelist: Optional[str] = None
    if selected_partition:
        selected_partition_data = partition_by_name.get(selected_partition)
        if not selected_partition_data:
            print("⚠️  Unable to load node information for the selected partition; skipping node selection.")
        else:
            nodes_in_part = selected_partition_data['nodes']
            nodes_info_part = [node_details_all[n] for n in nodes_in_part if n in node_details_all]

            print(f"\n📍 Detailed node list for partition '{selected_partition}':")
            print("=" * 120)
            print(f"{'Idx':<5} {'Node':<22} {'State':<10} {'Memory':<12} {'CPUs (used/total)':<20} {'GPUs (used/total)':<22} {'Users':<7}")
            print("-" * 120)

            available_for_selection: List[Tuple[int, str]] = []
            for idx, i in enumerate(sorted(nodes_info_part, key=lambda x: x['state'] + str(x.get('cpus_total') or 0)), start=1):
                mem_str = f"{int(i['memory_gb'] or 0):.0f}GB" if i['memory_gb'] is not None else "N/A"
                cpu_str = f"{i['cpus_alloc'] or 0}/{i['cpus_total'] or 'N/A'}"
                gpu_str = f"{i['gpus_in_use']}/{i['gpus_total']}"
                gpu_label_text = format_gpu_labels(i.get('gpu_labels', []))
                if not gpu_label_text and i['gpus_total']:
                    gpu_label_text = "unknown"
                if gpu_label_text:
                    gpu_str = f"{gpu_str} ({emphasize(gpu_label_text, Fore.MAGENTA)})"
                is_available = i['state'] in {'idle', 'mixed', 'mix'}
                marker = "✅" if is_available else " "
                user_count = len(i.get('users', []))
                print(f"{idx:<5} {marker} {nodes_in_part[idx-1]:<20} {i['state']:<10} {mem_str:<12} {cpu_str:<20} {gpu_str:<22} {user_count:<7}")
                detail_indent = " " * 8
                if verbosity >= 2:
                    job_records = [rec for rec in (i.get('job_records') or []) if isinstance(rec, dict)]
                    if job_records:
                        if verbosity >= 4:
                            print(f"{detail_indent}jobs:")
                            for rec in job_records:
                                name = rec.get('name') or "-"
                                jid = rec.get('id')
                                label = f"{name} (#{jid})" if jid else name
                                time_info = _format_job_time_info(rec.get("elapsed"), rec.get("time_left"))
                                resource_summary = (rec.get('resources') or {}).get('summary')
                                detail_bits: List[str] = []
                                if resource_summary:
                                    detail_bits.append(resource_summary)
                                if time_info:
                                    detail_bits.append(f"time {time_info}")
                                if detail_bits:
                                    print(f"{detail_indent}  - {label} | {' | '.join(detail_bits)}")
                                else:
                                    print(f"{detail_indent}  - {label}")
                        else:
                            job_summaries: List[str] = []
                            for rec in job_records:
                                name = rec.get('name') or "-"
                                jid = rec.get('id')
                                label = f"{name} (#{jid})" if jid else name
                                time_info = _format_job_time_info(rec.get("elapsed"), rec.get("time_left"))
                                if time_info:
                                    label = f"{label} [{time_info}]"
                                job_summaries.append(label)
                            if job_summaries:
                                print(f"{detail_indent}jobs: {', '.join(job_summaries)}")
                if verbosity >= 3:
                    users_list = i.get('users', [])
                    if users_list:
                        print(f"{detail_indent}users: {', '.join(users_list)}")
                if is_available:
                    available_for_selection.append((idx, nodes_in_part[idx-1]))

            print("=" * 120)
            print(f"✓ = Available for job submission ({len(available_for_selection)} nodes)\n")

            nodelist_choice = input("Enter index of specific node to use (leave empty for any): ").strip()
            if nodelist_choice:
                try:
                    nidx = int(nodelist_choice)
                    if 1 <= nidx <= len(nodes_in_part):
                        nodelist = nodes_in_part[nidx - 1]
                    else:
                        print("⚠️  Invalid index; proceeding without a specific node.")
                except ValueError:
                    print("⚠️  Invalid input; proceeding without a specific node.")
    else:
        print("\nℹ️  Skipping node-specific selection; SLURM will apply its default partition rules.")

    num_gpus = prompt_for_value("Enter number of GPUs per process (0 for CPU-only)", 0, int)
    gres = f"gpu:{num_gpus}" if num_gpus > 0 else None

    launch_config = {
        "partition": selected_partition,
        "walltime": walltime,
        "num_jobs": num_jobs,
        "processes": processes,
        "cores_per_process": cores_per_process,
        "threads_per_worker": threads_per_worker,
        "memory": memory,
        "nodelist": nodelist,
        "num_gpus": num_gpus,
        "gres": gres,
        "launch_kernels": launch_kernels,
    }
    execute_launch(launch_config, verbosity, prompt_to_save=True)


# --- CLI ---------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Dask-on-Slurm launcher (CLI-based Slurm introspection)")
    p.add_argument("--local", action="store_true", help="Launch a local Dask cluster instead of Slurm")
    p.add_argument("--hud", action="store_true", help="Start the browser HUD for SLURM usage")
    p.add_argument("--hud-host", default="0.0.0.0", help="HUD host to bind")
    p.add_argument("--hud-port", type=int, default=8765, help="HUD port to bind")
    p.add_argument("--hud-refresh", type=int, default=600, help="HUD refresh interval (seconds)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv, -vvv, -vvvv)")
    p.add_argument("config_name", nargs="?", help="Optional saved configuration name to launch directly")
    return p.parse_args(argv)


def main_cli() -> None:
    args = parse_args()
    if args.hud:
        from slurmcli.hud import run_hud
        run_hud(
            host=args.hud_host,
            port=args.hud_port,
            refresh_seconds=args.hud_refresh,
            verbosity=args.verbose,
        )
        return
    if args.local:
        get_cluster(local=True, verbose=args.verbose)
        print("\n(local cluster running; press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nStopping…")
        return

    if args.config_name:
        config = load_named_launch_config(args.config_name)
        if not config:
            print(f"❌ Saved configuration '{args.config_name}' not found or invalid.")
            return
        print(f"📂 Launching saved configuration '{args.config_name}'")
        execute_launch(config, args.verbose, require_confirmation=False)
        return

    main_interactive(args.verbose)


if __name__ == "__main__":
    main_cli()
