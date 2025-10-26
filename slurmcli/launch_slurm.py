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
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Third-party (kept for parity with original script)
import numpy as np  # noqa: F401
import jax, jax.numpy as jnp  # noqa: F401
from dask import delayed  # noqa: F401
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

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


# --- Helper parsers for CLI output -------------------------------------------

def parse_cpu_string(cpu_str: str) -> Optional[Dict[str, int]]:
    try:
        parts = cpu_str.split('/')
        if len(parts) == 4:
            allocated, idle, other, total = map(int, parts)
            return {"allocated": allocated, "idle": idle, "other": other, "total": total}
    except Exception:
        pass
    return None


def parse_memory_string(mem_str: str) -> Optional[float]:
    mem_str = mem_str.strip().rstrip('+')
    try:
        u = mem_str.upper()
        if 'T' in u:
            return float(u.replace('T', '').replace('B', '')) * 1024
        if 'G' in u:
            return float(u.replace('G', '').replace('B', ''))
        if 'M' in u:
            return float(u.replace('M', '').replace('B', '')) / 1024
        return float(mem_str) / 1024
    except Exception:
        return None


def expand_nodelist(nodelist: Optional[str]) -> List[str]:
    if not nodelist or nodelist in {"(null)", "None"}:
        return []
    try:
        result = subprocess.run([
            "scontrol", "show", "hostnames", str(nodelist)
        ], check=True, capture_output=True, text=True, timeout=5)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        return [nodelist]


def extract_gpu_total(gres_value: Optional[str]) -> int:
    if not gres_value:
        return 0
    gres_value = gres_value.strip()
    if not gres_value or gres_value in {"(null)", "N/A", "none"}:
        return 0
    total = 0
    for entry in gres_value.split(','):
        entry = entry.strip()
        if "gpu" not in entry.lower():
            continue
        entry_core = entry.split('(', 1)[0]
        parts = [part for part in entry_core.split(':') if part]
        for part in reversed(parts):
            m = re.search(r"(\d+)$", part)
            if m:
                total += int(m.group(1))
                break
    return total


def extract_gpu_labels(gres_value: Optional[str]) -> List[str]:
    labels: List[str] = []
    if not gres_value:
        return labels

    seen: set[str] = set()
    for raw_entry in gres_value.split(','):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "gpu" not in entry.lower():
            continue
        entry_core = entry.split('(', 1)[0]
        parts = [part for part in entry_core.split(':') if part]
        if parts and parts[0].lower() == "gpu":
            parts = parts[1:]
        if not parts:
            continue
        if parts and re.fullmatch(r'\d+', parts[-1]):
            parts = parts[:-1]
        label = ':'.join(parts).strip()
        if not label:
            label = "GPU"
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def format_gpu_labels(labels: Iterable[str]) -> str:
    seen: set[str] = set()
    ordered: List[str] = []
    for label in labels:
        label_clean = label.strip()
        if not label_clean:
            continue
        key = label_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(label_clean)
    return ", ".join(ordered)


def aggregate_gpu_counts(infos: Iterable[Dict[str, Any]]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    order: List[str] = []
    for info in infos:
        total = int(info.get('gpus_total') or 0)
        labels = info.get('gpu_labels') or []
        if not labels:
            if total:
                label = "unknown"
                if label not in counts:
                    counts[label] = 0
                    order.append(label)
                counts[label] += total
            continue
        if len(labels) == 1:
            label = labels[0]
            if label not in counts:
                counts[label] = 0
                order.append(label)
            counts[label] += total or 1
            continue
        distributed_total = total if total else len(labels)
        share = max(1, distributed_total // len(labels))
        for label in labels:
            if label not in counts:
                counts[label] = 0
                order.append(label)
            counts[label] += share
    return [(label, counts[label]) for label in order]


def format_gpu_counts(counts: Iterable[Tuple[str, int]]) -> str:
    parts: List[str] = []
    for label, count in counts:
        display_label = "unknown" if label == "unknown" else label
        if count > 0:
            parts.append(f"{count}x {display_label}")
        else:
            parts.append(display_label)
    return ", ".join(parts)


# --- Jobs & users per node ----------------------------------------------------

def get_jobs_by_node() -> Dict[str, List[Dict[str, Any]]]:
    """Return mapping node -> list of jobs {id,user} currently assigned there."""
    try:
        # %i jobid, %u user, %N nodelist
        result = subprocess.run(
            "squeue -h -o '%i|%u|%j|%N'",
            shell=True, check=True, capture_output=True, text=True, timeout=8
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

    by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in result.stdout.strip().splitlines():
        if '|' not in line:
            continue
        parts = line.split('|', 3)
        if len(parts) < 4:
            continue
        jid_s, user, job_name, nodelist = parts
        jid = None
        try:
            jid = int(jid_s)
        except Exception:
            pass
        job_name = job_name.strip() or None
        for node in expand_nodelist(nodelist.strip()):
            by_node[node].append({"id": jid, "user": user, "name": job_name})
    return by_node


def get_node_user_map() -> Dict[str, set]:
    try:
        result = subprocess.run(
            "squeue -h -o '%u|%N'",
            shell=True, check=True, capture_output=True, text=True, timeout=8,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

    node_users: Dict[str, set] = defaultdict(set)
    expansion_cache: Dict[str, List[str]] = {}

    for line in result.stdout.strip().splitlines():
        if '|' not in line:
            continue
        user, nodelist = line.split('|', 1)
        nodelist = nodelist.strip()
        if not nodelist or nodelist in {"(null)", "None"}:
            continue
        if nodelist not in expansion_cache:
            expansion_cache[nodelist] = expand_nodelist(nodelist)
        for node in expansion_cache[nodelist]:
            node_users[node].add(user.strip())

    return node_users


# --- Detailed node info via scontrol -----------------------------------------

def query_users_for_node(node: str) -> List[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-w", node, "-o", "%u"],
            check=True, capture_output=True, text=True, timeout=6,
        )
        users = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        return sorted(users)
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_detailed_node_info(nodes: Iterable[str], include_jobs: bool = False) -> Dict[str, Dict[str, Any]]:
    node_users_map = get_node_user_map() if include_jobs else {}
    jobs_by_node = get_jobs_by_node() if include_jobs else {}

    node_info: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        try:
            cmd = f"scontrol show node {node}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=8)
            output = result.stdout

            info = {
                'memory_gb': None,
                'cpus_total': None,
                'cpus_alloc': None,
                'gpus': '0',
                'gpus_total': 0,
                'gpus_in_use': 0,
                'gpu_labels': [],
                'state': 'unknown',
                'users': [],
                'user_count': 0,
                'jobs': [],
            }

            for line in output.split('\n'):
                if 'RealMemory=' in line:
                    for part in line.split():
                        if part.startswith('RealMemory='):
                            try:
                                mem_mb = int(part.split('=')[1])
                                info['memory_gb'] = mem_mb / 1024
                            except Exception:
                                pass
                if 'CPUTot=' in line:
                    for part in line.split():
                        if part.startswith('CPUTot='):
                            try:
                                info['cpus_total'] = int(part.split('=')[1])
                            except Exception:
                                pass
                        if part.startswith('CPUAlloc='):
                            try:
                                info['cpus_alloc'] = int(part.split('=')[1])
                            except Exception:
                                pass
                if 'Gres=' in line:
                    for part in line.split():
                        if part.startswith('Gres='):
                            gres = part.split('=', 1)[1]
                            info['gpus'] = gres
                            info['gpus_total'] = extract_gpu_total(gres)
                            info['gpu_labels'] = extract_gpu_labels(gres)
                if 'GresUsed=' in line:
                    for part in line.split():
                        if part.startswith('GresUsed='):
                            gres_used = part.split('=', 1)[1]
                            info['gpus_in_use'] = extract_gpu_total(gres_used)
                            break
                if info['gpus_in_use'] == 0 and 'AllocTRES=' in line:
                    for part in line.split():
                        if part.startswith('AllocTRES='):
                            tres = part.split('=', 1)[1]
                            m = re.search(r'gres/gpu=(\d+)', tres)
                            if m:
                                info['gpus_in_use'] = int(m.group(1))
                            break
                if 'State=' in line:
                    for part in line.split():
                        if part.startswith('State='):
                            info['state'] = part.split('=')[1].split('+')[0].lower()

            # Users & jobs
            if include_jobs:
                users = sorted(node_users_map.get(node, set()))
                if not users and (info.get('gpus_in_use', 0) or info.get('cpus_alloc', 0)):
                    users = query_users_for_node(node)
                info['users'] = users
                info['user_count'] = len(users)
                job_records = [
                    {
                        "id": j.get("id"),
                        "user": j.get("user"),
                        "name": j.get("name"),
                    }
                    for j in jobs_by_node.get(node, [])
                ]
                info['job_records'] = job_records
                info['jobs'] = job_records

            node_info[node] = info
        except Exception:
            continue

    return node_info


# --- Partition discovery via sinfo -------------------------------------------

def get_partitions(show_all: bool = False) -> Optional[Dict[int, Dict[str, Any]]]:
    """Parse `sinfo` for partitions and group nodes; optionally include admin/hidden with -a."""
    print("🔎 Checking for available SLURM resources…")
    try:
        # %P partition, %t state, %C ALLOC/IDLE/OTHER/TOTAL, %G GRES, %N nodelist
        base = "sinfo -h -o '%P|%t|%C|%G|%N'"
        cmd = ("sinfo -a -h -o '%P|%t|%C|%G|%N'") if show_all else base
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')

        part_data = defaultdict(lambda: {"states": set(), "cpu_stats": [], "gres": set(), "nodes": set()})
        expansion_cache: Dict[str, List[str]] = {}

        for line in lines:
            parts = line.split('|')
            if len(parts) < 5:
                continue
            partition, state, cpus, gres, nodes = parts[:5]
            partition = partition.strip('*')

            # Default view: only include partitions that have idle or mixed nodes
            # -vv (show_all=True): include every partition/state line
            include = show_all or (("idle" in state.lower()) or ("mix" in state.lower()))
            if not include:
                continue

            part_data[partition]['states'].add(state)
            part_data[partition]['cpu_stats'].append(cpus)
            part_data[partition]['gres'].add(gres)

            for node_expr in nodes.split(','):
                node_expr = node_expr.strip()
                if not node_expr:
                    continue
                if node_expr not in expansion_cache:
                    expansion_cache[node_expr] = expand_nodelist(node_expr)
                for expanded in expansion_cache[node_expr]:
                    part_data[partition]['nodes'].add(expanded)

        if not part_data:
            return None

        # Build compact choices map
        choices: Dict[int, Dict[str, Any]] = {}
        for i, (partition, pdata) in enumerate(part_data.items(), start=1):
            choices[i] = {
                "partition": partition,
                "states": sorted(pdata['states']),
                "cpu_info": parse_cpu_string(pdata['cpu_stats'][0]) if pdata['cpu_stats'] else None,
                "nodes": sorted(pdata['nodes']),
            }
        return choices

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Error: unable to run sinfo ({e})")
        return None


# --- Dask cluster launch ------------------------------------------------------
WALLTIME = "01:00:00"
MEMORY_GB_DEFAULT = 32
PROCESSES_PER_WORKER = 1
CORES = 1
NUM_JOBS = 4


def get_cluster(
    *,
    local: bool = True,
    num_jobs: int = NUM_JOBS,
    processes: int = PROCESSES_PER_WORKER,
    threads_per_worker: Optional[int] = None,
    cores: int = CORES,
    memory: str = f"{MEMORY_GB_DEFAULT}GB",
    queue: str = "cpu",
    account: str = "gcnu-ac",
    walltime: str = WALLTIME,
    log_dir: str = "dask_logs",
    job_name: str = "tfl",
    venv_activate: Optional[str] = VENV_ACTIVATE,
    nodelist: Optional[str] = None,
    gres: Optional[str] = None,
    verbose: int = 0,
):
    if threads_per_worker is None:
        threads_per_worker = cores

    threads_per_worker = int(max(1, threads_per_worker))
    processes = int(max(1, processes))
    num_jobs = int(max(1, num_jobs))
    cores_for_job = threads_per_worker * processes

    if local:
        print("🚀 Starting a local Dask cluster…")
        n_workers_local = num_jobs * processes
        cluster = LocalCluster(
            n_workers=n_workers_local,
            threads_per_worker=threads_per_worker,
            memory_limit=memory,
            processes=True,
        )
        client = Client(cluster)
    else:
        print(f"🚀 Submitting {num_jobs} jobs to the '{queue}' SLURM partition…")
        os.makedirs(log_dir, exist_ok=True)
        prologue: List[str] = []
        if venv_activate:
            prologue.append(f"source {venv_activate}/bin/activate")

        job_extra_directives: List[str] = []
        if nodelist:
            job_extra_directives.append(f"--nodelist={nodelist}")
        if gres and 'gpu' in (gres or ""):
            try:
                num_gpus = int(gres.split(':')[-1])
                if num_gpus > 0:
                    job_extra_directives.append(f"--gpus-per-task={num_gpus}")
                    job_extra_directives.append(f"--gpu-bind=single:{num_gpus}")
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
            scheduler_options={
                "port": PORT_SLURM_SCHEDULER,
                "dashboard_address": f":{PORT_SLURM_DASHBOARD}",
            },
        )
        cluster.scale(n=num_jobs)
        client = Client(cluster)

    total_workers_expected = num_jobs * processes
    print("Cluster dashboard:", getattr(cluster, "dashboard_link", "n/a"))
    print("Scheduler address:", client.scheduler.address)
    print(f"\n⏳ Waiting for {total_workers_expected} workers to connect…")
    client.wait_for_workers(n_workers=total_workers_expected, timeout=300)

    sched_info = client.scheduler_info()
    workers = sched_info.get("workers", {})
    n_workers = len(workers)
    total_cores = sum(w.get("nthreads", 0) for w in workers.values())
    total_mem_bytes = sum(w.get("memory_limit", 0) for w in workers.values())
    total_mem_gb = total_mem_bytes / (1024 ** 3) if total_mem_bytes else 0.0
    print(f"\n✅ Workers connected: {n_workers}")
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

    parts = get_partitions(show_all=show_admin)
    if not parts:
        print("\n😔 No suitable partitions found. Try again later or check permissions.")
        return

    print("\n✅ Partitions:\n")
    print("=" * 90)

    # Collect detailed node info once
    all_nodes = sorted({n for p in parts.values() for n in p["nodes"]})
    node_details_all = get_detailed_node_info(all_nodes, include_jobs=include_jobs)

    part_index: Dict[int, str] = {}
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
                        job_records = i.get('job_records') or i.get('jobs') or []
                        job_summaries: List[str] = []
                        for rec in job_records:
                            if isinstance(rec, dict):
                                name = rec.get('name') or "-"
                                jid = rec.get('id')
                                job_summaries.append(f"{name} (#{jid})" if jid else name)
                            else:
                                job_summaries.append(str(rec))
                        if job_summaries:
                            print(f"{detail_indent}jobs: {', '.join(job_summaries)}")
                    if verbosity >= 3:
                        users_list = i.get('users', [])
                        if users_list:
                            print(f"{detail_indent}users: {', '.join(users_list)}")

        print("-" * 90)
        part_index[idx] = pname

    # 1) Select partition
    while True:
        try:
            choice = int(input("\n➡️  Enter the number of your choice: "))
            if choice in part_index:
                selected_partition = part_index[choice]
                break
            else:
                print(f"❌ Invalid choice. Please select a number from 1 to {len(part_index)}.")
        except ValueError:
            print("❌ Please enter a valid number.")

    print(f"\n✅ You selected partition: '{selected_partition}'")

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
    nodes_in_part = parts[[k for k, v in parts.items() if v['partition'] == selected_partition][0]]['nodes']
    nodes_info_part = [node_details_all[n] for n in nodes_in_part if n in node_details_all]

    print(f"\n📍 Detailed node list for partition '{selected_partition}':")
    print("=" * 120)
    print(f"{'Idx':<5} {'Node':<22} {'State':<10} {'Memory':<12} {'CPUs (used/total)':<20} {'GPUs (used/total)':<22} {'Users':<7}")
    print("-" * 120)

    available_for_selection: List[Tuple[int, str]] = []
    for idx, i in enumerate(sorted(nodes_info_part, key=lambda x: x['state']+str(x.get('cpus_total') or 0)), start=1):
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
            job_records = i.get('job_records') or i.get('jobs') or []
            job_summaries: List[str] = []
            for rec in job_records:
                if isinstance(rec, dict):
                    name = rec.get('name') or "-"
                    jid = rec.get('id')
                    job_summaries.append(f"{name} (#{jid})" if jid else name)
                else:
                    job_summaries.append(str(rec))
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
    nodelist = None
    if nodelist_choice:
        try:
            nidx = int(nodelist_choice)
            if 1 <= nidx <= len(nodes_in_part):
                nodelist = nodes_in_part[nidx - 1]
            else:
                print("⚠️  Invalid index; proceeding without a specific node.")
        except ValueError:
            print("⚠️  Invalid input; proceeding without a specific node.")

    num_gpus = prompt_for_value("Enter number of GPUs per process (0 for CPU-only)", 0, int)
    gres = f"gpu:{num_gpus}" if num_gpus > 0 else None

    # 4) Summary
    print("\n" + "=" * 50)
    print("📋 LAUNCH SUMMARY")
    print("=" * 50)
    print(f"  Partition / Queue: {selected_partition}")
    print(f"  Walltime:          {walltime}")
    print(f"  SLURM Jobs:        {num_jobs}")
    print(f"  Processes per job: {processes}")
    print(f"  Cores per process: {cores_per_process}")
    print(f"  Threads per process (actual): {threads_per_worker}")
    print(f"  Total cores per job: {total_cores_for_job}")
    print(f"  Memory per job:    {memory}")
    print(f"  Total workers:     {num_jobs * processes}")
    print(f"  Specific node:     {nodelist or 'Any available'}")
    print(f"  GPUs per process:  {num_gpus}")
    print(f"  Launch kernels:    {'Yes' if launch_kernels else 'No'}")
    print("=" * 50)

    if input("Proceed with launch? (y/n): ").lower() != 'y':
        print("🚫 Launch aborted.")
        return

    # 5) Launch Cluster
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
            Path(".dask_scheduler_address").write_text(client.scheduler.address)
            print("   Scheduler address saved to .dask_scheduler_address")
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


# --- CLI ---------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Dask-on-Slurm launcher (CLI-based Slurm introspection)")
    p.add_argument("--local", action="store_true", help="Launch a local Dask cluster instead of Slurm")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv, -vvv)")
    return p.parse_args(argv)


def main_cli() -> None:
    args = parse_args()
    if args.local:
        get_cluster(local=True, verbose=args.verbose)
        print("\n(local cluster running; press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nStopping…")
        return

    main_interactive(args.verbose)


if __name__ == "__main__":
    main_cli()
