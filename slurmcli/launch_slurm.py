#!/usr/bin/env python3

"""
Notes JB:
- always use single process, core~=thread for GPU usage, otherwise OOM

Jupyter: jupyter server   --no-browser   --ip=0.0.0.0   --ServerApp.runtime_dir=/tmp/jrt    --ServerApp.allow_remote_access=True --ServerApp.allow_origin="*" --port=11833
"""

import glob
import json
import logging
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import jax, jax.numpy as jnp
from dask import delayed
from dask.distributed import Client, LocalCluster, progress
from dask_jobqueue import SLURMCluster
from functools import partial
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


# --- Configuration management ---

CONFIG_FILE = Path.home() / ".slurmcli"

logger = logging.getLogger("slurmcli")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

CONFIG_FIELDS: List[Dict[str, Any]] = [
    {
        "key": "mail",
        "env": "SLURMCLI_MAIL",
        "prompt": "Notification email address",
        "type": str,
        "default": "",
    },
    {
        "key": "scheduler_host",
        "env": "SLURMCLI_SCHEDULER_HOST",
        "prompt": "Scheduler hostname/IP (for reconnecting)",
        "type": str,
        "default": "192.168.240.53",
    },
    {
        "key": "scheduler_port",
        "env": "SLURMCLI_SCHEDULER_PORT",
        "prompt": "Dask scheduler port",
        "type": int,
        "default": 8786,
    },
    {
        "key": "dashboard_port",
        "env": "SLURMCLI_DASHBOARD_PORT",
        "prompt": "Dask dashboard port",
        "type": int,
        "default": 8787,
    },
    {
        "key": "jupyter_port",
        "env": "SLURMCLI_JUPYTER_PORT",
        "prompt": "Base Jupyter port",
        "type": int,
        "default": 11833,
    },
    {
        "key": "venv_activate",
        "env": "SLURMCLI_VENV_ACTIVATE",
        "prompt": "Path to Python virtualenv to activate in workers",
        "type": str,
        "default": "/nfs/nhome/live/jbauer/recurrent_feature/.venv",
    },
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
    """Load required credentials via env → file → interactive prompt."""

    config: Dict[str, Any] = {}
    file_cache = _load_config_file(CONFIG_FILE)
    missing: List[Dict[str, Any]] = []

    for field in CONFIG_FIELDS:
        key = field["key"]
        env_name = field["env"]
        typ = field["type"]
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
            key = field["key"]
            typ = field["type"]
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

    config_snapshot = {k: config.get(k) for k in sorted(config)}
    logger.info(
        "Loaded slurmcli configuration (env/file precedence applied): %s",
        config_snapshot,
    )

    return config


CONFIG = load_credentials()

MAIL = CONFIG.get("mail")
SCHEDULER_HOST = CONFIG.get("scheduler_host")
IP_OLDSPORT = SCHEDULER_HOST  # Backwards compatibility alias
PORT_SLURM_SCHEDULER = int(CONFIG.get("scheduler_port"))
PORT_SLURM_DASHBOARD = int(CONFIG.get("dashboard_port"))
JUPYTER_PORT = int(CONFIG.get("jupyter_port"))
VENV_ACTIVATE = CONFIG.get("venv_activate")

DEFAULT_JUPYTER_RUNTIME_BASE = "/tmp/jrt"


def start_jupyter_server_on_worker(
    port=JUPYTER_PORT,
    runtime_dir_base=DEFAULT_JUPYTER_RUNTIME_BASE,
    timeout=45.0,
    jupyter_exe=None,
    token=None,
    allow_origin="*",
):
    """Launch a Jupyter server from within a Dask worker process.

    Returns a serialisable dict containing connection details or error info.
    """

    # Imports duplicated locally so the function works when executed on workers.
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

    # Build a per-worker runtime directory to avoid collisions across workers.
    runtime_dir = os.path.join(runtime_dir_base, f"worker-{os.getpid()}")
    os.makedirs(runtime_dir, exist_ok=True)

    if jupyter_exe:
        cmd = [jupyter_exe, "server"]
    else:
        cmd = [sys.executable, "-m", "jupyter", "server"]

    env = os.environ.copy()
    env.setdefault("JUPYTER_RUNTIME_DIR", runtime_dir)

    # Respect explicit port if provided; otherwise allow Jupyter to choose.
    if port is None:
        port_arg = "--port=0"
    else:
        port_arg = f"--port={int(port)}"

    cmd += [
        "--no-browser",
        "--ip=0.0.0.0",
        port_arg,
        "--ServerApp.port_retries=0",
        f"--ServerApp.token={token}",
        "--ServerApp.allow_remote_access=True",
        f"--ServerApp.allow_origin={allow_origin}",
        f"--ServerApp.runtime_dir={runtime_dir}",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        env=env,
    )

    url = None
    parsed_url = None
    captured = []
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
            line = line.rstrip("\n")
            captured.append(line)
            match = url_re.search(line)
            if match:
                candidate = match.group(1)
                parsed = urlparse(candidate)
                qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
                qs["token"] = token  # enforce known token

                host_ip = _worker_host_ip()
                port_part = f":{parsed.port}" if parsed.port else ""

                # Replace loopback hosts with worker IP for remote access
                if parsed.hostname in {"127.0.0.1", "localhost"}:
                    netloc = f"{host_ip}{port_part}"
                else:
                    netloc = parsed.netloc

                new_query = urlencode(qs)
                parsed = parsed._replace(netloc=netloc, query=new_query)
                candidate = urlunparse(parsed)

                url = candidate
                parsed_url = parsed
                break

        if proc.poll() is not None:
            if proc.stdout:
                remainder = proc.stdout.read()
                if remainder:
                    captured.append(remainder)
            break

        if (time.time() - t0) > timeout:
            break

    if not url:
        files = glob.glob(os.path.join(runtime_dir, "nbserver-*.json"))
        files.sort(key=os.path.getmtime, reverse=True)
        if files:
            try:
                with open(files[0], "r") as fh:
                    data = json.load(fh)
                candidate = data.get("url") or data.get("base_url")
                if candidate:
                    parsed = urlparse(candidate)
                    host_ip = _worker_host_ip()
                    if parsed.hostname in {"127.0.0.1", "localhost"}:
                        port_part = f":{parsed.port}" if parsed.port else ""
                        netloc = f"{host_ip}{port_part}"
                    else:
                        netloc = parsed.netloc
                    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
                    qs.setdefault("token", data.get("token", token))
                    new_query = urlencode(qs)
                    parsed = parsed._replace(netloc=netloc, query=new_query)
                    url = urlunparse(parsed)
                    parsed_url = parsed
            except Exception as exc:
                captured.append(f"[runtime-json-error] {exc}")

    if url:
        host_for_log = parsed_url.hostname if parsed_url else None
        result = {
            "status": "ok",
            "pid": proc.pid,
            "url": url,
            "token": token,
            "host": host_for_log,
            "port": (parsed_url.port if parsed_url else None),
            "worker": worker_meta,
            "runtime_dir": runtime_dir,
            "cmd": cmd,
            "log_tail": captured[-20:],
        }
        return result

    # If we reached here, shut down the process and report error
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass

    return {
        "status": "error",
        "error": "no-url",
        "pid": proc.pid,
        "worker": worker_meta,
        "log_tail": captured[-40:],
    }


def stop_jupyter_server_on_worker(pid: int, sig: int = signal.SIGTERM, wait: float = 5.0):
    """Attempt to terminate a previously launched Jupyter server on a worker."""

    import os
    import time

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

# --- Core Cluster Logic (from launchscript.py) ---

WALLTIME = "01:00:00"
MEMORY_GB_DEFAULT = 32  # Default memory in GB
MEMORY = f"{MEMORY_GB_DEFAULT}GB"
PROCESSES_PER_WORKER = 1  # NEW: Default number of Dask processes per SLURM job
CORES = 1
NUM_JOBS = 4

def get_cluster(
    local=True,                  # if True, use local cluster instead of SLURM
    num_jobs=NUM_JOBS,                  # number of SLURM jobs / Dask workers (logical)
    processes=PROCESSES_PER_WORKER,     # Number of Dask worker processes per SLURM job
    threads_per_worker=None,             # NEW: explicit threads per Dask worker process (int)
    cores=CORES,                        # NOTE: legacy param; used when threads_per_worker not given
    memory=MEMORY,
    queue="cpu",
    account="gcnu-ac",           # set to "" if not used
    walltime=WALLTIME,
    log_dir="dask_logs",
    job_name="tfl",              # project name
    venv_activate=VENV_ACTIVATE,  # leave empty if not needed
    verbose=0,
    nodelist=None,               # specific node to use
    gres=None,                   # GPU resources (e.g., "gpu:2")
):
    """Initializes and returns a Dask cluster and client, either local or on SLURM.

    Behavior notes regarding threads:
      - threads_per_worker controls the number of threads each Dask worker process will use.
      - For local clusters we set n_workers = num_jobs * processes and threads_per_worker as provided.
      - For SLURMCluster we request cores-per-job = threads_per_worker * processes (so each process
        gets threads_per_worker threads). If threads_per_worker is None, we fall back to `cores`.
    """
    # Determine threads_per_worker fallback logic
    if threads_per_worker is None:
        # fall back to using cores as the threads per worker (legacy behavior)
        threads_per_worker = cores

    # Ensure sensible integer
    threads_per_worker = int(max(1, threads_per_worker))
    processes = int(max(1, processes))
    num_jobs = int(max(1, num_jobs))

    # For SLURMCluster we need to supply `cores` (total cores per job). Set it to threads_per_worker * processes.
    cores_for_job = threads_per_worker * processes

    if local:
        # Use local cluster for parallelization on the current machine
        print("🚀 Starting a local Dask cluster...")
        # Create one local Dask worker process per (job * processes) for parity with SLURM layout,
        # each with `threads_per_worker` threads.
        n_workers_local = num_jobs * processes
        cluster = LocalCluster(
            n_workers=n_workers_local,
            threads_per_worker=threads_per_worker,
            memory_limit=memory,
            processes=True,  # use separate processes (matches processes-per-job model)
        )
        client = Client(cluster)
    else:
        # Use SLURM cluster
        print(f"🚀 Submitting {num_jobs} jobs to the '{queue}' SLURM partition...")
        os.makedirs(log_dir, exist_ok=True)
        prologue = []
        if venv_activate:
            prologue.append(f"source {venv_activate}/bin/activate")

        # Build job_extra for additional SLURM options
        job_extra_directives = []
        if nodelist:
            job_extra_directives.append(f"--nodelist={nodelist}")
        
        if gres and 'gpu' in (gres or ""):
            try:
                num_gpus = int(gres.split(':')[-1])
                if num_gpus > 0:
                    job_extra_directives.append(f"--gpus-per-task={num_gpus}")
                    job_extra_directives.append(f"--gpu-bind=single:{num_gpus}")
                    print(f"✅ Configuring workers with --gpus-per-task={num_gpus}")
            except (ValueError, IndexError):
                print(f"⚠️  Could not parse GPU count from '{gres}'. Passing it directly.")
                if gres:
                    job_extra_directives.append(f"--gres={gres}")
        
        cluster = SLURMCluster(
            queue=queue,
            account=account or "",
            processes=processes, # number of worker processes per job
            cores=cores_for_job, # total cores per job (threads_per_worker * processes)
            memory=memory,
            walltime=walltime,
            job_name=job_name,
            log_directory=log_dir,
            job_script_prologue=prologue,
            job_extra_directives=job_extra_directives,
            scheduler_options={
                "port": PORT_SLURM_SCHEDULER,
                "dashboard_address": f":{PORT_SLURM_DASHBOARD}"
            },
        )

        # Scale to the number of SLURM jobs. Total workers (processes) will be num_jobs * processes
        cluster.scale(n=num_jobs)
        client = Client(cluster)
        
    total_workers_expected = num_jobs * processes
    print("Cluster dashboard:", getattr(cluster, "dashboard_link", "n/a"))
    print("Scheduler address:", client.scheduler.address)

    print(f"\n⏳ Waiting for {total_workers_expected} workers to connect...")
    client.wait_for_workers(n_workers=total_workers_expected, timeout=300)

    # --- Scheduler-provided worker info ---
    sched_info = client.scheduler_info()
    workers = sched_info.get("workers", {})
    n_workers = len(workers)
    print(f"\n✅ Workers connected: {n_workers}")

    # Compute totals and a small summary from scheduler metadata
    total_cores = sum(w.get("nthreads", 0) for w in workers.values())
    total_mem_bytes = sum(w.get("memory_limit", 0) for w in workers.values())
    total_mem_gb = total_mem_bytes / (1024 ** 3) if total_mem_bytes else 0.0
    print(f"   Summary: {n_workers} worker(s) — total cores={total_cores}, total mem≈{total_mem_gb:.2f}GB")
    print()
    
    if verbose > 0:
        print_worker_details(client, workers)
    
    return cluster, client

def print_worker_details(client, workers):
    """Probes workers for runtime info and prints a detailed table."""
    def _probe_worker():
        import os, platform
        info = {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
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


# --- Interactive CLI Logic (from interactive_launcher.py) ---

# All helper functions (parse_cpu_string, etc.) are unchanged...
def parse_cpu_string(cpu_str):
    """Parse SLURM CPU string format: ALLOCATED/IDLE/OTHER/TOTAL"""
    try:
        parts = cpu_str.split('/')
        if len(parts) == 4:
            allocated, idle, other, total = map(int, parts)
            return {
                'allocated': allocated,
                'idle': idle,
                'other': other,
                'total': total
            }
    except:
        pass
    return None

def parse_memory_string(mem_str):
    """Convert memory string to GB (handles MB, GB, etc.)"""
    mem_str = mem_str.strip().rstrip('+')
    try:
        if 'T' in mem_str.upper():
            return float(mem_str.upper().replace('T', '').replace('B', '')) * 1024
        elif 'G' in mem_str.upper():
            return float(mem_str.upper().replace('G', '').replace('B', ''))
        elif 'M' in mem_str.upper():
            return float(mem_str.upper().replace('M', '').replace('B', '')) / 1024
        else:
            # Assume MB if no unit
            return float(mem_str) / 1024
    except:
        return None

def get_detailed_node_info(nodes):
    """Get detailed information for a list of nodes."""
    node_info = {}
    
    for node in nodes:
        try:
            cmd = f"scontrol show node {node}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=5)
            output = result.stdout
            
            info = {
                'memory_gb': None,
                'cpus_total': None,
                'cpus_alloc': None,
                'gpus': '0',
                'state': 'unknown'
            }
            
            for line in output.split('\n'):
                # Memory
                if 'RealMemory=' in line:
                    for part in line.split():
                        if 'RealMemory=' in part:
                            try:
                                mem_mb = int(part.split('=')[1])
                                info['memory_gb'] = mem_mb / 1024
                            except:
                                pass
                
                # CPUs
                if 'CPUTot=' in line:
                    for part in line.split():
                        if 'CPUTot=' in part:
                            try:
                                info['cpus_total'] = int(part.split('=')[1])
                            except:
                                pass
                        if 'CPUAlloc=' in part:
                            try:
                                info['cpus_alloc'] = int(part.split('=')[1])
                            except:
                                pass
                
                # GPUs
                if 'Gres=' in line:
                    for part in line.split():
                        if 'Gres=' in part:
                            gres = part.split('=')[1]
                            if 'gpu:' in gres:
                                # Extract number of GPUs
                                gpu_part = gres.split('gpu:')[1]
                                # Handle formats like "gpu:4(S:0-1)" or "gpu:4"
                                info['gpus'] = gpu_part.split('(')[0] if '(' in gpu_part else gpu_part
                
                # State
                if 'State=' in line:
                    for part in line.split():
                        if 'State=' in part:
                            info['state'] = part.split('=')[1].split('+')[0].lower()
            
            node_info[node] = info
            
        except Exception as e:
            # If we can't get info for a node, skip it
            continue
    
    return node_info

def get_idle_partitions():
    """Parses `sinfo` to find partitions with idle or mixed nodes."""
    print("🔎 Checking for available SLURM resources...")
    try:
        # Get basic partition info
        cmd = "sinfo -h -o '%P|%t|%C|%G|%N'"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        partition_data = defaultdict(lambda: {
            'states': set(),
            'cpu_stats': [],
            'gres': set(),
            'nodes': set()
        })
        
        for line in lines:
            try:
                parts = line.split('|')
                if len(parts) < 5:
                    continue
                    
                partition, state, cpus, gres, nodes = parts[:5]
                partition = partition.strip('*')
                
                # Only include partitions with available resources
                if 'idle' in state or 'mixed' in state or 'mix' in state:
                    partition_data[partition]['states'].add(state)
                    partition_data[partition]['cpu_stats'].append(cpus)
                    partition_data[partition]['gres'].add(gres)
                    # Parse node list
                    for node in nodes.split(','):
                        partition_data[partition]['nodes'].add(node.strip())
                        
            except (ValueError, IndexError):
                continue

        if not partition_data:
            return None
        
        # Now get detailed info for each partition's nodes
        choices = {}
        for i, (partition, data) in enumerate(partition_data.items()):
            nodes_list = sorted(data['nodes'])
            
            # Get detailed node information
            node_details = get_detailed_node_info(nodes_list)
            
            # Aggregate CPU stats
            cpu_info = None
            if data['cpu_stats']:
                cpu_info = parse_cpu_string(data['cpu_stats'][0])
            
            # Group nodes by configuration
            node_configs = defaultdict(list)
            for node, info in node_details.items():
                config_key = (info['memory_gb'], info['cpus_total'], info['gpus'])
                node_configs[config_key].append((node, info))
            
            choices[i + 1] = {
                "partition": partition,
                "states": sorted(data['states']),
                "cpu_info": cpu_info,
                "node_configs": dict(node_configs),
                "all_nodes": node_details
            }
        
        return choices

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Error: `sinfo` command failed. Are you on a SLURM login node? ({e})")
        return None

def format_gpu_count(gpu_str):
    """Format GPU string for display."""
    if not gpu_str or gpu_str == '0' or gpu_str == '(null)' or 'null' in gpu_str.lower():
        return "0"
    return gpu_str

def prompt_for_value(prompt_text, default, type_converter=str):
    """Generic function to prompt user for a value with a default."""
    while True:
        value = input(f"➡️  {prompt_text} (default: {default}): ") or str(default)
        try:
            return type_converter(value)
        except ValueError:
            print(f"❌ Invalid input. Please enter a value of type {type_converter.__name__}.")


def main_interactive():
    """Main interactive loop to configure and launch the SLURM cluster."""
    idle_partitions = get_idle_partitions()

    if not idle_partitions:
        print("\n😔 No idle or mixed nodes found. Please try again later.")
        return

    print("\n✅ Found available partitions:\n")
    print("="*90)
    
    for key, val in idle_partitions.items():
        partition = val['partition']
        cpu_info = val['cpu_info']
        node_configs = val['node_configs']
        
        print(f"[{key}] Partition: {partition}")
        
        if cpu_info:
            avail = cpu_info['idle']
            total = cpu_info['total']
            alloc = cpu_info['allocated']
            print(f"    CPUs: {avail}/{total} available ({alloc} in use)")
        
        print(f"    Nodes by configuration:")
        for config, node_list in sorted(node_configs.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0), reverse=True):
            mem_gb, cpus, gpus = config
            count = len(node_list)
            mem_str = f"{mem_gb:.0f}GB" if mem_gb else "N/A"
            cpu_str = f"{cpus}" if cpus else "N/A"
            gpu_str = format_gpu_count(gpus)
            
            available_count = sum(1 for _, info in node_list if info['state'] in ['idle', 'mixed', 'mix'])
            
            print(f"      • {count} nodes: {mem_str} RAM, {cpu_str} CPUs, {gpu_str} GPUs ({available_count} available)")
            
            if count <= 10:
                for node, info in sorted(node_list):
                    state_icon = "✓" if info['state'] in ['idle', 'mixed', 'mix'] else "✗"
                    cpus_used = f"{info['cpus_alloc']}/{info['cpus_total']}" if info['cpus_alloc'] is not None else "N/A"
                    print(f"        {state_icon} {node:<20} ({info['state']}, CPUs: {cpus_used})")
        
        print("-"*90)

    # 1. Select Partition
    while True:
        try:
            choice = int(input("\n➡️  Enter the number of your choice: "))
            if choice in idle_partitions:
                selected_partition = idle_partitions[choice]['partition']
                available_nodes = idle_partitions[choice]['all_nodes']
                break
            else:
                print(f"❌ Invalid choice. Please select a number from 1 to {len(idle_partitions)}.")
        except ValueError:
            print("❌ Please enter a valid number.")

    print(f"\n✅ You selected partition: '{selected_partition}'")

    launch_kernels_choice = input(
        "\n🧪 Would you like to start a Jupyter server on each worker after launch? (y/n): "
    ).strip().lower()
    launch_kernels = launch_kernels_choice == 'y'

    if launch_kernels:
        print(
            "\nℹ️  Defaults updated for interactive Jupyter use:"
            "\n   • SLURM jobs: 1"
            "\n   • CPU cores per process: 16"
            "\n   • No additional limits on threads or processes\n"
        )

    print("Now, let's configure the cluster.\n")

    # 2. Gather Parameters --- MODIFIED SECTION

    walltime = prompt_for_value("Enter walltime [HH:MM:SS]", WALLTIME)
    default_num_jobs = 1 if launch_kernels else NUM_JOBS
    default_processes = PROCESSES_PER_WORKER
    default_cores_per_process = 16 if launch_kernels else CORES

    num_jobs = prompt_for_value("Enter number of SLURM jobs", default_num_jobs, int)
    processes = prompt_for_value("Enter processes per job", default_processes, int)
    cores_per_process = prompt_for_value("Enter CPU cores per process", default_cores_per_process, int)
    
    # NEW: interactive limit for threads per worker
    print(
        "\n🧵 Threads limit: You can optionally limit the number of threads each Dask worker process uses.\n"
        "If you enter 0 (or leave empty), no additional limit is applied and the value of 'cores per process' is used.\n"
        "Otherwise the threads-per-worker will be set to the smaller of the value you enter and 'cores per process'."
    )
    default_threads_limit = 0
    threads_limit = prompt_for_value("Limit threads per worker (0 for no limit)", default_threads_limit, int)
    
    memory_gb = prompt_for_value("Enter memory per job in GB (e.g., 32)", MEMORY_GB_DEFAULT, int)
    memory = f"{memory_gb}GB"

    # Compute threads_per_worker that will actually be used
    if threads_limit and threads_limit > 0:
        threads_per_worker = min(cores_per_process, threads_limit)
    else:
        threads_per_worker = cores_per_process

    # Calculate total cores for the job, as required by dask-jobqueue
    total_cores_for_job = threads_per_worker * processes

    # 3. Node and GPU specification
    print(f"\n📍 Detailed node list for partition '{selected_partition}':")
    print("="*90)
    print(f"{'Idx':<5} {'Node':<22} {'State':<10} {'Memory':<12} {'CPUs (used/total)':<18} {'GPUs':<6}")
    print("-"*90)
    
    available_for_selection = []
    sorted_nodes = sorted(available_nodes.items())
    for idx, (node, info) in enumerate(sorted_nodes, start=1):
        state = info['state']
        mem_str = f"{info['memory_gb']:.0f}GB" if info['memory_gb'] else "N/A"
        cpu_str = f"{info['cpus_alloc'] or 0}/{info['cpus_total'] or 'N/A'}"
        gpu_str = format_gpu_count(info['gpus'])
        is_available = state in ['idle', 'mixed', 'mix']
        marker = "✅" if is_available else " "
        print(f"{idx:<5} {marker} {node:<20} {state:<10} {mem_str:<12} {cpu_str:<18} {gpu_str:<6}")
        if is_available:
            available_for_selection.append((idx, node))
    
    print("="*90)
    print(f"✓ = Available for job submission ({len(available_for_selection)} nodes)")
    print()
    
    # NEW: choose node by enumerated index (or leave empty for any)
    while True:
        nodelist_choice = input("Enter index of specific node to use (leave empty for any): ").strip()
        if nodelist_choice == "":
            nodelist = None
            break
        try:
            nidx = int(nodelist_choice)
            # Validate index exists in sorted_nodes
            if 1 <= nidx <= len(sorted_nodes):
                nodelist = sorted_nodes[nidx - 1][0]
                break
            else:
                print(f"❌ Invalid index. Choose between 1 and {len(sorted_nodes)}.")
        except ValueError:
            print("❌ Please enter a valid integer index or leave empty.")

    num_gpus = prompt_for_value("Enter number of GPUs per process (0 for CPU-only)", 0, int)
    gres = f"gpu:{num_gpus}" if num_gpus > 0 else None

    # 4. Confirmation --- MODIFIED SECTION
    print("\n" + "="*50)
    print("📋 LAUNCH SUMMARY")
    print("="*50)
    print(f"  Partition / Queue: {selected_partition}")
    print(f"  Walltime:          {walltime}")
    print(f"  SLURM Jobs:        {num_jobs}")
    print(f"  Processes per job: {processes}")
    print(f"  Cores per process: {cores_per_process}")
    print(f"  Threads per process (actual): {threads_per_worker}")
    print(f"  Total cores per job: {total_cores_for_job}  (threads_per_worker * processes)")
    print(f"  Memory per job:    {memory}")
    print(f"  Total workers:     {num_jobs * processes}")
    print(f"  Specific node:     {nodelist or 'Any available'}")
    print(f"  GPUs per process:  {num_gpus}")
    print(f"  Launch kernels:    {'Yes' if launch_kernels else 'No'}")
    print("="*50)

    if input("Proceed with launch? (y/n): ").lower() != 'y':
        print("🚫 Launch aborted.")
        return

    # 5. Launch Cluster --- MODIFIED SECTION
    cluster, client = None, None
    started_jupyter_servers = []
    try:
        cluster, client = get_cluster(
            local=False,
            queue=selected_partition,
            num_jobs=num_jobs,
            processes=processes,
            threads_per_worker=threads_per_worker,
            cores=cores_per_process,  # legacy param, but get_cluster will use threads_per_worker to decide effective cores
            memory=memory,
            walltime=walltime,
            nodelist=nodelist,
            gres=gres,
            verbose=1
        )

        print("\n✅ Dask cluster is running!")
        print("   You can connect to this session from another terminal or notebook.")
        print(f"   Reconnect with: Client('{client.scheduler.address}')")
        print("   Press Ctrl+C to shut down the cluster.")
        
        try:
            with open('.dask_scheduler_address', 'w') as f:
                f.write(client.scheduler.address)
            print(f"   Scheduler address saved to .dask_scheduler_address")
        except:
            pass

        # If requested, attempt to launch kernels on each worker via Dask
        if launch_kernels:
            print("\n🔌 Attempting to start an IPython/Jupyter kernel on each worker via Dask...")
            worker_infos = client.scheduler_info().get("workers", {})
            worker_addresses = list(worker_infos.keys())
            if not worker_addresses:
                print("⚠️  No workers available to launch Jupyter kernels.")
            else:
                session_tag = uuid.uuid4().hex[:8]
                for idx, worker_addr in enumerate(worker_addresses):
                    worker_port = JUPYTER_PORT
                    if worker_port is not None:
                        worker_port = JUPYTER_PORT + idx
                        if worker_port > 65535:
                            worker_port = None
                    runtime_base = os.path.join(
                        DEFAULT_JUPYTER_RUNTIME_BASE,
                        f"session-{session_tag}",
                        f"worker-{idx}"
                    )
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
                        print(f"❌ Worker {worker_addr}: exception while starting Jupyter server -> {exc}")
                        continue

                    if info.get("status") == "ok" and info.get("url"):
                        host_display = info.get("host") or worker_addr
                        print(f"✅ Worker {worker_addr} ({host_display})")
                        print(f"   PID: {info.get('pid')} | URL: {info.get('url')}")
                        started_jupyter_servers.append({
                            "worker": worker_addr,
                            "pid": info.get("pid"),
                        })
                    else:
                        print(f"❌ Worker {worker_addr}: {info.get('error', 'failed to start Jupyter server')}")
                        log_tail = info.get("log_tail") or []
                        if log_tail:
                            print("   Log tail:")
                            for line in log_tail[-10:]:
                                print(f"   {line}")

        # Keep running until interrupted: maintain previous behavior
        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupt received. Shutting down...")
    except Exception as e:
        print(f"\n💥 An error occurred during cluster setup: {e}")
    finally:
        print("\n🛑 Shutting down cluster and client...")
        if client and started_jupyter_servers:
            print("🧹 Stopping remote Jupyter servers...")
            for entry in started_jupyter_servers:
                worker = entry.get("worker")
                pid = entry.get("pid")
                try:
                    submit_kwargs = {"pure": False}
                    if worker:
                        submit_kwargs.update({"workers": [worker], "allow_other_workers": False})
                    fut = client.submit(
                        stop_jupyter_server_on_worker,
                        pid,
                        **submit_kwargs,
                    )
                    res = fut.result(timeout=15)
                    status = res.get("status")
                except Exception as exc:
                    status = f"error: {exc}"
                print(f"   Worker {worker or 'unknown'} pid {pid}: {status}")
        if client: 
            try:
                client.close()
            except:
                pass
        if cluster: 
            try:
                cluster.close()
            except:
                pass
        print("✅ Cleanup complete.")

# --- Main Entry Point ---


def main_cli() -> None:
    """Console script entry point for the interactive launcher."""
    main_interactive()


if __name__ == "__main__":
    main_cli()
