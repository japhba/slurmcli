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
import random
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
    {"key": "jupyter_require_token", "env": "SLURMCLI_JUPYTER_REQUIRE_TOKEN", "prompt": "Require token auth for Jupyter (set False for VSCode)", "type": bool, "default": False},
    {"key": "venv_activate", "env": "SLURMCLI_VENV_ACTIVATE", "prompt": "Path to Python virtualenv to activate in workers", "type": str, "default": "/nfs/nhome/live/jbauer/recurrent_feature/.venv"},
]


def _cast_value(value: Any, typ: type) -> Any:
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    if typ is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
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


import readline

def input_with_prefill(prompt: str, text: str) -> str:
    def hook():
        readline.insert_text(text)
        readline.redisplay()
    readline.set_startup_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def _detect_venv_suggestions(cwd: Path) -> List[Dict[str, str]]:
    """Scan cwd for pyproject.toml and venv directories."""
    suggestions = []
    pyproject = cwd / "pyproject.toml"
    if pyproject.exists():
        suggestions.append({"label": "pyproject.toml", "type": "pyproject", "path": str(cwd.resolve())})
    for venv_name in (".venv", "venv"):
        venv_dir = cwd / venv_name
        if (venv_dir / "bin" / "activate").exists():
            suggestions.append({"label": f"{venv_name}/", "type": "venv", "path": str(venv_dir.resolve())})
    return suggestions


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
        
        # If it has a default, use it but don't consider it "missing" for the mandatory loop 
        # UNLESS we want to force prompting. 
        # For venv_activate, we will handle it separately below.
        if field.get("default") is not None:
             config[key] = field["default"]
        else:
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

    # Load venv_mode from file (not in CONFIG_FIELDS, derived from user selection)
    config.setdefault("venv_mode", file_cache.get("venv_mode", "activate"))

    # Interactive override for venv
    if sys.stdin.isatty():
        current_venv = str(config.get("venv_activate", ""))
        current_mode = str(config.get("venv_mode", "activate"))

        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")

        suggestions = _detect_venv_suggestions(Path.cwd())

        print(f"\n📦 Virtual environment:")
        if suggestions:
            print(f"  Detected in {Path.cwd()}:")
            for i, s in enumerate(suggestions, 1):
                desc = "build fresh venv with uv" if s["type"] == "pyproject" else "existing virtualenv"
                print(f"    [{i}] {s['label']} → {desc}")
        if current_venv:
            mode_label = " (pyproject)" if current_mode == "pyproject" else ""
            print(f"  Current: {current_venv}{mode_label}")

        hint_parts = []
        if suggestions:
            hint_parts.append(f"[1-{len(suggestions)}]")
        hint_parts.append("path")
        hint_parts.append("'none'")
        if current_venv:
            hint_parts.append("ENTER to keep")
        hint = ", ".join(hint_parts)

        while True:
            choice = input(f"  ({hint})> ").strip()

            if not choice:
                break

            if choice.lower() == "none":
                config["venv_activate"] = ""
                config["venv_mode"] = "activate"
                break

            # Numbered suggestion
            if choice.isdigit() and suggestions and 1 <= int(choice) <= len(suggestions):
                s = suggestions[int(choice) - 1]
                config["venv_activate"] = s["path"]
                config["venv_mode"] = s["type"]
                break

            # Path to pyproject.toml directly
            p = Path(choice).expanduser().resolve()
            if p.name == "pyproject.toml" and p.exists():
                config["venv_activate"] = str(p.parent)
                config["venv_mode"] = "pyproject"
                break

            # Directory containing pyproject.toml (but no venv)
            if p.is_dir() and (p / "pyproject.toml").exists() and not (p / "bin" / "activate").exists():
                config["venv_activate"] = str(p)
                config["venv_mode"] = "pyproject"
                break

            # Standard venv path validation
            if p.name == "activate" and p.parent.name == "bin":
                check_path = p
            elif p.name.startswith("python") and p.parent.name == "bin":
                check_path = p.parent / "activate"
            else:
                check_path = p / "bin" / "activate"

            if check_path.exists():
                config["venv_activate"] = choice
                config["venv_mode"] = "activate"
                break

            print(f"  ⚠️ Could not validate '{choice}'.")
            if input("  Keep anyway? (y/N): ").strip().lower() == 'y':
                config["venv_activate"] = choice
                config["venv_mode"] = "activate"
                break

        logger.info("venv_activate=%s venv_mode=%s", config.get("venv_activate"), config.get("venv_mode"))

        # Persist changes
        changed = False
        for key in ("venv_activate", "venv_mode"):
            if str(config.get(key, "")) != str(file_cache.get(key, "")):
                file_cache[key] = config.get(key, "")
                changed = True
        if changed:
            _save_config_file(CONFIG_FILE, file_cache)

    return config


CONFIG = load_credentials()
MAIL = CONFIG.get("mail")
SCHEDULER_HOST = CONFIG.get("scheduler_host")
PORT_SLURM_SCHEDULER = int(CONFIG.get("scheduler_port"))
PORT_SLURM_DASHBOARD = int(CONFIG.get("dashboard_port"))
JUPYTER_PORT = int(CONFIG.get("jupyter_port"))
JUPYTER_REQUIRE_TOKEN = _cast_value(CONFIG.get("jupyter_require_token", False), bool)
VENV_ACTIVATE = CONFIG.get("venv_activate")
VENV_MODE = CONFIG.get("venv_mode", "activate")
if VENV_MODE == "pyproject":
    # Infer project dir from the active venv's editable installs.
    # Convention: UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" means the venv
    # basename matches the project directory basename. We look up the project path
    # from the editable install's direct_url.json metadata (written by uv/pip on
    # `uv sync` / `pip install -e`), which records the source directory as a file:// URL.
    # Falls back to CWD if detection fails.
    def _infer_project_dir_from_venv() -> Optional[str]:
        import sys, json
        site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        for p in site_packages.glob("*.dist-info/direct_url.json"):
            data = json.loads(p.read_text())
            if data.get("dir_info", {}).get("editable"):
                url = data.get("url", "")
                if url.startswith("file://"):
                    candidate = Path(url.removeprefix("file://"))
                    if (candidate / "pyproject.toml").exists():
                        return str(candidate)
        return None
    VENV_ACTIVATE = _infer_project_dir_from_venv() or str(Path.cwd())
DEFAULT_JUPYTER_RUNTIME_BASE = "/tmp/jrt"
SCHEDULER_ADDRESS_FILE = Path().home() / ".dask_scheduler_address"
SCHEDULER_ADDRESS_MAP_FILE = Path().home() / ".dask_scheduler_addresses.json"

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
    "gpu_names",
    "gres",
    "launch_kernels",
    "jupyter_only",
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
    gpu_names = normalized.get("gpu_names")
    if isinstance(gpu_names, str):
        names = [n.strip() for n in gpu_names.split(",") if n.strip()]
        normalized["gpu_names"] = names or None
    elif isinstance(gpu_names, (list, tuple)):
        normalized["gpu_names"] = [str(n).strip() for n in gpu_names if str(n).strip()] or None
    normalized["launch_kernels"] = bool(normalized.get("launch_kernels", False))
    normalized["jupyter_only"] = bool(normalized.get("jupyter_only", False))
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
    begin_time = config.get("begin_time")
    num_jobs = int(config.get("num_jobs") or 0)
    processes = int(config.get("processes") or 1)
    cores_per_process = int(config.get("cores_per_process") or 1)
    threads_per_worker = int(config.get("threads_per_worker") or cores_per_process or 1)
    total_cores_for_job = threads_per_worker * processes
    memory = config.get("memory", "n/a")
    total_workers = num_jobs * processes
    nodelist = config.get("nodelist") or "Any available"
    num_gpus = int(config.get("num_gpus") or 0)
    gpu_names = config.get("gpu_names") or []
    gpu_names = config.get("gpu_names") or []
    launch_kernels = bool(config.get("launch_kernels"))
    jupyter_only = bool(config.get("jupyter_only"))
    mode = "Jupyter only" if jupyter_only else ("Jupyter + Dask" if launch_kernels else "Dask only")

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"  Launch mode:       {mode}")
    print(f"  Partition / Queue: {partition_display}")
    print(f"  Walltime:          {walltime}")
    if begin_time:
        print(f"  Begin time:        {begin_time}")
    print(f"  SLURM Jobs:        {num_jobs}")
    print(f"  Processes per job: {processes}")
    print(f"  Cores per process: {cores_per_process}")
    print(f"  Threads per process (actual): {threads_per_worker}")
    print(f"  Total cores per job: {total_cores_for_job}")
    print(f"  Memory per job:    {memory}")
    print(f"  Total workers:     {total_workers}")
    print(f"  Specific node:     {nodelist}")
    print(f"  GPUs per process:  {num_gpus}")
    if gpu_names:
        print(f"  GPU model filter:  {', '.join(gpu_names)}")
    venv_display = VENV_ACTIVATE or "None"
    if VENV_MODE == "pyproject":
        venv_display += " (pyproject → uv)"
    print(f"  Venv:              {venv_display}")
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


_CLUSTER_ID_CHOICES = [
    "brisk-otter",
    "mellow-fox",
    "sunny-heron",
    "quiet-bison",
    "lively-lynx",
    "gentle-orca",
    "frosty-wren",
    "dusky-koala",
    "amber-stoat",
    "silver-hare",
    "rapid-sparrow",
    "jolly-ibis",
    "calm-mantis",
    "bright-pika",
    "dusky-puma",
    "steady-kestrel",
    "bold-newt",
    "glowing-ibex",
    "swift-sable",
    "serene-tern",
]


def _generate_cluster_id() -> str:
    return random.choice(_CLUSTER_ID_CHOICES)


def _normalize_scheduler_address(address: str) -> str:
    if not address:
        return address
    if address.startswith("inproc://"):
        return address
    return address if "://" in address else f"tcp://{address}"


def _record_scheduler_address(address: str, cluster_type: Optional[str], cluster_id: Optional[str]) -> None:
    addr = _normalize_scheduler_address(address)
    try:
        SCHEDULER_ADDRESS_FILE.write_text(addr)
    except Exception:
        pass

    data: Dict[str, Any] = {}
    try:
        raw = SCHEDULER_ADDRESS_MAP_FILE.read_text(encoding="utf-8").strip()
        if raw:
            data = json.loads(raw)
    except Exception:
        data = {}

    data["last"] = addr
    if cluster_type:
        data[cluster_type] = addr
    if cluster_id:
        by_id = data.get("by_id", {})
        if not isinstance(by_id, dict):
            by_id = {}
        by_id[str(cluster_id)] = addr
        data["by_id"] = by_id

    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "address": addr,
            "cluster_type": cluster_type or "unknown",
            "cluster_id": cluster_id or "unknown",
            "timestamp": time.time(),
        }
    )
    data["history"] = history

    try:
        SCHEDULER_ADDRESS_MAP_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _resolve_gpu_nodelist(gpu_types: List[str], *, partition: Optional[str] = None) -> List[str]:
    """Return node names whose GRES contains any of the requested GPU types."""
    from slurmcli.slurm_parser import extract_gpu_labels
    cmd = "sinfo -h -o '%N|%G'"
    if partition:
        cmd += f" -p {partition}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=8)
    nodes: List[str] = []
    seen: set[str] = set()
    for line in result.stdout.strip().splitlines():
        if '|' not in line:
            continue
        nodelist_str, gres = line.split('|', 1)
        labels = [l.lower() for l in extract_gpu_labels(gres)]
        if any(g in labels for g in gpu_types):
            from slurmcli.cluster_status import expand_nodelist
            for n in expand_nodelist(nodelist_str.strip()):
                if n not in seen:
                    seen.add(n)
                    nodes.append(n)
    return nodes


def _get_node_gpu_labels() -> Dict[str, str]:
    """Return {hostname: gpu_label} by parsing sinfo GRES, e.g. 'gpu-sr675-31': 'l40s'."""
    from slurmcli.slurm_parser import extract_gpu_labels
    from slurmcli.cluster_status import expand_nodelist
    try:
        result = subprocess.run(["sinfo", "-h", "-o", "%N|%G"], capture_output=True, text=True, timeout=8)
        if result.returncode != 0:
            return {}
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        if "|" not in line:
            continue
        node_str, gres = line.split("|", 1)
        labels = extract_gpu_labels(gres)
        label = ",".join(dict.fromkeys(l.lower() for l in labels)) if labels else ""
        if label:
            for node in expand_nodelist(node_str.strip()):
                mapping[node] = label
    return mapping


def _parse_gres_count(gres_str: str) -> int:
    """Extract total GPU count from a gres string like ``gpu:h100:8(S:0-1)``."""
    total = 0
    for part in gres_str.split(","):
        stripped = re.sub(r"\(.*?\)", "", part.strip())  # drop socket/IDX info
        if stripped.startswith("gpu"):
            segs = stripped.split(":")
            total += int(segs[-1]) if segs[-1].isdigit() else 0
    return total


def _available_capacity(
    partition: str,
    *,
    gres: Optional[str] = None,
    nodelist: Optional[str] = None,
    gpu_names: Optional[Iterable[str]] = None,
) -> Optional[int]:
    """Return the max number of jobs the partition can currently satisfy, or *None* if unknown.

    For GPU partitions this queries both total and *used* GRES per node so that
    only actually free GPUs are counted (not just idle/mixed state).
    """
    wants_gpu = gres and "gpu" in gres.lower()
    if wants_gpu:
        cmd = ["sinfo", "-h", "-p", partition, "-N",
               "-O", "NodeList:|,Gres:|,GresUsed:|,StateLong:|"]
    else:
        cmd = ["sinfo", "-h", "-p", partition, "-o", "%N|%t"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if result.returncode != 0:
            return None
    except Exception:
        return None

    idle_nodes: set[str] = set()

    if wants_gpu:
        from slurmcli.slurm_parser import extract_gpu_labels
        gpu_filter = [g.strip().lower() for g in gpu_names] if gpu_names else None
        free_gpus = 0
        for line in result.stdout.strip().splitlines():
            fields = [f.strip() for f in line.split("|")]
            if len(fields) < 4:
                continue
            node, gres_total, gres_used, state = fields[0], fields[1], fields[2], fields[3]
            if "idle" not in state.lower() and "mix" not in state.lower():
                continue
            if gpu_filter:
                labels = [l.lower() for l in extract_gpu_labels(gres_total)]
                if not any(g in labels for g in gpu_filter):
                    continue
            free_gpus += _parse_gres_count(gres_total) - _parse_gres_count(gres_used)
            idle_nodes.add(node)
        try:
            gpus_per_job = int(gres.split(":")[-1])
        except (ValueError, IndexError):
            gpus_per_job = 1
        return free_gpus // gpus_per_job if gpus_per_job > 0 else len(idle_nodes)
    else:
        from slurmcli.cluster_status import expand_nodelist
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            node_str, state = parts[0], parts[1]
            if "idle" not in state.strip().lower() and "mix" not in state.strip().lower():
                continue
            idle_nodes.update(expand_nodelist(node_str.strip()))
        return len(idle_nodes)


def _normalize_slurm_memory(memory: str) -> str:
    text = (memory or "").strip()
    if not text:
        return text
    upper = text.upper()
    if upper.endswith("GB"):
        return upper[:-2] + "G"
    if upper.endswith("MB"):
        return upper[:-2] + "M"
    return text


def _resolve_venv_paths(venv_activate: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not venv_activate:
        return None, None
    path = Path(venv_activate).expanduser()
    if path.name == "activate" and path.parent.name == "bin":
        activate_path = path
        venv_root = path.parent.parent
    elif path.name.startswith("python") and path.parent.name == "bin":
        # Handle case where user provides path to python executable
        venv_root = path.parent.parent
        activate_path = path.parent / "activate"
    else:
        # Assume it is the root directory
        venv_root = path
        activate_path = path / "bin" / "activate"
    return str(venv_root), str(activate_path)


def _build_venv_prologue(
    venv_activate: Optional[str],
    venv_mode: str = "activate",
    *,
    launch_kernels: bool = False,
    jupyter_port: int = 11833,
    jupyter_require_token: bool = False,
    jupyter_foreground: bool = False,
    recreate_venv: bool = False,
) -> Tuple[List[str], Optional[str]]:
    """Build prologue commands and python path for worker job scripts.

    Returns (prologue_lines, python_path).
    """
    if not venv_activate:
        return [], None

    if venv_mode == "pyproject":
        project_dir = str(Path(venv_activate).expanduser().resolve())
        venv_local = os.environ["VENV_LOCAL"]
        prologue = [
            f'cd "{project_dir}"',
            f'export UV_PROJECT_ENVIRONMENT="{venv_local}/${{PWD##*/}}"',
        ]
        if recreate_venv:
            prologue.append('echo "=== slurmcli: removing $UV_PROJECT_ENVIRONMENT ==="; rm -rf "$UV_PROJECT_ENVIRONMENT"')
        prologue += [
            'echo "=== slurmcli: uv sync → $UV_PROJECT_ENVIRONMENT ==="',
            'uv sync || { echo "FATAL: uv sync failed"; exit 1; }',
            'source "$UV_PROJECT_ENVIRONMENT/bin/activate"',
            'echo "=== slurmcli: venv ready, python=$(which python) ==="',
        ]
        if launch_kernels:
            prologue += _jupyter_prologue_lines(jupyter_port, jupyter_require_token, foreground=jupyter_foreground)
        return prologue, "python"

    # activate mode (existing behavior)
    venv_root, activate_path = _resolve_venv_paths(venv_activate)
    prologue = [f"source {activate_path}"] if activate_path else []
    if launch_kernels:
        prologue += _jupyter_prologue_lines(jupyter_port, jupyter_require_token, foreground=jupyter_foreground)
    python_path = str(Path(venv_root) / "bin" / "python") if venv_root else None
    return prologue, python_path


def _jupyter_prologue_lines(port: int, require_token: bool, *, foreground: bool = False) -> List[str]:
    """Prologue lines to start a jupyter server (background or foreground)."""
    token_args = f'--ServerApp.token=$_SLURMCLI_JTOKEN --IdentityProvider.token=$_SLURMCLI_JTOKEN' if require_token else \
        '--ServerApp.token= --ServerApp.password= --ServerApp.disable_check_xsrf=True --IdentityProvider.token='
    jupyter_cmd = (f'python -m jupyter server --no-browser --ip=0.0.0.0 --port={port}'
        f' --ServerApp.port_retries=50 --ServerApp.allow_remote_access=True --ServerApp.allow_origin=*'
        f' {token_args}')
    token_line = f'export _SLURMCLI_JTOKEN=$(python -c "import uuid; print(uuid.uuid4().hex)")' if require_token else ''
    ip_line = '_WORKER_IP=$(hostname -I | awk \'{print $1}\')'
    url_line = (f'echo "=== slurmcli-jupyter: http://${{_WORKER_IP}}:{port}/?token=${{_SLURMCLI_JTOKEN:-}}"' if require_token else
        f'echo "=== slurmcli-jupyter: http://${{_WORKER_IP}}:{port}/"')
    if foreground:
        return [token_line, ip_line, url_line, f'exec {jupyter_cmd}']
    log_file = '/tmp/${SLURM_JOB_ID}/jupyter.log'
    extract_url = (f'_JURL=$(grep -oP "https?://[^\\s,;]+" {log_file} | head -1)'
        f' && [ -n "$_JURL" ] && echo "=== slurmcli-jupyter: $_JURL" || {{ {ip_line}; {url_line}; }}')
    return [
        token_line,
        f'nohup {jupyter_cmd} > {log_file} 2>&1 &',
        'sleep 3',
        extract_url,
    ]


def _srun_env() -> Dict[str, str]:
    """Return a copy of os.environ without stale SLURM variables.

    If slurmcli is invoked from inside an existing SLURM job (e.g. an
    interactive srun session), inherited SLURM_JOB_ID etc. would cause a
    new ``srun`` to try to run inside that allocation instead of creating
    a fresh one.
    """
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SLURM_"):
            del env[key]
    return env


def _build_srun_cmd(launch_cfg: Dict[str, Any]) -> List[str]:
    """Build an srun command from launch config, mirroring sbatch directives."""
    memory = _normalize_slurm_memory(launch_cfg.get("memory", "32G"))
    cmd = ['srun', '-J', 'slurmcli-jupyter', '-n', '1',
           '--cpus-per-task', str(launch_cfg.get("cores_per_process", 16)),
           '--mem', memory, '-t', launch_cfg.get("walltime", "01:00:00")]
    if launch_cfg.get("partition"):
        cmd += ['-p', launch_cfg["partition"]]
    if launch_cfg.get("nodelist"):
        cmd += ['--nodelist', launch_cfg["nodelist"]]
    num_gpus = int(launch_cfg.get("num_gpus") or 0)
    if num_gpus > 0:
        cmd += ['--gpus-per-task', str(num_gpus)]
    return cmd


def _build_jupyter_srun_script(*, recreate_venv: bool = False) -> str:
    """Build a bash script string for running jupyter via srun.

    Lines are joined with '; ' so each runs independently.  Critical
    commands already have their own ``|| { exit 1; }`` guards.
    """
    prologue, _ = _build_venv_prologue(
        VENV_ACTIVATE, VENV_MODE,
        launch_kernels=True, jupyter_port=JUPYTER_PORT, jupyter_require_token=JUPYTER_REQUIRE_TOKEN,
        jupyter_foreground=True, recreate_venv=recreate_venv,
    )
    return '; '.join(line for line in prologue if line)


def _run_squeue(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=8)
        return result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _get_squeue_rows(partition: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
    """Returns list of rows on success (possibly empty), or None if squeue failed."""
    base = ["squeue", "--all", "-h", "-o", "%i|%P|%T|%Q|%R|%u|%j"]
    if partition:
        base.extend(["-p", partition])

    rows_text = _run_squeue(base + ["--sort=-P"])
    if rows_text is None:
        rows_text = _run_squeue(base)
    if rows_text is None:
        return None

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
    if rows is None:
        print("ℹ️ Unable to query squeue for queue status.")
        return job_ids
    if not rows:
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


def _tail_slurm_logs(log_dir: str, job_ids: List[str], offsets: Dict[str, int]) -> Dict[str, int]:
    """Print new content from SLURM log files. Returns updated offsets."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return offsets
    for job_id in job_ids:
        for suffix in ("out", "err"):
            for pattern in (f"*-{job_id}.{suffix}",):
                for fpath in log_path.glob(pattern):
                    key = str(fpath)
                    offset = offsets.get(key, 0)
                    try:
                        size = fpath.stat().st_size
                        if size > offset:
                            with open(fpath, "r") as f:
                                f.seek(offset)
                                new_content = f.read()
                            if new_content.strip():
                                label = "stderr" if suffix == "err" else "stdout"
                                print(f"📄 [{job_id} {label}]")
                                for line in new_content.rstrip().splitlines():
                                    print(f"   {line}")
                            offsets[key] = size
                    except OSError:
                        pass
    return offsets


def _wait_for_workers_with_queue(
    client: Client,
    *,
    cluster: Any,
    total_workers_expected: int,
    partition: Optional[str],
    job_name: Optional[str],
    log_dir: str = "dask_logs",
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

    log_offsets: Dict[str, int] = {}

    while True:
        log_offsets = _tail_slurm_logs(log_dir, job_ids, log_offsets)
        try:
            client.wait_for_workers(n_workers=total_workers_expected, timeout=poll_interval)
            log_offsets = _tail_slurm_logs(log_dir, job_ids, log_offsets)
            return
        except TimeoutError:
            log_offsets = _tail_slurm_logs(log_dir, job_ids, log_offsets)
            job_ids = _report_queue_ahead(
                job_ids,
                partition=partition,
                job_name=job_name,
                max_rows=max_rows,
            )
        except Exception as exc:
            log_offsets = _tail_slurm_logs(log_dir, job_ids, log_offsets)
            print(f"⚠️ Error while waiting for workers ({type(exc).__name__}: {exc})")
            return


# --- Dask cluster launch ------------------------------------------------------
WALLTIME = "01:00:00"
MEMORY_GB_DEFAULT = 32
PROCESSES_PER_WORKER = 1
CORES = 4
NUM_JOBS = 4


def get_cluster(
    *,
    local: bool = True,
    cluster_id: Optional[str] = None,
    n_workers: Optional[int] = None,
    num_jobs: int = NUM_JOBS,
    processes: int = PROCESSES_PER_WORKER,
    threads_per_worker: Optional[int] = None,
    cores: int = CORES,
    memory: str = f"{MEMORY_GB_DEFAULT}GB",
    queue: Optional[str] = None,
    account: str = "gcnu-ac",
    walltime: str = WALLTIME,
    begin_time: Optional[str] = None,
    log_dir: str = "dask_logs",
    job_name: str = "tfl",
    venv_activate: Optional[str] = VENV_ACTIVATE,
    venv_mode: str = VENV_MODE,
    launch_kernels: bool = False,
    nodelist: Optional[str] = None,
    gres: Optional[str] = None,
    gpu_names: Optional[Iterable[str]] = None,
    verbose: int = 0,
    print_job_script: bool = False,
    local_processes: Optional[bool] = None,
    queue_poll_interval: float = 30.0,
    queue_report_max: int = 20,
):
    if threads_per_worker is None:
        threads_per_worker = cores

    threads_per_worker = int(max(1, threads_per_worker))
    processes = int(max(1, processes))
    num_jobs = int(max(1, num_jobs))
    if not local and cores == CORES:
        gpu_hint = (("gpu" in (queue or "").lower()) or ("gpu" in (gres or "").lower()))
        if gpu_hint:
            cores = 4
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
        if verbose >= 2:
            print(f"⚠️ Dask scheduler port {scheduler_port} already in use; requesting an ephemeral port instead.")
        scheduler_port = 0

    scheduler_options: Dict[str, Any] = {"port": scheduler_port if scheduler_port is not None else 0}
    if dashboard_address:
        scheduler_options["dashboard_address"] = dashboard_address

    if local:
        if cluster_id is None:
            cluster_id = _generate_cluster_id()
        if verbose >= 1:
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
        if cluster_id is None:
            cluster_id = _generate_cluster_id()
        queue_label = f"the '{queue}' SLURM partition" if queue else "the default SLURM partition"
        memory = _normalize_slurm_memory(memory)
        os.makedirs(log_dir, exist_ok=True)
        prologue, venv_python = _build_venv_prologue(
            venv_activate, venv_mode,
            launch_kernels=launch_kernels, jupyter_port=JUPYTER_PORT, jupyter_require_token=JUPYTER_REQUIRE_TOKEN,
        )

        job_extra_directives: List[str] = []
        worker_extra_args: List[str] = []
        if begin_time:
            job_extra_directives.append(f"--begin={begin_time}")
        if nodelist:
            job_extra_directives.append(f"--nodelist={nodelist}")
        if gpu_names:
            gpu_list = [str(name).strip().lower() for name in gpu_names if str(name).strip()]
            if gpu_list:
                matching_nodes = _resolve_gpu_nodelist(gpu_list, partition=queue)
                if matching_nodes:
                    gpu_nodelist = ",".join(matching_nodes)
                    if nodelist:
                        job_extra_directives.append(f"--nodelist={nodelist},{gpu_nodelist}")
                    else:
                        job_extra_directives.append(f"--nodelist={gpu_nodelist}")
                    if verbose >= 2:
                        print(f"✅ GPU filter {gpu_list} → {len(matching_nodes)} node(s): {gpu_nodelist}")
                else:
                    if verbose >= 2:
                        print(f"⚠️ No nodes found with GPU type(s) {gpu_list} in partition '{queue}'; submitting without node filter.")
        if gres and 'gpu' in (gres or ""):
            try:
                num_gpus = int(gres.split(':')[-1])
                if num_gpus > 0:
                    job_extra_directives.append(f"--gpus-per-task={num_gpus}")
                    job_extra_directives.append(f"--gpu-bind=single:{num_gpus}")
                    worker_extra_args.extend(["--resources", f"GPU={num_gpus}"])
                    if verbose >= 2:
                        print(f"✅ Configuring workers with --gpus-per-task={num_gpus}")
            except Exception:
                if gres:
                    job_extra_directives.append(f"--gres={gres}")

        cluster_kwargs = dict(
            queue=queue,
            account=account or "",
            processes=processes,
            cores=cores_for_job,
            memory=memory,
            walltime=walltime,
            death_timeout="15",
            job_name=job_name,
            log_directory=log_dir,
            job_script_prologue=prologue,
            job_extra_directives=job_extra_directives,
            worker_extra_args=worker_extra_args,
            scheduler_options=scheduler_options,
        )
        if venv_python:
            cluster_kwargs["python"] = venv_python
        cluster = SLURMCluster(**cluster_kwargs)
        try:
            job_script = cluster.job_script()
            logger.debug("SLURM job script:\n%s", job_script)
            if print_job_script:
                print("\n--- SLURM job script (debug) ---")
                print(job_script)
                print("--- End SLURM job script ---\n")
        except Exception as exc:
            logger.debug("Unable to render SLURM job script: %s", exc, exc_info=False)
        # Pre-flight: warn if requesting more jobs than available capacity (but don't clip — let SLURM queue)
        if queue:
            capacity = _available_capacity(queue, gres=gres, nodelist=nodelist, gpu_names=gpu_names)
            if capacity is not None and capacity < num_jobs:
                logger.info(
                    "Partition '%s' has ~%d free slot(s) for %d requested jobs — SLURM will queue the rest.",
                    queue, capacity, num_jobs,
                )

        if verbose >= 1:
            print(f"🚀 Submitting {num_jobs} jobs to {queue_label}…")
        cluster.scale(n=num_jobs)
        client = Client(cluster)

    cluster_type = "local" if local else ("gpu" if ("gpu" in (queue or "").lower() or "gpu" in (gres or "").lower()) else "cpu")
    _record_scheduler_address(client.scheduler.address, cluster_type, cluster_id)

    if n_workers is not None:
        total_workers_expected = int(max(1, n_workers))
    else:
        total_workers_expected = num_jobs * processes
    if verbose >= 2:
        print(f"🪪 Cluster id: {cluster_id}")
        print("Cluster dashboard:", getattr(cluster, "dashboard_link", "n/a"))
    if verbose >= 1:
        print("Scheduler address:", client.scheduler.address)
        print(f"⏳ Waiting for {total_workers_expected} workers to connect…")
    try:
        if local or verbose < 2:
            client.wait_for_workers(n_workers=total_workers_expected, timeout=300)
        else:
            _wait_for_workers_with_queue(
                client,
                cluster=cluster,
                total_workers_expected=total_workers_expected,
                partition=queue,
                job_name=job_name,
                log_dir=log_dir,
                poll_interval=queue_poll_interval,
                max_rows=queue_report_max,
            )
    except Exception as exc:
        print(f"⚠️ Timed out waiting for {total_workers_expected} workers ({type(exc).__name__}: {exc})")
        
        # FAILURE HANDLING: Check for recent error logs
        if not local:
            try:
                log_path = Path(log_dir)
                if log_path.exists():
                    # Find identifying job ID if possible, otherwise just latest err file
                    # We might not know the exact job ID if submission returned successfully but we didn't parse output.
                    # But we can look for the most recently modified *.err file
                    err_files = sorted(log_path.glob("*.err"), key=os.path.getmtime, reverse=True)
                    if err_files:
                        latest_err = err_files[0]
                        # Check if it was modified recently (e.g. in the last 2 minutes)
                        if time.time() - os.path.getmtime(latest_err) < 120:
                            print(f"\n🔎 Inspecting latest error log: {latest_err}")
                            print("--- Last 10 lines of stderr ---")
                            try:
                                with open(latest_err, "r") as f:
                                    lines = f.readlines()
                                    for line in lines[-10:]:
                                        print("   " + line.rstrip())
                            except Exception:
                                print("   (Unable to read log file)")
                            print("--------------------------------")
            except Exception:
                pass

    sched_info = client.scheduler_info()
    workers = sched_info.get("workers", {})
    n_workers = len(workers)
    total_cores = sum(w.get("nthreads", 0) for w in workers.values())
    total_mem_bytes = sum(w.get("memory_limit", 0) for w in workers.values())
    total_mem_gb = total_mem_bytes / (1024 ** 3) if total_mem_bytes else 0.0
    if verbose >= 1:
        import socket
        from collections import Counter
        def _resolve(ip):
            try:
                return socket.gethostbyaddr(ip)[0].split(".")[0]
            except Exception:
                return ip
        node_counts = Counter(_resolve(addr.split("://")[-1].split(":")[0]) for addr in workers)
        # Look up GPU types per node from sinfo
        node_gpu = _get_node_gpu_labels()
        parts = []
        for node, count in sorted(node_counts.items()):
            gpu_label = node_gpu.get(node, "")
            tag = f"{count}x {node}" if count > 1 else node
            if gpu_label:
                tag += f" ({gpu_label})"
            parts.append(tag)
        print(f"✅ Workers connected: {n_workers} on {', '.join(parts)}")
        if n_workers < total_workers_expected:
            print(f"⚠️ Requested {total_workers_expected} worker(s), but only {n_workers} connected.")
        print(f"   Summary: {n_workers} worker(s) — total cores={total_cores}, total mem≈{total_mem_gb:.2f}GB")

    if verbose >= 2:
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
    require_token: bool = True,
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

    token = token or uuid.uuid4().hex if require_token else None
    runtime_dir = os.path.join(runtime_dir_base, f"worker-{os.getpid()}")
    os.makedirs(runtime_dir, exist_ok=True)

    cmd = [jupyter_exe, "server"] if jupyter_exe else [sys.executable, "-m", "jupyter", "server"]
    env = os.environ.copy(); env.setdefault("JUPYTER_RUNTIME_DIR", runtime_dir)
    port_arg = "--port=0" if port is None else f"--port={int(port)}"

    cmd += [
        "--no-browser", "--ip=0.0.0.0", port_arg,
        "--ServerApp.port_retries=50",
        "--ServerApp.allow_remote_access=True", f"--ServerApp.allow_origin={allow_origin}",
        f"--ServerApp.runtime_dir={runtime_dir}",
    ]

    if require_token:
        cmd += [
            f"--ServerApp.token={token}",
            f"--IdentityProvider.token={token}",  # Jupyter Server 2.x
        ]
    else:
        cmd += [
            "--ServerApp.token=",  # Empty = no auth (for VSCode compatibility)
            "--ServerApp.password=",
            "--ServerApp.disable_check_xsrf=True",
            "--IdentityProvider.token=",
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
                host_ip = _worker_host_ip(); port_part = f":{parsed.port}" if parsed.port else ""
                netloc = f"{host_ip}{port_part}" if parsed.hostname in {"127.0.0.1", "localhost"} else parsed.netloc
                query = urlencode({"token": token}) if require_token else ""
                parsed = parsed._replace(netloc=netloc, query=query)
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
                    from urllib.parse import urlparse, urlunparse, urlencode
                    parsed = urlparse(candidate)
                    host_ip = _worker_host_ip()
                    port_part = f":{parsed.port}" if parsed.port else ""
                    netloc = f"{host_ip}{port_part}" if parsed.hostname in {"127.0.0.1", "localhost"} else parsed.netloc
                    fallback_token = data.get("token", token) if require_token else None
                    query = urlencode({"token": fallback_token}) if require_token else ""
                    parsed = parsed._replace(netloc=netloc, query=query)
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
    begin_time = launch_cfg.get("begin_time") or None
    nodelist = launch_cfg.get("nodelist") or None
    num_gpus = int(launch_cfg.get("num_gpus") or 0)
    gpu_names = launch_cfg.get("gpu_names") or None
    gres = launch_cfg.get("gres")
    if not gres and num_gpus > 0:
        gres = f"gpu:{num_gpus}"
    launch_kernels = bool(launch_cfg.get("launch_kernels"))
    jupyter_only = bool(launch_cfg.get("jupyter_only"))

    launch_cfg.update(
        {
            "num_jobs": num_jobs,
            "processes": processes,
            "cores_per_process": cores_per_process,
            "threads_per_worker": threads_per_worker,
            "walltime": walltime,
            "begin_time": begin_time,
            "memory": memory,
            "partition": selected_partition,
            "nodelist": nodelist,
            "num_gpus": num_gpus,
            "gpu_names": gpu_names,
            "gres": gres,
            "launch_kernels": launch_kernels,
            "jupyter_only": jupyter_only,
        }
    )

    print_launch_summary(launch_cfg)
    if VENV_MODE == "pyproject":
        project_dir = str(Path(VENV_ACTIVATE).expanduser().resolve())
        venv_name = Path(project_dir).name
        print(f"\n📂 uv sync will run in: {project_dir}")
        print(f"   venv: $VENV_LOCAL/{venv_name}")
        if input("   Is this the correct project directory? (y/n): ").lower() != 'y':
            print("🚫 Launch aborted.")
            return
        if input("   Recreate venv from scratch? (y/N): ").strip().lower() == 'y':
            launch_cfg["_recreate_venv"] = True
    if require_confirmation:
        if input("Proceed with launch? (y/n): ").lower() != 'y':
            print("🚫 Launch aborted.")
            return

    if prompt_to_save:
        maybe_save_named_config(launch_cfg)

    save_last_launch_config(launch_cfg)

    if jupyter_only:
        _execute_jupyter_only(launch_cfg, verbosity)
    else:
        _execute_dask_launch(launch_cfg, verbosity)


def _execute_jupyter_only(launch_cfg: Dict[str, Any], verbosity: int) -> None:
    """Launch a Jupyter server via srun (foreground, no Dask)."""
    script = _build_jupyter_srun_script(recreate_venv=bool(launch_cfg.get("_recreate_venv")))
    srun_cmd = _build_srun_cmd(launch_cfg) + ['bash', '-c', script]
    print(f"\n$ {' '.join(srun_cmd)}\n")
    subprocess.run(srun_cmd, env=_srun_env())


def _execute_dask_launch(launch_cfg: Dict[str, Any], verbosity: int) -> None:
    """Standard Dask cluster launch."""
    num_jobs = launch_cfg["num_jobs"]
    processes = launch_cfg["processes"]
    threads_per_worker = launch_cfg["threads_per_worker"]
    cores_per_process = launch_cfg["cores_per_process"]
    memory = launch_cfg["memory"]
    walltime = launch_cfg["walltime"]
    selected_partition = launch_cfg.get("partition")
    begin_time = launch_cfg.get("begin_time")
    nodelist = launch_cfg.get("nodelist")
    gres = launch_cfg.get("gres")
    gpu_names = launch_cfg.get("gpu_names")
    launch_kernels = launch_cfg.get("launch_kernels", False)

    cluster = client = None
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
            begin_time=begin_time,
            nodelist=nodelist,
            gres=gres,
            gpu_names=gpu_names,
            launch_kernels=False,
            verbose=verbosity,
            print_job_script=True,
        )
        print("\n✅ Dask cluster is running!")
        print(f"   Reconnect with: Client('{client.scheduler.address}')")
        try:
            SCHEDULER_ADDRESS_FILE.write_text(client.scheduler.address)
            print(f"   Scheduler address saved to {SCHEDULER_ADDRESS_FILE}")
        except Exception:
            pass

        if launch_kernels:
            script = _build_jupyter_srun_script(recreate_venv=bool(launch_cfg.get("_recreate_venv")))
            srun_cmd = _build_srun_cmd(launch_cfg) + ['bash', '-c', script]
            print(f"\n$ {' '.join(srun_cmd)}\n")
            subprocess.run(srun_cmd, env=_srun_env())
        else:
            while True:
                time.sleep(3600)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupt received. Shutting down…")
    except Exception as e:
        import traceback
        print(f"\n💥 Error during cluster setup: {e}")
        traceback.print_exc()
    finally:
        print("\n🛑 Shutting down cluster and client…")
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

    jupyter_choice = input("\n🧪 Launch mode: [n]o jupyter, [j]upyter + dask workers, jupyter [o]nly (no dask)? (n/j/o): ").strip().lower()
    launch_kernels = jupyter_choice in ('j', 'o')
    jupyter_only = jupyter_choice == 'o'
    if launch_kernels:
        print(
            "\nℹ️  Defaults updated for interactive Jupyter use:\n"
            "   • SLURM jobs: 1\n   • CPU cores per process: 16\n   • No additional thread limits\n"
        )

    # 2) Parameters
    walltime = prompt_for_value("Enter walltime [HH:MM:SS]", WALLTIME)
    begin_time = None
    schedule_choice = input("Schedule begin time? (y/n): ").strip().lower()
    if schedule_choice == "y":
        default_begin_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        begin_time = prompt_for_value("Enter begin time for SLURM (--begin)", default_begin_time, str)
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

            sorted_info = sorted(nodes_info_part, key=lambda x: x['state'] + str(x.get('cpus_total') or 0))
            available_for_selection: List[Tuple[int, str]] = []
            for idx, i in enumerate(sorted_info, start=1):
                node_name = i['node']
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
                print(f"{idx:<5} {marker} {node_name:<20} {i['state']:<10} {mem_str:<12} {cpu_str:<20} {gpu_str:<22} {user_count:<7}")
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
                    available_for_selection.append((idx, node_name))

            print("=" * 120)
            print(f"✓ = Available for job submission ({len(available_for_selection)} nodes)\n")

            nodelist_choice = input("Enter index or comma-separated nodes/indices (leave empty for any): ").strip()
            if nodelist_choice:
                choices = [c.strip() for c in nodelist_choice.split(",") if c.strip()]
                selected_nodes: List[str] = []
                sorted_node_names = [si['node'] for si in sorted_info]
                for choice in choices:
                    if choice.isdigit():
                        nidx = int(choice)
                        if 1 <= nidx <= len(sorted_node_names):
                            selected_nodes.append(sorted_node_names[nidx - 1])
                        else:
                            print(f"⚠️  Invalid index '{choice}'; skipping.")
                    else:
                        if choice in nodes_in_part:
                            selected_nodes.append(choice)
                        else:
                            print(f"⚠️  Unknown node '{choice}'; skipping.")
                if selected_nodes:
                    seen = set()
                    unique_nodes: List[str] = []
                    for node in selected_nodes:
                        if node not in seen:
                            unique_nodes.append(node)
                            seen.add(node)
                    nodelist = ",".join(unique_nodes)
                else:
                    print("⚠️  No valid nodes selected; proceeding without a specific node.")
    else:
        print("\nℹ️  Skipping node-specific selection; SLURM will apply its default partition rules.")

    num_gpus = prompt_for_value("Enter number of GPUs per process (0 for CPU-only)", 0, int)
    gres = f"gpu:{num_gpus}" if num_gpus > 0 else None
    gpu_names_input = input("Optional GPU model filter (comma-separated, leave empty for any): ").strip()
    gpu_names = [n.strip() for n in gpu_names_input.split(",") if n.strip()] if gpu_names_input else None

    launch_config = {
        "partition": selected_partition,
        "walltime": walltime,
        "begin_time": begin_time,
        "num_jobs": num_jobs,
        "processes": processes,
        "cores_per_process": cores_per_process,
        "threads_per_worker": threads_per_worker,
        "memory": memory,
        "nodelist": nodelist,
        "num_gpus": num_gpus,
        "gpu_names": gpu_names,
        "gres": gres,
        "launch_kernels": launch_kernels,
        "jupyter_only": jupyter_only,
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
