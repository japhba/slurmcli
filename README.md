# slurmcli

A Python toolkit for interactive SLURM cluster management and distributed computing via Dask. Includes a CLI launcher, a programmatic API (`get_client`, `smap`), a cluster status system (JSON + browser HUD), and a native macOS menu-bar app.

## Directory Structure

```
slurmcli/
├── pyproject.toml
├── slurmcli/
│   ├── __init__.py            # Lazy imports: main_interactive, main_cli, get_client, smap
│   ├── slurm_parser.py        # Pure-text SLURM output parsers (GRES, scontrol)
│   ├── cluster_status.py      # SLURM queries + JSON snapshot builder
│   ├── launch_slurm.py        # Interactive TUI launcher + get_cluster()
│   ├── run_slurm.py           # Programmatic API: get_client(), smap(), log tailing
│   ├── status_cli.py          # `slurmcli-status` CLI (JSON to stdout)
│   └── hud.py                 # Browser-based HUD (stdlib HTTP server, self-contained SPA)
├── macos/SlurmHUD/            # Native macOS menu-bar app (Swift, communicates via SSH)
└── Formula/slurmhud.rb        # Homebrew formula for the macOS app
```

## Dependencies

`dask[distributed]`, `dask-jobqueue`, `numpy`, `colorama`, `tqdm`. Python >= 3.9.

## CLI Entry Points

### `slurmcli` (→ `launch_slurm:main_cli`)

The primary interactive launcher.

```
slurmcli [options] [config_name]

Options:
  --local             Launch a local Dask cluster (no SLURM)
  --hud               Start the browser-based HUD web server
  --hud-host HOST     HUD bind host (default: 0.0.0.0)
  --hud-port PORT     HUD port (default: 8765)
  --hud-refresh SECS  HUD auto-refresh interval (default: 600)
  -v / -vv / -vvv / -vvvv   Verbosity levels
  config_name         Name of a saved config for non-interactive launch
```

Verbosity levels:
- default: concise partition summary
- `-v`: per-node lines (when ≤10 nodes in bucket)
- `-vv`: show admin/hidden partitions (via `sinfo -a`), job names per node
- `-vvv`: also show user lists per node
- `-vvvv`: also show reserved TRES resources for each job

### `slurmcli-status` (→ `status_cli:main`)

Emits a JSON cluster snapshot to stdout. Consumed by the macOS app via SSH.

```
slurmcli-status [--pretty] [-v/-vv/-vvv/-vvvv]
```

Verbosity thresholds: `show_all` at -vv, `include_jobs` at -vv, `include_job_resources` at -vvvv.

---

## Module Reference

### `slurm_parser.py` — Pure Text Parsers (No IO)

- **`split_gres_entries(gres_value: str) -> List[str]`** — Comma-split respecting parenthesized GRES groups like `gpu:h100:8(S:0,1)`.
- **`extract_gpu_total(gres_value: str) -> int`** — Total GPU count from a GRES string (e.g. `gpu:h100:8` → 8). Handles multi-entry GRES and parenthesized socket annotations.
- **`extract_gpu_labels(gres_value: str) -> List[str]`** — GPU model labels (e.g. `gpu:h100:8` → `["h100"]`). Deduplicated, order-preserved.
- **`parse_scontrol_output(text: str) -> Dict[str, str]`** — Parses `scontrol show node` output into `{Key: Value}` dict. Handles multi-line output and balanced parentheses in values.
- **`build_node_info(fields: Dict, node_name: str) -> Dict`** — Transforms scontrol fields into normalized dict: `node`, `memory_gb`, `cpus_total`, `cpus_alloc`, `cpu_arch`, `gpus`, `gpus_total`, `gpus_in_use`, `gpu_labels`, `state`, `users`, `user_count`, `jobs`. Falls back to `AllocTRES` for GPU-in-use when `GresUsed` is absent.

### `cluster_status.py` — SLURM Queries + Snapshot Builder

**SLURM commands used:** `sinfo`, `scontrol show node`, `scontrol show hostnames`, `scontrol show job`, `squeue`

Core functions:

- **`expand_nodelist(nodelist: str) -> List[str]`** — Calls `scontrol show hostnames` to expand compact range notation (e.g. `node[1-3]`).
- **`parse_cpu_string(cpu_str: str) -> Optional[Dict[str, int]]`** — Parses SLURM's `A/I/O/T` CPU string from `sinfo`.
- **`parse_memory_string(mem_str: str) -> Optional[float]`** — Converts SLURM memory strings to float GB.
- **`get_partitions(show_all=False) -> Optional[Dict[int, Dict]]`** — Runs `sinfo -h -o '%P|%t|%C|%G|%N'` (or `-a` for all). Filters to idle/mixed unless `show_all`. Returns `{idx: {partition, states, cpu_info, nodes}}`.
- **`get_jobs_by_node() -> Dict[str, List[Dict]]`** — Runs `squeue --all -h -o '%i|%u|%j|%M|%L|%N'`. Returns `{node: [{id, user, name, elapsed, time_left}]}`.
- **`get_node_user_map() -> Dict[str, set]`** — Runs `squeue --all -h -o '%u|%N'`. Cached node expansion.
- **`query_users_for_node(node: str) -> List[str]`** — Fallback: `squeue --all -h -w <node> -o %u`.
- **`get_job_resource_details(job_id: int) -> Dict`** — Calls `scontrol show job`. Returns AllocTRES, ReqTRES, ReqMem. In-process cached (`JOB_RESOURCE_CACHE`).
- **`get_detailed_node_info(nodes, *, include_jobs, include_job_resources) -> Dict[str, Dict]`** — Calls `scontrol show node` per node. Merges user/job data if requested.
- **`build_cluster_snapshot(*, show_all, include_jobs, include_job_resources) -> Dict`** — Top-level function. Assembles full JSON-serializable snapshot used by `slurmcli-status` and the HUD.

Helpers: `format_gpu_labels`, `aggregate_gpu_counts`, `format_gpu_counts`, `_format_job_time_info`, `_summarize_job_resources`.

**Snapshot schema:**
```json
{
  "generated_at": "ISO8601Z",
  "partitions": [{
    "idx": 1,
    "partition": "gpu_lowp",
    "states": ["idle", "mixed"],
    "cpu_info": {"allocated": 0, "idle": 128, "total": 128},
    "nodes": ["gpu-node01"],
    "configs": [{"mem_gb": 512, "cpus": 128, "gpus": 8, "node_count": 1, "available_count": 1,
                 "gpu_in_use": 0, "gpu_total": 8, "gpu_counts": [["h100", 8]],
                 "users_min": 0, "users_max": 0, "nodes": ["gpu-node01"]}],
    "node_details": [{"node": "gpu-node01", "state": "idle", "memory_gb": 512.0,
                      "cpus_alloc": 0, "cpus_total": 128,
                      "gpus_in_use": 0, "gpus_total": 8,
                      "gpu_labels": ["h100"], "gpu_label_text": "h100",
                      "users": [], "user_count": 0,
                      "jobs": []}]
  }],
  "show_all": false,
  "include_jobs": true,
  "include_job_resources": false
}
```

### `launch_slurm.py` — Interactive Launcher + Cluster Lifecycle

The core of the package. Handles configuration, TUI interaction, venv setup, and Dask cluster creation.

#### Configuration System

Config file: `~/.slurmcli` (JSON). Precedence: env var > file > default.

| Config key | Env variable | Default |
|---|---|---|
| `mail` | `SLURMCLI_MAIL` | `""` |
| `scheduler_host` | `SLURMCLI_SCHEDULER_HOST` | `"192.168.240.53"` |
| `scheduler_port` | `SLURMCLI_SCHEDULER_PORT` | `8786` |
| `dashboard_port` | `SLURMCLI_DASHBOARD_PORT` | `8787` |
| `jupyter_port` | `SLURMCLI_JUPYTER_PORT` | `11833` |
| `jupyter_require_token` | `SLURMCLI_JUPYTER_REQUIRE_TOKEN` | `False` |
| `venv_activate` | `SLURMCLI_VENV_ACTIVATE` | `/nfs/.../recurrent_feature/.venv` |

Derived key: `venv_mode` — `"activate"` (standard venv) or `"pyproject"` (runs `uv sync` per worker).

**Side effect:** Importing `launch_slurm` calls `load_credentials()` at module level, which prompts interactively if stdin is a tty and no config exists.

#### Persistent Files

| File | Purpose |
|---|---|
| `~/.slurmcli` | Credentials/config |
| `~/.slurmcli_last_launch.json` | Last launch params (auto-saved) |
| `~/.slurmcli_saved_configs.json` | Named saved configurations |
| `~/.dask_scheduler_address` | Plain text: most recent scheduler address |
| `~/.dask_scheduler_addresses.json` | History map: `last`, `gpu`, `cpu`, `local`, `by_id`, `history` |

#### Key Functions

- **`load_credentials() -> Dict`** — Loads config from env vars + `~/.slurmcli`. Prompts for venv interactively. Tab completion for paths. Detects venv suggestions from cwd.
- **`_detect_venv_suggestions(cwd) -> List[Dict]`** — Scans cwd for `pyproject.toml` and `.venv`/`venv` dirs.
- **`_build_venv_prologue(venv_activate, venv_mode, ...) -> Tuple[List[str], Optional[str]]`** — Builds shell commands prepended to SLURM job scripts. `pyproject` mode: `cd <dir>; export UV_PROJECT_ENVIRONMENT=...; uv sync; source ...`. `activate` mode: `source <path>`. Optionally appends Jupyter startup.
- **`_jupyter_prologue_lines(port, require_token, ...) -> List[str]`** — Generates shell lines to start `jupyter server` (background via `nohup` or foreground via `exec`).
- **`_normalize_slurm_memory(memory: str) -> str`** — Normalizes "32GB" → "32G" for SLURM.
- **`_resolve_gpu_nodelist(gpu_types, *, partition) -> List[str]`** — Runs `sinfo -h -o '%N|%G'`, filters nodes matching GPU labels, returns expanded list. Used instead of SLURM constraints to avoid partition incompatibilities.
- **`is_port_available(port) -> bool`** — Checks via `socket.bind`.
- **`get_cluster(...) -> Tuple[cluster, Client]`** — Main cluster creation:
  - `local=True`: `LocalCluster` (multiprocessing or threading depending on Jupyter).
  - `local=False`: `dask_jobqueue.SLURMCluster` with all params → sbatch directives; `cluster.scale(n=num_jobs)`.
  - Saves scheduler address. Waits for workers with queue-ahead reporting and log tailing.
- **`_wait_for_workers_with_queue(client, ...)`** — Polls `client.wait_for_workers()` in a loop, tails SLURM logs, reports queue position.
- **`_tail_slurm_logs(log_dir, job_ids, offsets) -> Dict`** — Reads new content from `<log_dir>/*-<jobid>.{out,err}` via byte offsets.
- **`_get_squeue_rows(partition) -> Optional[List[Dict]]`** — Runs `squeue --all`, falls back to unsorted if `--sort=-P` fails.
- **`_collect_slurm_job_ids(cluster, timeout, poll) -> List[str]`** — Polls `dask_jobqueue` cluster for assigned SLURM job IDs.
- **`_report_queue_ahead(job_ids, ...)`** — Shows how many jobs are ahead in queue with details.
- **`_build_srun_cmd(launch_cfg) -> List[str]`** — Builds `srun` command for Jupyter-only sessions (no Dask).
- **`_srun_env() -> Dict`** — `os.environ` with all `SLURM_*` vars stripped (prevents nested-job issues).
- **`execute_launch(config, verbosity, ...)`** — Validates/normalizes config, prints summary, confirms, saves history, dispatches to `_execute_jupyter_only` or `_execute_dask_launch`.
- **`_execute_jupyter_only(launch_cfg, verbosity)`** — Runs `srun ... bash -c '<venv+jupyter>'` foreground.
- **`_execute_dask_launch(launch_cfg, verbosity)`** — Calls `get_cluster(local=False, ...)`, keeps alive until Ctrl+C.

#### Interactive TUI Flow (`main_interactive`)

1. Check for last launch config, offer to re-launch
2. Run `get_partitions`, display partition table
3. Prompt: choose partition by number
4. Prompt: launch mode (no jupyter / jupyter+dask / jupyter-only)
5. Prompt: walltime, begin time, num_jobs, processes, cores_per_process, thread limit, memory
6. Display detailed node table for chosen partition, offer node selection
7. Prompt: GPUs per process, GPU model filter
8. Call `execute_launch` with `prompt_to_save=True`

#### Cluster ID System

Each launch gets a random human-readable ID from a 20-word list (e.g. `"brisk-otter"`) for identification in address maps.

### `run_slurm.py` — Programmatic API

- **`get_client(...) -> Union[Client, Tuple[Client, cluster]]`** — Unified entry point for notebooks/scripts.
  - `reuse_existing=True`: reconnects to a saved scheduler address. Validates cluster type (local/cpu/gpu) matches target.
  - `reuse_existing=False` (default): launches new cluster via `get_cluster()`.
  - `select_scheduler=True`: prompts user to pick from history.
  - Registers cleanup via `atexit`, `weakref.finalize`, SIGTERM/SIGINT when `shutdown_on_exit=True`.
  - Starts background log tailer thread if `tail_logs=True`.
  - `return_cluster=True` returns `(client, cluster)`.

  Key kwargs: `local`, `cluster_id`, `n_workers`/`num_workers`, `partition`, `walltime`, `begin_time`, `num_jobs`, `processes`, `cores_per_process`, `threads_per_worker`, `memory`, `nodelist`, `num_gpus`, `gpu_names`, `gres`, `account`, `job_name`, `log_dir`, `venv_activate`, `verbose`, `print_job_script`, `tail_logs`, `log_interval`, `include_scheduler_logs`, `max_log_lines`, `shutdown_on_exit`, `return_cluster`, `log_worker_details`, `worker_probe_attempts`, `worker_probe_delay`, `ensure_gc_cleanup`, `launch_if_no_workers`, `cluster_type`, `connect_timeout`, `local_processes`, `reuse_existing`, `select_scheduler`, `wait_for_workers`, `wait_timeout`, `wait_poll`.

- **`smap(fun, in_axes=0, out_axes=0, *, sequential=False, client=None, show_progress=True, progress_mode="auto", cache_check="auto", return_info=False, disconnect=True, gpu_names=None, **client_kwargs)`** — JAX-style vectorized map over Dask workers.
  - Maps inputs along `in_axes` (int or list of ints/None per arg; None = broadcast).
  - Cache-check protocol: if `fun.check_call_in_cache` exists and returns True for all slices, skips Dask and runs locally.
  - `sequential=True`: forces local for-loop (requires `local=True`).
  - Creates a Dask client via `get_client(**client_kwargs)` if none provided; closes after gather if `disconnect=True`.
  - Uses `_TqdmProgressBar` (custom `distributed.ProgressBar` subclass) for tqdm progress.
  - Returns `np.stack`ed results when `out_axes=0`, raw list when `out_axes=None`.
  - Returns `(output, info_dict)` if `return_info=True`.

- **`list_known_schedulers(max_entries=20, *, print_out=True)`** — Reads `~/.dask_scheduler_addresses.json` history and prints numbered list.
- **`close_workers()`** — Connects to saved scheduler, calls `client.retire_workers()`.
- **`stop_log_tailing()`** — Signals the background log-tailer daemon thread to stop.

**`_TqdmProgressBar`:** Subclass of `distributed.diagnostics.progressbar.ProgressBar`. Uses tqdm (auto/notebook/std depending on `progress_mode`). Overrides `_draw_bar` and `_draw_stop`.

### `hud.py` — Browser-Based HUD

A pure-stdlib HTTP server (`http.server`) serving a self-contained HTML/JS/CSS SPA.

- **`run_hud(host, port, refresh_seconds, verbosity)`** — Starts the server.
- Routes: `GET /` (HTML SPA), `GET /api/status` (JSON snapshot), `GET /api/health`.
- Frontend polls `/api/status` on a timer, renders dark-themed partition cards with color-coded node grid (green=idle/mixed, orange=draining, red=down).
- Launched via `slurmcli --hud` or `python -m slurmcli.hud`.

### `status_cli.py` — JSON Status CLI

Thin wrapper: parses `-v` flags and `--pretty`, calls `build_cluster_snapshot`, dumps JSON to stdout.

---

## macOS SlurmHUD App (`macos/SlurmHUD/`)

Native Swift menu-bar app, independent of the Python package at runtime. Communicates over SSH.

- Reads `~/.slurmhud.json`: `host`, `refresh_seconds`, `timeout_seconds`, `command` (default: `"slurmcli-status -vvvv"`)
- `StatusFetcher` spawns `/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=N <host> <command>` via `SSHRunner`
- Parses JSON into Swift models (`ClusterSnapshot`, `Partition`, `NodeDetail`, `JobDetail`) — snake_case fields mirror `build_cluster_snapshot` output
- Caches to `~/.slurmhud-cache.json` for the WidgetKit widget (which can't run long processes)
- Displays partition/node summary in the menu bar

Components: `Config.swift` (config + cache), `Models.swift` (data models), `SSHRunner.swift` (subprocess wrapper), `StatusFetcher.swift` (ObservableObject), `SlurmHUDApp.swift` (entry point), `StatusView.swift` (main UI), `SettingsView.swift` (settings form), `SlurmHUDWidget.swift` (WidgetKit timeline widget).

---

## SLURM Commands Used

| Command | Where | Purpose |
|---|---|---|
| `sinfo -h -o '%P\|%t\|%C\|%G\|%N'` | `get_partitions` | List partitions: state, CPUs, GRES, nodes |
| `sinfo -a -h -o '%P\|%t\|%C\|%G\|%N'` | `get_partitions(show_all=True)` | Same, including hidden/admin partitions |
| `sinfo -h -o '%N\|%G'` | `_resolve_gpu_nodelist` | Find nodes matching GPU type |
| `scontrol show hostnames <nodelist>` | `expand_nodelist` | Expand compact node range notation |
| `scontrol show node <name>` | `get_detailed_node_info` | Full node info (mem, CPUs, GPUs, state, GRES) |
| `scontrol show job <id>` | `get_job_resource_details` | AllocTRES, ReqTRES, ReqMem per job |
| `squeue --all -h -o '%i\|%u\|%j\|%M\|%L\|%N'` | `get_jobs_by_node` | All jobs: user, name, elapsed, time_left, nodes |
| `squeue --all -h -o '%u\|%N'` | `get_node_user_map` | User→node mapping |
| `squeue --all -h -w <node> -o '%u'` | `query_users_for_node` | Users on a specific node (fallback) |
| `squeue --all -h -o '%i\|%P\|%T\|%Q\|%R\|%u\|%j'` | `_get_squeue_rows` | Full queue for ahead-of-us reporting |
| `sbatch` (via `dask_jobqueue.SLURMCluster`) | `get_cluster` | Submit Dask worker jobs |
| `srun -J slurmcli-jupyter ...` | `_execute_jupyter_only` | Interactive Jupyter session |

---

## Design Notes

- **Layered architecture:** `slurm_parser` (pure text) → `cluster_status` (subprocess IO) → `launch_slurm` (cluster lifecycle) → `run_slurm` (programmatic API). `status_cli` and `hud` are output adapters. macOS app is a remote SSH consumer.
- **Module-level side effects:** Importing `launch_slurm` triggers `load_credentials()`, which prompts interactively if no config exists and stdin is a tty.
- **GPU filtering:** Uses `sinfo` GRES label matching + `--nodelist` rather than SLURM constraints, avoiding partition incompatibilities.
- **Port conflict handling:** Checks `socket.bind` before creating Dask scheduler; falls back to ephemeral port.
- **SLURM env stripping:** `_srun_env()` removes inherited `SLURM_*` vars to prevent nested-job absorption.
- **`smap` cache protocol:** Functions with `check_call_in_cache(*args) -> bool` can skip Dask entirely when all slices are cached.
- **Log tailing:** Two mechanisms — byte-offset SLURM file tailing during startup wait, and a background daemon thread for Dask worker/scheduler log streaming.
- **Dask-jobqueue integration:** `get_cluster` translates kwargs into `SLURMCluster` params → `#SBATCH` directives: `queue`→`--partition`, `cores`→`--ntasks-per-node`+`--cpus-per-task`, `memory`→`--mem`, `walltime`→`--time`, and raw directives for `--nodelist`, `--begin`, `--gpus-per-task`, `--gpu-bind`.
- **Venv modes:** `activate` (source a venv) vs `pyproject` (cd + UV_PROJECT_ENVIRONMENT + uv sync + source). In `pyproject` mode, `venv_activate` from `~/.slurmcli` is ignored at import time; instead, the project directory is auto-detected from the active venv's editable install metadata (`direct_url.json`), falling back to `Path.cwd()`. This means `smap` always targets the project whose venv you're running in, regardless of the `~/.slurmcli` config. Convention: venvs are built with `UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}"`, so the venv basename matches the project directory basename.
