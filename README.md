# slurmcli

Command-line helpers for launching and monitoring Dask clusters on SLURM.

## Install 

### Editable (recommended)
```pip install -e git+https://github.com/japhba/slurmcli.git#egg=slurmcli```

### Static
```pip install git+https://github.com/japhba/slurmcli.git#egg=slurmcli```

## Features

- Interactive prompts for choosing partitions and worker configuration.
- Optional remote Jupyter server bootstrap on each worker, with automatic
  credential discovery.
- Simple client helper for connecting to a running scheduler.

Run `slurmcli` after installation to open the interactive launcher.

## Quickstart

1. Install (editable)
```bash
pip install -e git+https://github.com/japhba/slurmcli.git#egg=slurmcli
```
2. Set any preferred defaults (optional)
```
export SLURMCLI_VENV_ACTIVATE=$HOME/myproject/.venv/bin/activate
```

3. Launch on a SLURM login node, it will guide you through the initial setup
```
slurmcli
```

4. Connect from your notebook/script with the printed URL or Jupter kernel URL
(example is also at ```slurmcli/run_slurm.py```
```
python - <<'PY'
from dask.distributed import Client
client = Client(open(".dask_scheduler_address").read().strip())
print("Connected to", client.scheduler.address)
PY
```
5. Shut down with Ctrl+C when finished, this will cancel your cluster jobs.


## Usage

### 1. Prerequisites

- Run from a SLURM login node with `sinfo` and `sbatch` available.
- Ensure you have credentials to submit jobs to your preferred partitions.
- Optional but recommended: create a Python virtual environment on the shared filesystem that workers can activate.

### 2. Launch the interactive workflow

```bash
$ slurmcli
```

The first time you run the command you will be prompted for a few persistent settings (email, scheduler host/ports, worker virtual environment, …). These answers are cached in `~/.slurmcli` and can also be provided via environment variables:

| Key              | Environment variable           | Description                              |
| ---------------- | ------------------------------ | ---------------------------------------- |
| `mail`           | `SLURMCLI_MAIL`                | Address to receive SLURM notifications   |
| `scheduler_host` | `SLURMCLI_SCHEDULER_HOST`      | Hostname/IP reported to Dask clients     |
| `scheduler_port` | `SLURMCLI_SCHEDULER_PORT`      | Dask scheduler port (default `8786`)     |
| `dashboard_port` | `SLURMCLI_DASHBOARD_PORT`      | Dask dashboard port (default `8787`)     |
| `jupyter_port`   | `SLURMCLI_JUPYTER_PORT`        | Base port for worker-side Jupyter servers|
| `venv_activate`  | `SLURMCLI_VENV_ACTIVATE`       | Path to `activate` script on workers     |

Set any of these before calling `slurmcli` to skip the prompt:

```bash
export SLURMCLI_VENV_ACTIVATE=$HOME/myproject/.venv/bin/activate
export SLURMCLI_MAIL=you@example.com
slurmcli
```

### 3. Configure the job

The launcher shows a partition summary (pulled via `sinfo`) and lets you:

1. Choose the target partition/queue.
2. Decide whether to bootstrap a Jupyter server on each worker.
3. Enter walltime, number of jobs, processes per job, cores/threads, memory, and GPU needs.
4. Optionally pin the allocation to a specific node using its index.
5. Confirm the launch summary before submission.

After submission, `slurmcli` creates a `dask-jobqueue` `SLURMCluster`, prints the scheduler address, and saves it in `.dask_scheduler_address` for convenience.

### 4. Connect from a Python session

```python
from dask.distributed import Client

# replace with the address printed by slurmcli (saved to .dask_scheduler_address)
client = Client("tcp://scheduler-host:8786")
```

If you opted into worker-side Jupyter servers, their URLs are printed after startup.

### 5. Shutting down

Press `Ctrl+C` in the `slurmcli` session to gracefully close the Dask client, shut down any worker Jupyter servers, and cancel the SLURM jobs.

## Tips

- Want different defaults for a single run? Just overtype the pre-filled values in the prompts.
- To reset cached credentials, delete `~/.slurmcli`.
- Logs and scheduler address are written to the current working directory; run `slurmcli` from a project folder if you want to keep those artifacts.
- See `slurmcli/run_slurm.py` for a complete example that reconnects to a scheduler, inspects worker resources, and runs a distributed π-estimation workload using JAX and Dask.
