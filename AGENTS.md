# slurmcli – Notes for AI agents

## Jupyter + Dask: tqdm forwarding workaround

`distributed.Client.register_plugin()` causes immediate worker disconnection
("Received heartbeat from unregistered worker") when the Dask scheduler runs
in-process inside a Jupyter kernel. The root cause is a conflict between
Jupyter's tornado event loop and the async round-trip that `register_plugin`
triggers between scheduler and workers.

**Fix (run_slurm.py `smap`):** When `_is_jupyter()` is True, the
`_TqdmForwardPlugin` is *not* registered via `register_plugin`. Instead, the
user's function is wrapped so that `plugin.setup(worker)` runs lazily on the
worker when the first task executes. The client-side `_TqdmForwarder`
(which uses `subscribe_topic`) works fine in both environments.

Outside Jupyter (CLI, scripts), `register_plugin` is used as before.

**Symptoms if this regresses:**
- Workers connect ("Workers connected: N") then immediately become
  "unregistered" with `CommClosedError` / `StreamClosedError`.
- `smap` hangs at 0% progress.
- The same code works from a standalone Python script (`verbose=2`).

**Diagnosis:** Disable the `if verbose >= 1:` block in `smap._mapped` that
sets up tqdm forwarding. If workers stay connected, the forwarding is the
cause.
