"""slurmcli package."""

import logging

logger = logging.getLogger("slurmcli")
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

__all__ = ["main_interactive", "main_cli", "get_client", "smap"]

def __getattr__(name):
    if name in ("main_interactive", "main_cli"):
        from .launch_slurm import main_interactive, main_cli
        return main_interactive if name == "main_interactive" else main_cli
    if name in ("get_client", "smap"):
        from .run_slurm import get_client, smap
        return get_client if name == "get_client" else smap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
