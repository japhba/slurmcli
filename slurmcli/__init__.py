"""slurmcli package."""

import logging

logger = logging.getLogger("slurmcli")
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

from .launch_slurm import main_interactive, main_cli
from .run_slurm import get_client, vmap

__all__ = ["main_interactive", "main_cli", "get_client", "vmap"]
