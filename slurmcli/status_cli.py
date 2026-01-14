from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from slurmcli.cluster_status import build_cluster_snapshot


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit SLURM cluster status JSON")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv, -vvv, -vvvv)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    show_all = args.verbose >= 2
    include_jobs = args.verbose >= 2
    include_job_resources = args.verbose >= 4
    snapshot = build_cluster_snapshot(
        show_all=show_all,
        include_jobs=include_jobs,
        include_job_resources=include_job_resources,
    )
    if args.pretty:
        json.dump(snapshot, sys.stdout, indent=2, sort_keys=False)
    else:
        json.dump(snapshot, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
