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

