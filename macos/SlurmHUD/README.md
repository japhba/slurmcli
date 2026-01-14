SlurmHUD (macOS menu bar app + widget)
======================================

This folder contains an XcodeGen project definition and Swift sources for a
menu bar app and a WidgetKit widget that fetch SLURM cluster status via SSH.

Config
------
- Config file: `~/.slurmhud.json`
- Example:
  {
    "host": "mycluster",
    "refresh_seconds": 600,
    "timeout_seconds": 10,
    "command": "slurmcli-status -vv"
  }

The app writes the config file when you save settings from the menu.

Build (local)
-------------
1) Install xcodegen: `brew install xcodegen`
2) Generate Xcode project:
   `xcodegen generate --spec project.yml`
3) Open `SlurmHUD.xcodeproj` in Xcode and build/run.

Build (CLI)
-----------
From this directory:
`./build.sh Release`
The app bundle is written to `./build/SlurmHUD.app`.

CLI on the remote host
----------------------
Make sure `slurmcli-status` is on the remote PATH. It is included in this repo
and can be installed with `pip install -e .` or as a package.

Homebrew (draft)
----------------
If you want a Homebrew build, generate the Xcode project and then point a
formula at `xcodebuild` with an app bundle install step. A minimal formula can
live under `Formula/slurmhud.rb` in this repo.
