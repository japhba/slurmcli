SlurmHUD (macOS menu bar app + widget)
======================================

This folder contains an XcodeGen project definition and Swift sources for a
menu bar app and a WidgetKit widget that fetch SLURM cluster status via SSH.

Config
------
- Config file: shared App Group container (legacy `~/.slurmhud.json` is migrated on first launch)
- Example:
  {
    "host": "mycluster",
    "refresh_seconds": 600,
    "timeout_seconds": 120,
    "command": "slurmcli-status -vv",
    "widget_partition": "gpu_lowp"
  }

The app writes config and widget cache data into an App Group container so the
widget can read it reliably. Existing legacy files in your home directory are
migrated automatically the first time the app or widget accesses them.
Set `widget_partition` to show only one partition in the widget; omit it or
leave it blank to show all partitions.

Build (local)
-------------
1) Install xcodegen: `brew install xcodegen`
2) Generate Xcode project:
   `xcodegen generate --spec project.yml`
3) Open `SlurmHUD.xcodeproj` in Xcode and build/run.

Signing / App Group
-------------------
- The widget can compile and preview without proper signing, but it will not be
  able to read the shared config/cache unless `SlurmHUD` and `SlurmHUDWidget`
  are signed with the same `Development Team`.
- Both targets must also include the same `App Groups` entitlement entry.
- If this is wrong, the widget typically shows the “Open SlurmHUD to fetch
  cluster data.” state even while the app itself has live data.
- After changing signing or App Group settings, run the `SlurmHUD` app scheme
  once, refresh the app, then re-add the widget if macOS is still showing an
  older timeline.

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
