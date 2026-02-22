#!/usr/bin/env bash
set -euo pipefail

CONFIGURATION="${1:-Release}"
OUT_DIR="${2:-/Applications}"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "xcodegen not found. Install with: brew install xcodegen" >&2
  exit 1
fi

xcodegen generate --spec project.yml

xcodebuild \
  -project SlurmHUD.xcodeproj \
  -scheme SlurmHUD \
  -configuration "${CONFIGURATION}" \
  "CONFIGURATION_BUILD_DIR=${OUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${OUT_DIR}/SlurmHUD.app/Contents/Resources"
cp "${SCRIPT_DIR}/Sources/SlurmHUDApp/AppIcon.icns" "${OUT_DIR}/SlurmHUD.app/Contents/Resources/AppIcon.icns"

echo "App bundle: ${OUT_DIR}/SlurmHUD.app"
