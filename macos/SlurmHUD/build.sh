#!/usr/bin/env bash
set -euo pipefail

CONFIGURATION="${1:-Release}"
OUT_DIR="${2:-$(pwd)/build}"

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

echo "App bundle: ${OUT_DIR}/SlurmHUD.app"
