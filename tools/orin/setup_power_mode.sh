#!/bin/bash
# Jetson AGX Orin power-mode setup for the AR navigation pipeline.
#
# Run this once after boot on the Orin (requires sudo). It selects a fixed
# power mode (MAXN by default) and locks clocks to their maximum for that
# mode, so HLoc inference latency is consistent run-to-run instead of
# varying with the default dynamic governor.
#
# Usage:
#   ./setup_power_mode.sh          # MAXN (~60W, max performance)
#   ./setup_power_mode.sh 30W      # 30W mode, still comfortably above 0.5Hz target
#   ./setup_power_mode.sh 15W      # 15W mode, lowest power
set -euo pipefail

MODE="${1:-MAXN}"

declare -A MODE_IDS=(
  ["MAXN"]=0
  ["50W"]=1
  ["30W"]=2
  ["15W"]=3
)

if [[ -z "${MODE_IDS[$MODE]+x}" ]]; then
  echo "Unknown mode '$MODE'. Valid: ${!MODE_IDS[*]}" >&2
  echo "Run 'sudo nvpmodel -q --verbose' to see the modes available on this board," >&2
  echo "since exact IDs/names can vary by JetPack version." >&2
  exit 1
fi

echo "Setting nvpmodel to $MODE (id ${MODE_IDS[$MODE]})..."
sudo nvpmodel -m "${MODE_IDS[$MODE]}"

echo "Locking clocks to max for this mode (jetson_clocks)..."
sudo jetson_clocks

echo
echo "Current status:"
sudo nvpmodel -q
sudo jetson_clocks --show
