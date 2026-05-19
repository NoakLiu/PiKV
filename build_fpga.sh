#!/usr/bin/env bash
# Build PiKV-FPGA: C host, RTL sim, Vivado project / bitstream
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/core/fpga"

export PIKV_PART="${PIKV_PART:-xcu55c-fsvh2892-2L-e}"
export PIKV_JOBS="${PIKV_JOBS:-8}"

case "${1:-all}" in
  host)
    make host
    ;;
  sim)
    make sim
    ;;
  sim-soc)
    make sim-soc
    ;;
  test)
    make test
    ;;
  vivado)
    make vivado
    ;;
  bitstream)
    make bitstream
    ;;
  bd)
    make bd
    ;;
  clean)
    make clean
    ;;
  all)
    make host
    if command -v iverilog >/dev/null 2>&1; then
      make sim || true
      make sim-soc || true
    fi
    make test
    ;;
  *)
    echo "Usage: $0 [all|host|sim|sim-soc|test|vivado|bitstream|bd|clean]"
    echo "  PIKV_PART=$PIKV_PART  PIKV_JOBS=$PIKV_JOBS"
    exit 1
    ;;
esac

echo "PiKV-FPGA: done (part=$PIKV_PART)"
