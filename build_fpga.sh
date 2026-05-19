#!/usr/bin/env bash
# Build PiKV-FPGA C host library and optional Verilog simulation
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/core/fpga"

case "${1:-all}" in
  host)
    make host
    ;;
  sim)
    make sim
    ;;
  test)
    make test
    ;;
  clean)
    make clean
    ;;
  all)
    make host
    if command -v iverilog >/dev/null 2>&1; then
      make sim || true
    fi
    make test
    ;;
  *)
    echo "Usage: $0 [all|host|sim|test|clean]"
    exit 1
    ;;
esac

echo "PiKV-FPGA build done. Library: core/fpga/libpikv_fpga.so"
