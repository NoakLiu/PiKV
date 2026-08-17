#!/usr/bin/env bash
# PiKV CUDA build (2026) — routing + full compression + scheduling kernels
#
# Usage (from repo root):
#   ./scripts/build_cuda.sh              # release build (full compression)
#   ./scripts/build_cuda.sh release
#   ./scripts/build_cuda.sh debug
#   ./scripts/build_cuda.sh profile
#   ./scripts/build_cuda.sh compression  # alias of release with full suite
#   ./scripts/build_cuda.sh test
#   ./scripts/build_cuda.sh test-py
#   ./scripts/build_cuda.sh install
#   ./scripts/build_cuda.sh install-user
#   ./scripts/build_cuda.sh clean
#   ARCH_FLAGS='-arch=native' ./scripts/build_cuda.sh release
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DIR="$ROOT/core/cuda"
cd "$ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*" >&2; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

check_cuda() {
  info "Checking CUDA toolkit..."
  if ! command_exists nvcc; then
    err "nvcc not found. Install CUDA Toolkit 12.x (matches PiKV install defaults)."
    exit 1
  fi
  local ver
  ver="$(nvcc --version | grep release | sed 's/.*release \([0-9.]*\).*/\1/')"
  ok "nvcc $ver"
  if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | while read -r line; do
      info "GPU: $line"
    done
  else
    warn "nvidia-smi not found (drivers may be missing)"
  fi
}

build_mode() {
  local mode="${1:-release}"
  info "Building CUDA kernels (mode=$mode) — includes full compression suite:"
  info "  pikv_kernels + routing + compression (LoRA/Quant/Pyramid/SVD/Hybrid) + scheduling"
  (
    cd "$CUDA_DIR"
    make clean >/dev/null 2>&1 || true
    case "$mode" in
      debug) make debug ;;
      profile) make profile ;;
      compression|release|build|"") make release ;;
      *) err "Unknown build mode: $mode"; exit 1 ;;
    esac
  )
  if [[ ! -f "$CUDA_DIR/libpikv_kernels.so" ]]; then
    err "libpikv_kernels.so missing after build"
    exit 1
  fi
  ok "Built $(ls -lh "$CUDA_DIR/libpikv_kernels.so" | awk '{print $5, $9}')"
  # Symbol smoke-check for full compression exports
  if command_exists nm; then
    local missing=0
    for sym in launch_lora_compression_kernel launch_quantization_compression_kernel \
               launch_pyramid_compression_kernel launch_svd_compression_kernel \
               launch_hybrid_compression_kernel moe_routing_cuda; do
      if ! nm -D "$CUDA_DIR/libpikv_kernels.so" 2>/dev/null | grep -q "$sym"; then
        warn "symbol not found: $sym"
        missing=1
      fi
    done
    [[ "$missing" -eq 0 ]] && ok "Full compression + routing symbols present"
  fi
}

run_tests() {
  info "Running C++ kernel tests..."
  (
    cd "$CUDA_DIR"
    if [[ -x ./test_pikv_kernels ]]; then
      ./test_pikv_kernels
    else
      make test
    fi
  )
  ok "C++ tests finished"
}

run_py_tests() {
  info "Running Python CUDA tests (optional CuPy path)..."
  (
    cd "$ROOT"
    PYTHONPATH="$ROOT" python -m core.cuda.test_kernels 2>/dev/null \
      || PYTHONPATH="$ROOT" python "$CUDA_DIR/test_kernels.py" \
      || warn "Python tests skipped (GPU/CuPy may be unavailable)"
  )
}

install_lib() {
  local prefix="${1:-/usr/local}"
  info "Installing to $prefix"
  (
    cd "$CUDA_DIR"
    if [[ "$prefix" == /usr* ]] && [[ "$(id -u)" -ne 0 ]]; then
      sudo make install PREFIX="$prefix"
    else
      make install PREFIX="$prefix"
    fi
  )
  ok "Installed libpikv_kernels.so → $prefix/lib"
}

show_help() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

main() {
  if [[ ! -d "$CUDA_DIR" ]]; then
    err "core/cuda not found (ROOT=$ROOT)"
    exit 1
  fi
  local cmd="${1:-build}"
  case "$cmd" in
    help|-h|--help) show_help; exit 0 ;;
    clean)
      (cd "$CUDA_DIR" && make clean)
      ok "Cleaned"
      ;;
    check)
      check_cuda
      (cd "$CUDA_DIR" && make check-cuda)
      ;;
    build|release|compression|debug|profile)
      check_cuda
      build_mode "$cmd"
      ;;
    test)
      check_cuda
      build_mode release
      run_tests
      ;;
    test-py)
      check_cuda
      [[ -f "$CUDA_DIR/libpikv_kernels.so" ]] || build_mode release
      run_py_tests
      ;;
    install)
      check_cuda
      build_mode release
      install_lib "${PREFIX:-/usr/local}"
      ;;
    install-user)
      check_cuda
      build_mode release
      install_lib "${HOME}/.local"
      ;;
    *)
      err "Unknown option: $cmd"
      show_help
      exit 1
      ;;
  esac
  ok "build_cuda.sh done"
}

main "$@"
