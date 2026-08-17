#!/usr/bin/env bash
# Install PiKV conda/venv + dependencies (updated 2026).
#
# Usage:
#   ./scripts/install_pikv.sh
#   ENV_NAME=pikv311 ./scripts/install_pikv.sh
#   USE_VENV=1 ./scripts/install_pikv.sh
#   SKIP_TORCH=1 ./scripts/install_pikv.sh
#   WITH_VLLM=1 WITH_DEEPSPEED=1 ./scripts/install_pikv.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-pikv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_CUDA="${PYTORCH_CUDA:-12.4}"
USE_VENV="${USE_VENV:-0}"
SKIP_TORCH="${SKIP_TORCH:-0}"
WITH_VLLM="${WITH_VLLM:-0}"
WITH_DEEPSPEED="${WITH_DEEPSPEED:-0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"

echo "==> PiKV install"
echo "    root=$ROOT  env=$ENV_NAME  python=$PYTHON_VERSION  cuda=$PYTORCH_CUDA"

activate_env() {
  if [[ "$USE_VENV" == "1" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
  else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
  fi
}

if [[ "$USE_VENV" == "1" ]]; then
  echo "==> Creating venv at .venv (python$PYTHON_VERSION)..."
  if [[ ! -d "$ROOT/.venv" ]]; then
    python"$PYTHON_VERSION" -m venv "$ROOT/.venv" 2>/dev/null \
      || python3 -m venv "$ROOT/.venv"
  fi
  activate_env
  python -m pip install -U pip setuptools wheel
  if [[ "$SKIP_TORCH" != "1" ]]; then
    echo "==> Installing PyTorch (pip CUDA index; override if CPU-only)..."
    pip install torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/cu124" \
      || pip install torch torchvision torchaudio
  fi
else
  if ! command -v conda &>/dev/null; then
    echo "Conda not found. Install Miniconda/Anaconda, or re-run with USE_VENV=1" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Conda env '$ENV_NAME' exists — updating..."
  else
    echo "==> Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" pip -y
  fi
  conda activate "$ENV_NAME"
  python -m pip install -U pip setuptools wheel

  if [[ "$SKIP_TORCH" != "1" ]]; then
    echo "==> Installing PyTorch via conda (pytorch-cuda=$PYTORCH_CUDA)..."
    conda install -y pytorch torchvision torchaudio "pytorch-cuda=$PYTORCH_CUDA" \
      -c pytorch -c nvidia \
      || {
        echo "Conda CUDA install failed; falling back to pip cu124 wheels..."
        pip install torch torchvision torchaudio \
          --index-url "https://download.pytorch.org/whl/cu124"
      }
  fi
fi

echo "==> Installing Python deps..."
SETUP_DIR="$ROOT/setup"
if [[ "$SKIP_TORCH" != "1" ]] && [[ -f "$SETUP_DIR/requirements-conda.txt" ]]; then
  pip install -r "$SETUP_DIR/requirements-conda.txt"
else
  pip install -r "$SETUP_DIR/requirements.txt"
fi

echo "==> Installing PiKV (editable)..."
pip install -e "$ROOT"

if [[ "$WITH_VLLM" == "1" ]]; then
  echo "==> Installing vLLM (optional)..."
  pip install "vllm>=0.5.0"
fi

if [[ "$WITH_DEEPSPEED" == "1" ]]; then
  echo "==> Installing DeepSpeed (optional)..."
  pip install deepspeed
fi

echo "==> Verifying import..."
python - <<'PY'
import torch
import transformers
import datasets
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}  datasets={datasets.__version__}")
import data
print(f"data package OK  default_dir={data.default_data_dir()}")
PY

if [[ "$DOWNLOAD_DATA" == "1" ]]; then
  echo "==> Downloading eval corpus into data/..."
  python -m data.download_data --max-prompts 256 || true
fi

echo ""
echo "Installation completed."
if [[ "$USE_VENV" == "1" ]]; then
  echo "  Activate:  source .venv/bin/activate"
else
  echo "  Activate:  conda activate $ENV_NAME"
fi
echo "  Setup:     setup/  (requirements, environment.yml, setup.py)"
echo "  CUDA:      ./scripts/build_cuda.sh release"
echo "  Data:      python -m data.download_data"
echo "  Eval:      python -m downstream_tasks.eval.eval_with_data --max-prompts 32"
echo "  Ablation:  python -m downstream_tasks.eval.ablation_study --preset factor --from-data"
