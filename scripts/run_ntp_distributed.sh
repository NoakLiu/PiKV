#!/usr/bin/env bash
# Multi-GPU NTP distributed training (torchrun)
# Usage: ./scripts/run_ntp_distributed.sh [epochs] [save_every] [model_type]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NTP_DIR="$ROOT/downstream_tasks/llm/next_tok_pred"
cd "$NTP_DIR"

NGPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
NGPUS="${NGPUS:-1}"
echo "Detected $NGPUS GPUs"

if [[ "$NGPUS" -lt 2 ]]; then
  echo "Warning: Only $NGPUS GPU(s) detected. Distributed training works best with 2+ GPUs."
fi

EPOCHS="${1:-10}"
SAVE_EVERY="${2:-5}"
MODEL_TYPE="${3:-pikv}"

echo "Starting distributed training:"
echo "  GPUs=$NGPUS  epochs=$EPOCHS  save_every=$SAVE_EVERY  model=$MODEL_TYPE"

mkdir -p "$ROOT/data"
if [[ ! -f "$ROOT/data/train.txt" ]]; then
  echo "No data/train.txt — run: python -m data.download_data"
  echo "Creating a tiny placeholder corpus..."
  cat > "$ROOT/data/train.txt" <<'EOF'
The quick brown fox jumps over the lazy dog.
Machine learning combines statistics and computer science.
Natural language processing enables computers to understand language.
EOF
fi

# Prefer package data path via symlink/copy into local expectations if needed
ln -sfn "$ROOT/data" ./data 2>/dev/null || true

echo "Launching torchrun..."
torchrun \
  --nproc_per_node="$NGPUS" \
  --master_port=29500 \
  train_distributed.py \
  --epochs "$EPOCHS" \
  --save_every "$SAVE_EVERY" \
  --model_type "$MODEL_TYPE"

echo "Distributed training completed."
[[ -d checkpoints ]] && ls -la checkpoints/ || echo "No checkpoints found."
