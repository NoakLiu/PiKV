#!/usr/bin/env python3
"""
Evaluation test using the shared PiKV dataloader.

Measures next-token CE loss and token accuracy on frozen prompts from
``data/prompts_eval.txt`` (via ``python -m data.download_data``).

Examples:
  python -m data.download_data
  python -m downstream_tasks.eval.eval_with_data --max-prompts 32
  python -m downstream_tasks.eval.eval_with_data --model pikv --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_model(name: str, vocab_size: int, hidden_size: int, device: torch.device):
    if name == "linear":
        # Tiny reference: embed → linear → vocab
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.out = nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids):
                return self.out(self.embed(input_ids))

        return TinyLM().to(device)

    if name == "pikv":
        from core.single.moe import create_moe

        class PiKVWrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.moe = create_moe(
                    "base",
                    hidden_size=hidden_size,
                    num_experts=4,
                    top_k=2,
                    use_normalization=True,
                )
                self.out = nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                y, _ = self.moe(x)
                return self.out(y)

        return PiKVWrap().to(device)

    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    correct = 0
    t0 = time.perf_counter()
    for batch in loader:
        if len(batch) == 3:
            input_ids, target_ids, _text = batch
        else:
            input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction="sum",
        )
        pred = logits.argmax(dim=-1)
        correct += int((pred == target_ids).sum().item())
        n = target_ids.numel()
        total_loss += float(loss.item())
        total_tok += n
    elapsed = time.perf_counter() - t0
    return {
        "ce_loss": total_loss / max(total_tok, 1),
        "token_accuracy": correct / max(total_tok, 1),
        "num_tokens": total_tok,
        "wall_s": elapsed,
        "tokens_per_s": total_tok / max(elapsed, 1e-9),
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Eval test on downloaded PiKV data")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", default="linear", choices=["linear", "pikv"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )

    from transformers import AutoTokenizer

    from data.dataloader import create_eval_dataloader, default_data_dir

    data_dir = args.data_dir or default_data_dir()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        loader = create_eval_dataloader(
            data_dir=data_dir,
            tokenizer=tokenizer,
            prefill_tokens=args.prefill_tokens,
            batch_size=args.batch_size,
            max_prompts=args.max_prompts,
        )
    except FileNotFoundError as e:
        print(e)
        print("Hint: python -m data.download_data")
        return 1

    model = _build_model(args.model, tokenizer.vocab_size, args.hidden_size, device)
    metrics = evaluate(model, loader, device)

    report: Dict[str, Any] = {
        "model": args.model,
        "device": str(device),
        "data_dir": os.path.abspath(data_dir),
        "prefill_tokens": args.prefill_tokens,
        "max_prompts": args.max_prompts,
        "hidden_size": args.hidden_size,
        "metrics": metrics,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "EXPERIMENTAL_PROTOCOL.md + data/README.md",
    }

    print("PiKV eval_with_data")
    print("=" * 50)
    print(f"data_dir={report['data_dir']}")
    print(f"model={args.model} device={device}")
    print(
        f"CE={metrics['ce_loss']:.4f}  acc={metrics['token_accuracy']*100:.2f}%  "
        f"tok/s={metrics['tokens_per_s']:.1f}"
    )

    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"eval_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
