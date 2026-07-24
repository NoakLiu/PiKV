#!/usr/bin/env python3
"""
PiKV evaluation data download script (repo-root ``data/``).

Downloads public corpora used by next-token-prediction training/eval and by the
experimental protocol (frozen prompts for fair module comparisons).

Default corpus: WikiText-2 (wikitext-2-raw-v1) via HuggingFace ``datasets``.

Outputs (under ``data/`` by default):
  train.txt          full training text
  test.txt           full test / validation text
  prompts_eval.txt   one prompt per line (prefill-budget friendly)
  manifest.json      dataset id, sizes, seed, protocol knobs

Examples:
  python -m data.download_data
  python -m data.download_data --max-prompts 256 --prompt-chars 800
  python -m data.download_data --dataset wikitext --out-dir data
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _repo_root() -> str:
    # data/download_data.py → repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_out_dir() -> str:
    return os.path.join(_repo_root(), "data")


def _ensure_deps():
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: datasets. Install with:\n"
            "  pip install datasets transformers\n"
            f"Original error: {e}"
        )


def load_corpus(dataset: str, config: Optional[str], split_train: str, split_test: str):
    from datasets import load_dataset

    if dataset == "wikitext":
        config = config or "wikitext-2-raw-v1"
        ds = load_dataset("wikitext", config)
        train_key = "train"
        test_key = "test" if "test" in ds else "validation"
        return ds, train_key, test_key, f"wikitext/{config}"

    if dataset == "openwebtext-sample":
        ds = load_dataset("Skylion007/openwebtext", split="train[:1%]")

        class _Wrap:
            def __getitem__(self, k):
                if k == "train":
                    return ds
                raise KeyError(k)

            def __contains__(self, k):
                return k == "train"

        return _Wrap(), "train", "train", "openwebtext[:1%]"

    if config:
        ds = load_dataset(dataset, config)
    else:
        ds = load_dataset(dataset)
    train_key = split_train if split_train in ds else list(ds.keys())[0]
    test_key = split_test if split_test in ds else (
        "test" if "test" in ds else ("validation" if "validation" in ds else train_key)
    )
    label = f"{dataset}/{config}" if config else dataset
    return ds, train_key, test_key, label


def _join_text_column(split_obj, text_field: str = "text") -> str:
    lines: List[str] = []
    if hasattr(split_obj, "column_names") and text_field in split_obj.column_names:
        for text in split_obj[text_field]:
            if text and str(text).strip():
                lines.append(str(text).strip())
    else:
        for row in split_obj:
            text = row.get(text_field, "") if isinstance(row, dict) else ""
            if text and str(text).strip():
                lines.append(str(text).strip())
    return "\n".join(lines) + ("\n" if lines else "")


def build_eval_prompts(text: str, max_prompts: int, prompt_chars: int, seed: int) -> List[str]:
    """Slice non-overlapping windows for frozen evaluation prompts."""
    import random

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 40]
    rng = random.Random(seed)
    rng.shuffle(paragraphs)
    prompts: List[str] = []
    for p in paragraphs:
        chunk = p[:prompt_chars].strip()
        if len(chunk) < 40:
            continue
        prompts.append(chunk)
        if len(prompts) >= max_prompts:
            break
    i = 0
    while len(prompts) < max_prompts and i + prompt_chars <= len(text):
        chunk = text[i : i + prompt_chars].strip()
        if len(chunk) >= 40:
            prompts.append(chunk)
        i += prompt_chars
    return prompts[:max_prompts]


def download(
    dataset: str = "wikitext",
    config: Optional[str] = None,
    out_dir: Optional[str] = None,
    max_prompts: int = 256,
    prompt_chars: int = 800,
    seed: int = 42,
    split_train: str = "train",
    split_test: str = "test",
    tokenize_stats: bool = True,
) -> Dict[str, Any]:
    _ensure_deps()
    out_dir = out_dir or _default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    ds, train_key, test_key, label = load_corpus(dataset, config, split_train, split_test)
    print(f"Loading corpus: {label} (train={train_key}, test={test_key})")
    train_text = _join_text_column(ds[train_key])
    test_text = _join_text_column(ds[test_key])

    train_path = os.path.join(out_dir, "train.txt")
    test_path = os.path.join(out_dir, "test.txt")
    prompts_path = os.path.join(out_dir, "prompts_eval.txt")
    manifest_path = os.path.join(out_dir, "manifest.json")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_text)

    prompts = build_eval_prompts(test_text or train_text, max_prompts, prompt_chars, seed)
    with open(prompts_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p.replace("\n", " ").strip() + "\n")

    stats: Dict[str, Any] = {
        "dataset": label,
        "out_dir": os.path.abspath(out_dir),
        "train_chars": len(train_text),
        "test_chars": len(test_text),
        "num_eval_prompts": len(prompts),
        "prompt_chars": prompt_chars,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "train": train_path,
            "test": test_path,
            "prompts_eval": prompts_path,
        },
        "protocol": {
            "prefill_tokens": 512,
            "decode_tokens": 128,
            "context": 4096,
            "note": "Use data/prompts_eval.txt as frozen prompts in EXPERIMENTAL_PROTOCOL.md",
        },
    }

    if tokenize_stats:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("gpt2")
            if len(train_text) < 500_000:
                stats["train_tokens"] = len(tok.encode(train_text))
            else:
                sample = train_text[:200_000]
                stats["train_tokens_approx"] = len(tok.encode(sample)) * max(
                    1, len(train_text) // max(len(sample), 1)
                )
            if len(test_text) < 500_000:
                stats["test_tokens"] = len(tok.encode(test_text))
        except Exception as e:
            stats["tokenize_warning"] = str(e)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Data download complete.")
    print(f"  train:   {train_path} ({stats['train_chars']} chars)")
    print(f"  test:    {test_path} ({stats['test_chars']} chars)")
    print(f"  prompts: {prompts_path} ({len(prompts)} lines)")
    print(f"  manifest:{manifest_path}")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download PiKV evaluation / training corpora into data/")
    parser.add_argument("--dataset", default="wikitext", help="wikitext | openwebtext-sample | HF name")
    parser.add_argument("--config", default=None, help="HF dataset config (default: wikitext-2-raw-v1)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <repo>/data)")
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--prompt-chars", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-train", default="train")
    parser.add_argument("--split-test", default="test")
    parser.add_argument("--no-tokenize-stats", action="store_true")
    args = parser.parse_args(argv)

    download(
        dataset=args.dataset,
        config=args.config,
        out_dir=args.out_dir,
        max_prompts=args.max_prompts,
        prompt_chars=args.prompt_chars,
        seed=args.seed,
        split_train=args.split_train,
        split_test=args.split_test,
        tokenize_stats=not args.no_tokenize_stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
