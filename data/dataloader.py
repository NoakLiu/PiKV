#!/usr/bin/env python3
"""
Shared PiKV text dataloaders (repo-root ``data/``).

Consumes files produced by ``python -m data.download_data``:
  - train.txt / test.txt  → sliding-window next-token Dataset
  - prompts_eval.txt      → frozen eval prompts (protocol fairness)

Used by:
  - downstream_tasks/llm/next_tok_pred/* training scripts
  - downstream_tasks/eval/ablation_study.py (--from-data)
  - downstream_tasks/eval/eval_with_data.py
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def default_data_dir() -> str:
    """Canonical corpus directory: ``<repo>/data``."""
    return os.path.abspath(os.path.dirname(__file__))


def resolve_data_path(*names: str, data_dir: Optional[str] = None) -> str:
    """Return first existing path among candidates under data_dir / CWD."""
    root = data_dir or default_data_dir()
    candidates = []
    for name in names:
        candidates.append(os.path.join(root, name))
        candidates.append(name)
        candidates.append(os.path.join("data", name))
        candidates.append(
            os.path.join(
                os.path.dirname(root),
                "downstream_tasks",
                "llm",
                "next_tok_pred",
                "data",
                name,
            )
        )
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Missing data file. Looked for: "
        + ", ".join(candidates)
        + "\nRun: python -m data.download_data"
    )


class TextWindowDataset(Dataset):
    """Next-token prediction windows from a plain-text corpus."""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 512,
        stride: Optional[int] = None,
        max_sequences: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(text)
        else:
            tokens = list(tokenizer(text)["input_ids"])

        self.sequences: List[Tuple[List[int], List[int]]] = []
        for i in range(0, max(0, len(tokens) - max_length), self.stride):
            sequence = tokens[i : i + max_length]
            if len(sequence) < max_length:
                break
            self.sequences.append((sequence[:-1], sequence[1:]))
            if max_sequences is not None and len(self.sequences) >= max_sequences:
                break

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        inp, tgt = self.sequences[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


class PromptEvalDataset(Dataset):
    """Frozen eval prompts (one per line) from prompts_eval.txt."""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        prefill_tokens: int = 512,
        max_prompts: Optional[int] = None,
        pad_to_length: bool = True,
    ):
        self.tokenizer = tokenizer
        self.prefill_tokens = prefill_tokens
        self.pad_to_length = pad_to_length
        self.prompts: List[str] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.prompts.append(line)
                if max_prompts is not None and len(self.prompts) >= max_prompts:
                    break
        if not self.prompts:
            raise ValueError(f"No prompts in {file_path}")

        eos = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "pad_token_id", 0) or 0
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos
        self.pad_id = int(pad_id)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int):
        text = self.prompts[idx]
        ids = self.tokenizer.encode(text)
        ids = ids[: self.prefill_tokens]
        if self.pad_to_length and len(ids) < self.prefill_tokens:
            ids = ids + [self.pad_id] * (self.prefill_tokens - len(ids))
        input_ids = torch.tensor(ids, dtype=torch.long)
        target = torch.cat([input_ids[1:], input_ids[-1:]])
        return input_ids, target, text


def create_train_dataloader(
    data_dir: Optional[str] = None,
    tokenizer=None,
    max_length: int = 512,
    batch_size: int = 4,
    shuffle: bool = True,
    split: str = "train",
    max_sequences: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    fname = "train.txt" if split == "train" else "test.txt"
    path = resolve_data_path(fname, data_dir=data_dir)
    ds = TextWindowDataset(
        path, tokenizer, max_length=max_length, max_sequences=max_sequences
    )
    if len(ds) == 0:
        raise RuntimeError(
            f"No sequences in {path}. Re-run: python -m data.download_data"
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def create_eval_dataloader(
    data_dir: Optional[str] = None,
    tokenizer=None,
    prefill_tokens: int = 512,
    batch_size: int = 1,
    max_prompts: Optional[int] = 64,
    num_workers: int = 0,
) -> DataLoader:
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    path = resolve_data_path("prompts_eval.txt", "test.txt", data_dir=data_dir)
    ds = PromptEvalDataset(
        path, tokenizer, prefill_tokens=prefill_tokens, max_prompts=max_prompts
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def iter_eval_prompts(
    data_dir: Optional[str] = None, max_prompts: Optional[int] = None
) -> Iterator[str]:
    try:
        path = resolve_data_path("prompts_eval.txt", data_dir=data_dir)
    except FileNotFoundError:
        path = resolve_data_path("test.txt", data_dir=data_dir)
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line
            n += 1
            if max_prompts is not None and n >= max_prompts:
                break


def prompts_to_hidden(
    prompts: Sequence[str],
    hidden_size: int,
    seq_len: int,
    tokenizer=None,
    device: str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    """Map text prompts → [B, seq_len, H] with a seeded Embedding (fair ablations)."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    embed = torch.nn.Embedding(tokenizer.vocab_size, hidden_size)
    torch.nn.init.normal_(embed.weight, generator=g)
    embed.eval()

    batch = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    for text in prompts:
        ids = tokenizer.encode(text)[:seq_len]
        if len(ids) < seq_len:
            ids = ids + [pad_id] * (seq_len - len(ids))
        batch.append(torch.tensor(ids, dtype=torch.long))
    input_ids = torch.stack(batch, dim=0)
    with torch.no_grad():
        hidden = embed(input_ids)
    return hidden.to(device)
