#!/usr/bin/env python3
"""Compression component experiment — measures ratio & latency.

For the full factorial, run:
  python -m downstream_tasks.eval.ablation_study --preset compression
Protocol: downstream_tasks/EXPERIMENTAL_PROTOCOL.md
"""

from __future__ import annotations

import time

import torch

from core.single.module.pikv_compression import LoRACompressor, PyramidCompressor, SVDCompressor


def _bench(comp, keys, values, reps: int = 20):
    for _ in range(3):
        comp(keys, values)
    t0 = time.perf_counter()
    last = None
    for _ in range(reps):
        last = comp(keys, values)
    ms = (time.perf_counter() - t0) * 1000.0 / reps
    return ms, last


def main():
    seq, hidden = 512, 512
    keys = torch.randn(1, seq, hidden)
    values = torch.randn(1, seq, hidden)
    raw = keys.nelement() * keys.element_size() + values.nelement() * values.element_size()

    compressors = {
        "PyramidCompressor": PyramidCompressor(hidden_size=hidden),
        "LoRACompressor": LoRACompressor(hidden_size=hidden),
        "SVDCompressor": SVDCompressor(hidden_size=hidden),
    }
    print("Compression experiment (isolated)")
    print(f"input keys/values shape={(1, seq, hidden)} raw_bytes={raw}")
    for name, comp in compressors.items():
        ms, out = _bench(comp, keys, values)
        ck, cv = out[0], out[1]
        stored = ck.nelement() * ck.element_size() + cv.nelement() * cv.element_size()
        ratio = raw / max(stored, 1)
        print(
            f"  {name:20s}  {ms:7.3f} ms  "
            f"out={tuple(ck.shape)}  ratio={ratio:.2f}x  stored_MB={stored/1e6:.3f}"
        )


if __name__ == "__main__":
    main()
