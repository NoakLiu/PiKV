#!/usr/bin/env python3
"""Scheduling component experiment — measures eviction latency & retained size.

For the full factorial, run:
  python -m downstream_tasks.eval.ablation_study --preset scheduling
Protocol: downstream_tasks/EXPERIMENTAL_PROTOCOL.md
"""

from __future__ import annotations

import time

import torch

from core.single.module.pikv_scheduling import H2OScheduler, QUESTScheduler, StreamingLLMScheduler


def _bench(scheduler, keys, values, metadata, reps: int = 20):
    for _ in range(3):
        scheduler.select_eviction_candidates(keys, values, metadata)
    t0 = time.perf_counter()
    last = None
    for _ in range(reps):
        last = scheduler.select_eviction_candidates(keys, values, metadata)
    ms = (time.perf_counter() - t0) * 1000.0 / reps
    return ms, last


def main():
    # Protocol: cache holds a 4k-context budget proxy
    cache_size, hidden, tokens = 1024, 512, 512
    keys = torch.randn(tokens, hidden)
    values = torch.randn(tokens, hidden)
    metadata = {"importance": torch.rand(tokens)}

    schedulers = {
        "H2OScheduler": H2OScheduler(cache_size=cache_size, hidden_size=hidden),
        "StreamingLLMScheduler": StreamingLLMScheduler(cache_size=cache_size, hidden_size=hidden),
        "QUESTScheduler": QUESTScheduler(cache_size=cache_size, hidden_size=hidden),
    }
    print("Scheduling experiment (isolated)")
    print(f"tokens={tokens} cache_size={cache_size} hidden={hidden}")
    for name, sched in schedulers.items():
        ms, cand = _bench(sched, keys, values, metadata)
        shape = getattr(cand, "shape", type(cand))
        print(f"  {name:24s}  {ms:7.3f} ms/select  candidates={shape}")


if __name__ == "__main__":
    main()
