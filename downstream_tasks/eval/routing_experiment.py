#!/usr/bin/env python3
"""Routing component experiment — measures isolated router cost & load balance.

For the full factorial (routing/compression/scheduling/sharding), run:
  python -m downstream_tasks.eval.ablation_study --preset routing
Protocol: downstream_tasks/EXPERIMENTAL_PROTOCOL.md
"""

from __future__ import annotations

import time

import torch

from core.single.module.pikv_routing import AdaptiveRouter, BaseRouter, TopKBalancedRouter


def _bench(router, x, reps: int = 20):
    # warmup
    for _ in range(3):
        router(x)
    t0 = time.perf_counter()
    last = None
    for _ in range(reps):
        last = router(x)
    ms = (time.perf_counter() - t0) * 1000.0 / reps
    return ms, last


def main():
    hidden, experts, batch, seq = 512, 8, 1, 512
    x = torch.randn(batch, seq, hidden)
    routers = {
        "BaseRouter": BaseRouter(hidden_size=hidden, num_experts=experts),
        "TopKBalancedRouter": TopKBalancedRouter(hidden_size=hidden, num_experts=experts),
        "AdaptiveRouter": AdaptiveRouter(hidden_size=hidden, num_experts=experts),
    }
    print("Routing experiment (isolated)")
    print(f"shape=({batch},{seq},{hidden}) experts={experts}")
    for name, router in routers.items():
        ms, out = _bench(router, x)
        # outputs vary by router; print first tensor shape if tuple
        shape = out[0].shape if isinstance(out, tuple) else getattr(out, "shape", None)
        print(f"  {name:22s}  {ms:7.3f} ms/fwd  out0={shape}")


if __name__ == "__main__":
    main()
