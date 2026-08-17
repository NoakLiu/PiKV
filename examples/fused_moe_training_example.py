#!/usr/bin/env python3
"""Demo: fused LAER-MoE + MoEBlaze + FSMoE training systems."""

import torch

from core.distributed import (
    create_fsmoe,
    create_fused_moe_training,
    create_laer_moe,
    create_moeblaze,
)


def run_fused(steps: int = 40):
    print("=== Fused MoE Training (LAER + MoEBlaze + FSMoE) ===")
    model = create_fused_moe_training(
        hidden_size=128,
        num_experts=8,
        top_k=2,
        num_layers=2,
        enable_laer=True,
        enable_moeblaze=True,
        enable_fsmoe=True,
        relayout_interval=8,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        # Skewed tokens to stress LAER re-layout / FSMoE staging
        x = torch.randn(4, 32, 128)
        # Bias first dims so gate sees non-uniform inputs over time
        x[..., :16] = x[..., :16] + (1.5 if step % 3 == 0 else -0.5)
        metrics = model.train_step(x, optimizer=opt)
        if step % 10 == 0 or step == steps - 1:
            layer0 = metrics["layers"][0]
            print(
                f"step={step:03d} loss={metrics['loss']:.4f} "
                f"laer_imb={layer0.get('laer_relayout', {}).get('load_imbalance', 0):.3f} "
                f"relayouts={layer0.get('laer_relayout', {}).get('relayout_events', 0)} "
                f"w_hit={layer0.get('moeblaze_weight', {}).get('hit_rate', 0):.2f} "
                f"fsmoe_adapt={layer0.get('fsmoe', {}).get('adaptations', 0)}"
            )
    print("fused OK\n")


def run_components():
    print("=== Standalone components ===")
    x = torch.randn(2, 16, 64)
    laer = create_laer_moe(hidden_size=64, num_experts=4, top_k=2, relayout_interval=4)
    blaze = create_moeblaze(hidden_size=64, num_experts=4, top_k=2)
    fs = create_fsmoe(hidden_size=64, num_experts=4, top_k=2)
    for i in range(12):
        _ = laer(x)
        _ = blaze(x)
        _ = fs(x)
    print("LAER:", laer.get_laer_metrics()["relayout"])
    print("MoEBlaze:", blaze.get_moeblaze_metrics())
    print("FSMoE:", fs.get_fsmoe_metrics())
    print("components OK")


if __name__ == "__main__":
    run_fused()
    run_components()
