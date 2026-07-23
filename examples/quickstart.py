#!/usr/bin/env python3
"""
PiKV Quickstart Demo

Live entry point referenced in the paper (§3.2).
Runs a short smoke demo of EPiKV-MoE, unified MoE factory, and optional vLLM setup.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch


def demo_epikv_moe():
    print("=" * 60)
    print("EPiKV-MoE quickstart")
    print("=" * 60)

    from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

    model = create_enhanced_pikv_moe(
        enable_dynamic_balancing=True,
        enable_async_execution=True,
        enable_communication_optimization=True,
        enable_smartmoe=False,
        world_size=1,
    )

    x = torch.randn(2, 16, 512)
    query = torch.randn(2, 16, 512)
    with torch.no_grad():
        logits = model(x, query=query)

    metrics = model.get_performance_metrics()
    print(f"Output shape: {tuple(logits.shape)}")
    if "load_balancing" in metrics:
        print(f"Load imbalance: {metrics['load_balancing']['load_imbalance']:.4f}")
    print("EPiKV-MoE OK\n")
    return model


def demo_moe_factory():
    print("=" * 60)
    print("Unified MoE factory quickstart")
    print("=" * 60)

    from core.single.moe import create_moe

    model = create_moe("base", hidden_size=256, num_experts=4, top_k=2)
    x = torch.randn(2, 16, 256)
    output, aux_loss = model(x)
    print(f"Output shape: {tuple(output.shape)}, aux_loss: {aux_loss:.4f}")
    print("MoE factory OK\n")
    return model


def demo_vllm_config():
    print("=" * 60)
    print("vLLM integration config quickstart")
    print("=" * 60)

    from core.single.vllm_integration import create_pikv_vllm

    engine = create_pikv_vllm(
        model_name="microsoft/DialoGPT-medium",
        enable_compression=True,
        enable_scheduling=True,
        enable_kvcache_centric=True,
    )
    stats = engine.get_performance_stats()
    print(f"Engine ready. Stats keys: {sorted(stats.keys())}")
    print("vLLM integration OK\n")
    return engine


def main():
    print("PiKV Quickstart\n")
    demo_epikv_moe()
    demo_moe_factory()
    demo_vllm_config()
    print("All quickstart demos completed successfully.")


if __name__ == "__main__":
    main()
