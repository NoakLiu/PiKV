#!/usr/bin/env python3
"""
PiKV-FPGA offload example (paper §3.5).

Runs the metadata pipeline on FPGA simulation: routing → compression → scheduling.
GPU would consume packed {(K̂, V̂, idx)} returned per token.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.fpga import create_pikv_fpga, estimate_bram_budget, is_fpga_available
from core.fpga.config import FPGAConfig, FPGAEngineMapping, FPGACompressionEngine, FPGARoutingEngine, FPGASchedulingEngine


def main():
    print("PiKV-FPGA offload example")
    print(f"FPGA available (sim or device): {is_fpga_available()}")

    cfg = FPGAConfig(
        num_experts=64,
        hidden_size=128,
        top_k=4,
        pages_per_gpu=16,
        compression_ratio=4.0,
        engines=FPGAEngineMapping(
            routing=FPGARoutingEngine.CACHE_AWARE,
            compression=FPGACompressionEngine.LORA,
            scheduling=FPGASchedulingEngine.ADAKV,
        ),
    )
    bram = estimate_bram_budget(cfg)
    print(f"BRAM budget: {bram['total_kb']:.1f} KB (page table {bram['bram_page_table_kb']:.1f} KB)")

    fpga = create_pikv_fpga(
        num_experts=cfg.num_experts,
        hidden_size=cfg.hidden_size,
        top_k=cfg.top_k,
        compression_ratio=cfg.compression_ratio,
    )

    d = cfg.hidden_size
    for t in range(8):
        q = torch.randn(d)
        k = torch.randn(d)
        v = torch.randn(d)
        packed, experts = fpga.process_token(q, k, v, token_id=t, attention_score=1.0 / (t + 1))
        print(f"  t={t} experts={experts} packed_shape={tuple(packed.shape)}")

    fpga.update_scheduler_theta(target_hit_rate=0.85)
    stats = fpga.get_stats()
    print("\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
