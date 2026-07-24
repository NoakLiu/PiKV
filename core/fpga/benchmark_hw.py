"""
PiKV-FPGA / CXL hardware characterization (paper §3.5).

Reports platform specs, modeled bandwidth, on-chip latency, and
end-to-end metadata-path gains vs a CPU baseline and a GPU-only baseline.

This is the artifact companion for hardware claims: it makes assumptions
explicit and prints comparable numbers. On real U55C silicon, replace the
modeled CXL bandwidth with `xbutil` / PCIe/CXL counters.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch

from core.fpga import create_pikv_fpga, estimate_bram_budget, is_fpga_available
from core.fpga.config import (
    FPGACompressionEngine,
    FPGAConfig,
    FPGAEngineMapping,
    FPGARoutingEngine,
    FPGASchedulingEngine,
    estimate_fpga_latency_breakdown,
    estimate_fpga_latency_us,
)


@dataclass(frozen=True)
class HardwarePlatformSpec:
    """Canonical platform used for PiKV-FPGA claims (Alveo U55C path)."""

    board: str = "AMD Alveo U55C"
    fpga_part: str = "xcu55c-fsvh2892-2L-e"
    logic_cells_m: float = 1.3  # ≈1.3M system logic cells (datasheet)
    hbm2_capacity_gb: float = 16.0
    hbm2_peak_bw_gbps: float = 460.0  # HBM2 peak (on-card)
    user_clock_mhz: float = 300.0
    host_link: str = "PCIe Gen4 x16 (XDMA) + optional CXL Type-3.mem"
    cxl_modeled_bw_gbps: float = 64.0  # conservative effective CXL.mem payload BW
    pcie_modeled_bw_gbps: float = 32.0  # effective host↔FPGA MMIO/DMA payload BW
    notes: str = (
        "Bitstream target: vivado/constraints/pikv_u55c.xdc. "
        "Without CXL silicon, AXI-MM maps to on-card HBM/DDR via axi_noc for bring-up."
    )


def platform_spec_dict() -> Dict[str, Any]:
    return asdict(HardwarePlatformSpec())


def model_bandwidth_table(cfg: FPGAConfig) -> Dict[str, Any]:
    """
    Bandwidth / latency model used in §3.5 tables.

    All values are *effective payload* estimates, not raw link signaling rates.
    """
    spec = HardwarePlatformSpec()
    bytes_per_token = 2 * cfg.compressed_dim * 2  # K'+V', fp16
    tokens_per_s_cxl = (cfg.mem_bandwidth_gbps * 1e9 / 8.0) / max(bytes_per_token, 1)
    tokens_per_s_pcie = (spec.pcie_modeled_bw_gbps * 1e9 / 8.0) / max(bytes_per_token, 1)
    meta_us = estimate_fpga_latency_us(cfg, path="metadata")
    full_us = estimate_fpga_latency_us(cfg, path="full")
    breakdown = estimate_fpga_latency_breakdown(cfg)

    # Sweep effective CXL BW for sensitivity (comparative experiment)
    bw_sweep = {}
    for bw in (16.0, 32.0, 64.0, 128.0, spec.hbm2_peak_bw_gbps):
        cfg_bw = FPGAConfig(
            num_experts=cfg.num_experts,
            hidden_size=cfg.hidden_size,
            top_k=cfg.top_k,
            pages_per_gpu=cfg.pages_per_gpu,
            compression_ratio=cfg.compression_ratio,
            mem_bandwidth_gbps=bw,
            fpga_freq_ghz=cfg.fpga_freq_ghz,
            engines=cfg.engines,
        )
        bw_sweep[f"{bw:g}_GBps"] = {
            "metadata_us": estimate_fpga_latency_us(cfg_bw, path="metadata"),
            "full_us": estimate_fpga_latency_us(cfg_bw, path="full"),
        }

    return {
        "compressed_dim": cfg.compressed_dim,
        "bytes_per_token_kv_body": bytes_per_token,
        "cxl_effective_bw_gbps": cfg.mem_bandwidth_gbps,
        "pcie_effective_bw_gbps": spec.pcie_modeled_bw_gbps,
        "hbm2_peak_bw_gbps": spec.hbm2_peak_bw_gbps,
        "modeled_tokens_per_s_cxl": tokens_per_s_cxl,
        "modeled_tokens_per_s_pcie_meta": tokens_per_s_pcie,
        "modeled_fpga_metadata_latency_us": meta_us,
        "modeled_fpga_full_latency_us": full_us,
        "latency_breakdown_us": breakdown,
        "bandwidth_sensitivity": bw_sweep,
        "user_clock_mhz": cfg.fpga_freq_ghz * 1e3,
        "assumption": (
            "CXL BW defaults to FPGAConfig.mem_bandwidth_gbps "
            f"(default {HardwarePlatformSpec().cxl_modeled_bw_gbps} GB/s effective). "
            "Replace with measured counters on deployed hardware. "
            "HBM2 peak is datasheet context only, not claimed app throughput."
        ),
    }


def _cpu_baseline_metadata(
    cfg: FPGAConfig, num_tokens: int, reps: int = 5
) -> Dict[str, float]:
    """Pure-PyTorch metadata path on CPU (routing+compress+schedule proxy)."""
    d = cfg.hidden_size
    router = torch.nn.Linear(d, cfg.num_experts)
    times: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for t in range(num_tokens):
            q = torch.randn(d)
            logits = router(q)
            _ = torch.topk(logits, k=cfg.top_k)
            # cheap compress/schedule proxies
            k = torch.randn(d)
            v = torch.randn(d)
            _ = k[:: max(1, int(cfg.compression_ratio))], v[:: max(1, int(cfg.compression_ratio))]
        times.append((time.perf_counter() - t0) * 1e6 / num_tokens)
    return {
        "cpu_us_per_token_mean": float(sum(times) / len(times)),
        "cpu_us_per_token_std": float(
            (sum((x - sum(times) / len(times)) ** 2 for x in times) / len(times)) ** 0.5
        ),
    }


def _fpga_sim_metadata(
    cfg: FPGAConfig, num_tokens: int, reps: int = 5
) -> Dict[str, float]:
    fpga = create_pikv_fpga(
        num_experts=cfg.num_experts,
        hidden_size=cfg.hidden_size,
        top_k=cfg.top_k,
        compression_ratio=cfg.compression_ratio,
        simulate=True,
    )
    d = cfg.hidden_size
    times: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for t in range(num_tokens):
            q = torch.randn(d)
            k = torch.randn(d)
            v = torch.randn(d)
            fpga.process_token(q, k, v, token_id=t, attention_score=1.0 / (t + 1))
        times.append((time.perf_counter() - t0) * 1e6 / num_tokens)
    modeled = estimate_fpga_latency_us(cfg, path="metadata")
    modeled_full = estimate_fpga_latency_us(cfg, path="full")
    return {
        "fpga_sim_us_per_token_mean": float(sum(times) / len(times)),
        "fpga_sim_us_per_token_std": float(
            (sum((x - sum(times) / len(times)) ** 2 for x in times) / len(times)) ** 0.5
        ),
        "fpga_modeled_metadata_us_per_token": modeled,
        "fpga_modeled_full_us_per_token": modeled_full,
        "stats": fpga.get_stats(),
    }


def end_to_end_comparison(cfg: Optional[FPGAConfig] = None, num_tokens: int = 128) -> Dict[str, Any]:
    """
    Comparative experiment: CPU baseline vs FPGA metadata / full paths.

    Reports absolute latency and relative gain. Does *not* claim HBM peak BW
    as achieved application throughput.
    """
    cfg = cfg or FPGAConfig(
        engines=FPGAEngineMapping(
            routing=FPGARoutingEngine.CACHE_AWARE,
            compression=FPGACompressionEngine.LORA,
            scheduling=FPGASchedulingEngine.ADAKV,
        )
    )
    cpu = _cpu_baseline_metadata(cfg, num_tokens=num_tokens)
    fpga = _fpga_sim_metadata(cfg, num_tokens=num_tokens)
    cpu_mean = cpu["cpu_us_per_token_mean"]
    meta = fpga["fpga_modeled_metadata_us_per_token"]
    full = fpga["fpga_modeled_full_us_per_token"]
    return {
        "platform": platform_spec_dict(),
        "bandwidth_model": model_bandwidth_table(cfg),
        "bram_budget_kb": estimate_bram_budget(cfg),
        "cpu_baseline": cpu,
        "fpga_path": {k: v for k, v in fpga.items() if k != "stats"},
        "fpga_runtime_stats": fpga.get("stats", {}),
        "comparative_table": [
            {
                "system": "CPU metadata (PyTorch proxy)",
                "us_per_token": cpu_mean,
                "notes": "Host-side route+compress+schedule proxy",
            },
            {
                "system": "FPGA metadata (analytic)",
                "us_per_token": meta,
                "speedup_vs_cpu": (cpu_mean / meta) if meta > 0 else None,
                "notes": "PiKV-CTRL route + Γ only (CXL body excluded)",
            },
            {
                "system": "FPGA full gather (analytic)",
                "us_per_token": full,
                "speedup_vs_cpu": (cpu_mean / full) if full > 0 else None,
                "notes": "Includes CXL/HBM body DMA + Codecρ at configured BW",
            },
            {
                "system": "FPGA software sim (wall)",
                "us_per_token": fpga["fpga_sim_us_per_token_mean"],
                "notes": "Functional sim only — not silicon timing",
            },
        ],
        "e2e": {
            "num_tokens": num_tokens,
            "reps": 5,
            "modeled_metadata_speedup_vs_cpu": (cpu_mean / meta) if meta > 0 else None,
            "modeled_full_speedup_vs_cpu": (cpu_mean / full) if full > 0 else None,
            "interpretation": (
                "Primary hardware claim is metadata offload (route+Γ). "
                "Full-path latency is BW-sensitive; see bandwidth_sensitivity. "
                "fpga_sim_* is software functional sim wall-clock, not silicon."
            ),
        },
        "fpga_device_available": is_fpga_available(),
    }


def main():
    parser = argparse.ArgumentParser(description="PiKV-FPGA / CXL hardware characterization")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report = end_to_end_comparison(num_tokens=args.tokens)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    spec = report["platform"]
    bw = report["bandwidth_model"]
    e2e = report["e2e"]
    print("PiKV-FPGA Hardware Characterization")
    print("=" * 60)
    print(f"Board: {spec['board']}  Part: {spec['fpga_part']}")
    print(f"User clock: {spec['user_clock_mhz']} MHz")
    print(f"Host link: {spec['host_link']}")
    print(f"HBM2: {spec['hbm2_capacity_gb']} GB @ peak {spec['hbm2_peak_bw_gbps']} GB/s")
    print(f"Modeled CXL effective BW: {bw['cxl_effective_bw_gbps']} GB/s")
    print(f"Modeled PCIe effective BW: {bw['pcie_effective_bw_gbps']} GB/s")
    print(f"Modeled metadata latency: {bw['modeled_fpga_metadata_latency_us']:.4f} µs/token")
    print(f"Modeled full-path latency: {bw['modeled_fpga_full_latency_us']:.4f} µs/token")
    print(f"Breakdown (µs): {bw['latency_breakdown_us']}")
    print(f"BRAM budget: {report['bram_budget_kb']['total_kb']:.1f} KB")
    print("-" * 60)
    print("Comparative experiment:")
    for row in report["comparative_table"]:
        spd = row.get("speedup_vs_cpu")
        spd_s = f"  ({spd:.2f}× vs CPU)" if isinstance(spd, (int, float)) else ""
        print(f"  {row['system']:36s} {row['us_per_token']:10.3f} µs{spd_s}")
    print("-" * 60)
    print(
        f"CPU baseline: {report['cpu_baseline']['cpu_us_per_token_mean']:.2f} "
        f"± {report['cpu_baseline']['cpu_us_per_token_std']:.2f} µs/token"
    )
    print(
        f"FPGA sim wall: {report['fpga_path']['fpga_sim_us_per_token_mean']:.2f} "
        f"± {report['fpga_path']['fpga_sim_us_per_token_std']:.2f} µs/token"
    )
    print(
        f"Metadata speedup vs CPU: {e2e['modeled_metadata_speedup_vs_cpu']:.2f}×"
    )
    print(f"Device available: {report['fpga_device_available']}")
    print(e2e["interpretation"])
    print("\nBW sensitivity (µs/token):")
    for k, v in bw["bandwidth_sensitivity"].items():
        print(f"  {k:16s}  meta={v['metadata_us']:.4f}  full={v['full_us']:.4f}")


if __name__ == "__main__":
    main()
