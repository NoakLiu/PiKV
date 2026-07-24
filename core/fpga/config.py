"""
FPGA configuration and resource budgeting (PiKV paper §3.5, Table 4–5).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class FPGARoutingEngine(str, Enum):
    """Routing method → FPGA engine (Table 4)."""
    HASH = "rb"           # hash / round-robin — Dlookup O(1)
    TOPK = "rt"           # TopK softmax — ScoreFuse + radix Top-k
    LOAD_BALANCE = "rlb"  # load balance — ScoreFuse − α(μe − μ̄)
    CACHE_AWARE = "rp"    # cache-aware — ScoreFuse − λ log(1+me) + prefetch
    ENTROPY_LB = "re"     # entropy-penalised LB — ScoreFuse − β H(pe)
    HIERARCHICAL = "rh"   # 2-stage ScoreFuse


class FPGACompressionEngine(str, Enum):
    """Compression method → FPGA engine (Table 4)."""
    LORA = "clo"          # rank-r URAM matvec O(dr)
    PYRAMID = "cpy"       # multi-level Codecρ O(d)
    CHUNK = "cch"         # block-PCA O(dr)
    FASTV = "cf"          # tail crop + pad O(d)
    PRUNE = "cpr"         # sparse mask gen O(d)


class FPGASchedulingEngine(str, Enum):
    """Scheduling method → FPGA engine (Table 4)."""
    H2O = "sh2o"          # max-reduce attention O(K)
    SLIDING = "ssl"       # age comparator O(1)
    QUEST = "squest"      # DSP MLP + θ BRAM O(dK)
    LRU = "slru"          # recency sort network O(K log K)
    ADAKV = "sadakv"      # fused ∑ αjφj; MMIO θ O(K)
    DUO = "sduo"          # L-way acc on a^(ℓ)_i O(LK)


@dataclass
class FPGAEngineMapping:
    """Per-stage engine selection for PiKV-FPGA stack."""
    routing: FPGARoutingEngine = FPGARoutingEngine.TOPK
    compression: FPGACompressionEngine = FPGACompressionEngine.LORA
    scheduling: FPGASchedulingEngine = FPGASchedulingEngine.H2O

    def latency_cycles(self, num_experts: int, top_k: int, hidden_size: int) -> Dict[str, int]:
        """Rough on-chip cycle estimates from Table 4 (Tfpga column)."""
        e, k, d = num_experts, top_k, hidden_size
        route = {
            FPGARoutingEngine.HASH: 1,
            FPGARoutingEngine.TOPK: max(1, int(e * (k.bit_length() or 1))),
            FPGARoutingEngine.LOAD_BALANCE: e,
            FPGARoutingEngine.CACHE_AWARE: e,
            FPGARoutingEngine.ENTROPY_LB: e,
            FPGARoutingEngine.HIERARCHICAL: e + k * max(1, (k).bit_length()),
        }[self.routing]
        comp = {
            FPGACompressionEngine.LORA: d * 4,  # rank r=4 default
            FPGACompressionEngine.PYRAMID: d,
            FPGACompressionEngine.CHUNK: d * 4,
            FPGACompressionEngine.FASTV: d,
            FPGACompressionEngine.PRUNE: d,
        }[self.compression]
        sched = {
            FPGASchedulingEngine.H2O: 16,
            FPGASchedulingEngine.SLIDING: 1,
            FPGASchedulingEngine.QUEST: d * 16,
            FPGASchedulingEngine.LRU: 16 * max(1, (16).bit_length()),
            FPGASchedulingEngine.ADAKV: 16,
            FPGASchedulingEngine.DUO: 16 * 2,
        }[self.scheduling]
        return {"route": route, "compress": comp, "schedule": sched}


@dataclass
class FPGAConfig:
    """
    PiKV-FPGA topology: GPU MMIO → PiKV-CTRL → engines → CXL.mem DDR pool.

    Default tile matches paper: E=64, S=256, k=4, K=16, d=128.
    """
    num_experts: int = 64
    shard_size: int = 256
    top_k: int = 4
    pages_per_gpu: int = 16
    hidden_size: int = 128
    compression_ratio: float = 4.0  # ρ = d / d'
    lora_rank: int = 4
    fpga_freq_ghz: float = 0.3
    mem_bandwidth_gbps: float = 64.0
    mmio_queue_depth: int = 32
    cxl_link_width: int = 64
    engines: FPGAEngineMapping = field(default_factory=FPGAEngineMapping)
    device_path: Optional[str] = None  # e.g. /dev/pikv_fpga0
    simulate: bool = True

    @property
    def compressed_dim(self) -> int:
        return max(1, int(self.hidden_size / self.compression_ratio))


def estimate_bram_budget(cfg: FPGAConfig) -> Dict[str, float]:
    """
    On-chip BRAM/URAM budget (bytes), paper §3.5.

    BRAM_Γ = E·S·(32+48), BRAM_meta = k·K·S·(16+16+16), URAM_W = d·r
    """
    e, s, k, ks, d, r = (
        cfg.num_experts,
        cfg.shard_size,
        cfg.top_k,
        cfg.pages_per_gpu,
        cfg.hidden_size,
        cfg.lora_rank,
    )
    # Bit-widths in paper → bytes per entry
    bram_gamma = e * s * ((32 + 48 + 7) // 8)
    bram_meta = k * ks * s * ((16 + 16 + 16 + 7) // 8)
    uram_w = d * r * 2  # 16-bit weights typical on FPGA
    return {
        "bram_page_table_kb": bram_gamma / 1024,
        "bram_metadata_kb": bram_meta / 1024,
        "uram_lora_kb": uram_w / 1024,
        "total_kb": (bram_gamma + bram_meta + uram_w) / 1024,
    }


def estimate_fpga_latency_us(
    cfg: FPGAConfig,
    num_active_pages: Optional[int] = None,
    path: str = "full",
) -> float:
    """
    T_fpga latency model (paper §3.5), in microseconds.

    path:
      - "metadata": route + page-table Γ only (no CXL body DMA) — SmartNIC offload claim
      - "full": route + Γ + active-page DDR/CXL gather + codec (end-to-end KV path)
    """
    import math

    e, k, ks, d_prime = cfg.num_experts, cfg.top_k, cfg.pages_per_gpu, cfg.compressed_dim
    f_fpga = cfg.fpga_freq_ghz * 1e9
    b_mem = cfg.mem_bandwidth_gbps * 1e9 / 8

    t_route = math.ceil(e / 16) / f_fpga
    t_gamma = 2 / f_fpga
    t_ddr = 2 * d_prime / b_mem
    t_codec = cfg.engines.latency_cycles(e, k, cfg.hidden_size)["compress"] / f_fpga

    if path == "metadata":
        # One Γ probe per selected expert; no body DMA
        t_fpga = t_route + k * t_gamma
        return t_fpga * 1e6

    # Active pages fetched per token (bounded); default uses top-k experts × 1 page
    # rather than k×K worst-case, which overstates gather cost.
    default_pages = k  # one hot page per selected expert
    pt = min(default_pages, num_active_pages or default_pages)
    t_fpga = t_route + k * t_gamma + pt * (t_ddr + t_codec)
    return t_fpga * 1e6  # microseconds


def estimate_fpga_latency_breakdown(
    cfg: FPGAConfig, num_active_pages: Optional[int] = None
) -> Dict[str, float]:
    """Per-stage µs breakdown for comparative tables."""
    import math

    e, k, d_prime = cfg.num_experts, cfg.top_k, cfg.compressed_dim
    f_fpga = cfg.fpga_freq_ghz * 1e9
    b_mem = cfg.mem_bandwidth_gbps * 1e9 / 8
    t_route = math.ceil(e / 16) / f_fpga * 1e6
    t_gamma = (2 / f_fpga) * k * 1e6
    pt = num_active_pages or k
    t_ddr = pt * (2 * d_prime / b_mem) * 1e6
    t_codec = (
        pt
        * cfg.engines.latency_cycles(e, k, cfg.hidden_size)["compress"]
        / f_fpga
        * 1e6
    )
    return {
        "route_us": t_route,
        "page_table_us": t_gamma,
        "cxl_ddr_us": t_ddr,
        "codec_us": t_codec,
        "metadata_total_us": t_route + t_gamma,
        "full_total_us": t_route + t_gamma + t_ddr + t_codec,
    }
