"""
PiKV-FPGA host driver and execution framework (Algorithm 1 + §3.5).

Topology:
  GPU --MMIO--> PiKV-CTRL --> {D+, ScoreFuse, Codecρ, DMA} --CXL.mem--> DDR pool
  GPU <--PCIe/CXL-- packed {(K̂, V̂, idx)} for active pages P_t
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import FPGAConfig, estimate_bram_budget, estimate_fpga_latency_us
from .engines import CodecRhoEngine, PageTableEngine, SchedulerFPGAEngine, ScoreFuseEngine


class MMIOCommand(IntEnum):
    """32 B MMIO command queue opcodes."""
    NOP = 0
    ROUTE = 1
    COMPRESS = 2
    SCHEDULE = 3
    PREFETCH = 4
    UPDATE_THETA = 5
    DMA_READ = 6
    DMA_WRITE = 7


@dataclass
class MMIOEntry:
    """One MMIO command (packed to 32 bytes on hardware)."""
    cmd: MMIOCommand
    token_id: int = 0
    expert_id: int = 0
    page_id: int = 0
    arg0: float = 0.0

    def pack(self) -> bytes:
        return struct.pack(
            "iiiffxxxx",
            int(self.cmd),
            self.token_id,
            self.expert_id,
            self.page_id,
            self.arg0,
        )


class PiKVFPGACTRL:
    """
    PiKV-CTRL: MMIO command queue exposed to the GPU driver.

    In simulation mode, commands are executed immediately in software.
    """

    def __init__(self, cfg: FPGAConfig):
        self.cfg = cfg
        self.queue: List[MMIOEntry] = []
        self.device_open = False
        if cfg.device_path and os.path.exists(cfg.device_path):
            self.device_open = True

    def enqueue(self, entry: MMIOEntry) -> bool:
        if len(self.queue) >= self.cfg.mmio_queue_depth:
            return False
        self.queue.append(entry)
        return True

    def drain(self, fpga: "PiKVFPGA") -> int:
        processed = 0
        while self.queue:
            entry = self.queue.pop(0)
            fpga._dispatch_mmio(entry)
            processed += 1
        return processed


class PiKVFPGA:
    """
    End-to-end PiKV-FPGA stack: routes experts, compresses KV, schedules pages.

    GPU runs f_enc and f_attn; metadata-heavy stages run on FPGA (or sim).
    """

    def __init__(self, cfg: Optional[FPGAConfig] = None):
        self.cfg = cfg or FPGAConfig()
        self.ctrl = PiKVFPGACTRL(self.cfg)
        self.page_table = PageTableEngine(self.cfg)
        self.score_fuse = ScoreFuseEngine(self.cfg)
        self.codec = CodecRhoEngine(self.cfg)
        self.scheduler = SchedulerFPGAEngine(self.cfg)
        # Disaggregated KV store (CXL DDR pool simulation)
        self.kv_pool: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._next_addr = 0
        self.stats: Dict[str, Any] = {
            "tokens_processed": 0,
            "fpga_mmio_ops": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_to_gpu": 0,
        }

    @staticmethod
    def is_available() -> bool:
        return is_fpga_available()

    def process_token(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_id: int,
        attention_score: float = 1.0,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        One step of Algorithm 1 with FPGA offload for routing/compress/schedule.

        Returns:
            packed_kv: tensor of active compressed KV for GPU attention
            active_experts: list of expert ids in g_t
        """
        self.stats["tokens_processed"] += 1

        # 1) PiKV Routing on ScoreFuse
        expert_idx, expert_w = self.score_fuse(query)
        experts = expert_idx[0].tolist()
        self.ctrl.enqueue(MMIOEntry(MMIOCommand.ROUTE, token_id=token_id))
        self.ctrl.drain(self)

        active_pages: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        for e in experts:
            addr, hit = self.page_table.lookup(token_id, e)
            if hit:
                self.stats["cache_hits"] += 1
            else:
                self.stats["cache_misses"] += 1

            # 2) Codecρ compression
            k_hat, v_hat = self.codec.compress(
                key.unsqueeze(0) if key.dim() == 1 else key,
                value.unsqueeze(0) if value.dim() == 1 else value,
            )
            self.ctrl.enqueue(MMIOEntry(MMIOCommand.COMPRESS, token_id=token_id, expert_id=e))

            pool_id = self._next_addr
            self._next_addr += 1
            self.kv_pool[pool_id] = (k_hat.detach(), v_hat.detach())
            self.page_table.insert(token_id, e, pool_id)

            # 3) Scheduling
            ui = self.scheduler.score(pool_id, k_hat.squeeze(0), v_hat.squeeze(0), attention_score)
            self.scheduler.touch(pool_id, attention_score)
            if self.scheduler.should_retain(ui):
                active_pages.append((k_hat.squeeze(0), v_hat.squeeze(0), pool_id))

        self.ctrl.enqueue(MMIOEntry(MMIOCommand.SCHEDULE, token_id=token_id, arg0=attention_score))
        self.ctrl.drain(self)

        if not active_pages:
            return torch.zeros(0), experts

        packed = torch.stack([torch.cat([k, v]) for k, v, _ in active_pages])
        d_prime = self.cfg.compressed_dim
        self.stats["bytes_to_gpu"] += packed.numel() * 4
        # B_step ≈ 2·k·d'·|P_t|/ρ_link + k·log(E)  (paper)
        return packed, experts

    def _dispatch_mmio(self, entry: MMIOEntry) -> None:
        self.stats["fpga_mmio_ops"] += 1
        if entry.cmd == MMIOCommand.UPDATE_THETA:
            self.scheduler.update_theta(entry.arg0, entry.page_id / max(1, self.stats["tokens_processed"]))

    def get_stats(self) -> Dict[str, Any]:
        bram = estimate_bram_budget(self.cfg)
        latency_us = estimate_fpga_latency_us(self.cfg)
        hits = self.stats["cache_hits"]
        misses = self.stats["cache_misses"]
        total = hits + misses
        return {
            **self.stats,
            "hit_rate": hits / total if total else 0.0,
            "bram_budget": bram,
            "estimated_fpga_latency_us": latency_us,
            "simulate": self.cfg.simulate,
        }

    def update_scheduler_theta(self, target_hit_rate: float = 0.9) -> None:
        """Push θ update through MMIO (SAdaKV / SQUEST adaptive threshold)."""
        obs = self.stats["cache_hits"] / max(
            1, self.stats["cache_hits"] + self.stats["cache_misses"]
        )
        self.ctrl.enqueue(
            MMIOEntry(MMIOCommand.UPDATE_THETA, arg0=target_hit_rate, page_id=int(obs * 1000))
        )
        self.ctrl.drain(self)


def is_fpga_available() -> bool:
    """True if device node exists or simulation is enabled."""
    path = os.environ.get("PIKV_FPGA_DEVICE", "/dev/pikv_fpga0")
    return os.path.exists(path) or os.environ.get("PIKV_FPGA_SIM", "1") == "1"


def create_pikv_fpga(
    num_experts: int = 64,
    hidden_size: int = 128,
    top_k: int = 4,
    compression_ratio: float = 4.0,
    simulate: bool = True,
    device_path: Optional[str] = None,
    **kwargs: Any,
) -> PiKVFPGA:
    """Factory for PiKV-FPGA with paper-default tile parameters."""
    cfg = FPGAConfig(
        num_experts=num_experts,
        hidden_size=hidden_size,
        top_k=top_k,
        compression_ratio=compression_ratio,
        simulate=simulate,
        device_path=device_path,
        **{k: v for k, v in kwargs.items() if k in FPGAConfig.__dataclass_fields__},
    )
    return PiKVFPGA(cfg)
