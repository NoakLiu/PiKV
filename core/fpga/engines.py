"""
Reconfigurable FPGA engines for PiKV (Table 4–5, §3.5).

Each engine mirrors a hardware block; software paths use PyTorch for simulation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    FPGACompressionEngine,
    FPGAConfig,
    FPGARoutingEngine,
    FPGASchedulingEngine,
)


class PageTableEngine:
    """Module D+: page table Γ, (t,e) ↦ address (gather/scatter)."""

    def __init__(self, cfg: FPGAConfig):
        self.cfg = cfg
        e, s = cfg.num_experts, cfg.shard_size
        # BRAM_Γ: E × S entries (token id, addr)
        self.page_table = torch.zeros(e, s, 2, dtype=torch.int32)
        self.miss_count = torch.zeros(e, dtype=torch.int32)

    def shard_id(self, token_id: int, expert_id: int) -> int:
        """s(t,e) = (t mod Ntok) ⊕ (e mod Nexp)."""
        return (token_id % self.cfg.shard_size) ^ (expert_id % self.cfg.num_experts)

    def lookup(self, token_id: int, expert_id: int) -> Tuple[int, bool]:
        s = self.shard_id(token_id, expert_id) % self.cfg.shard_size
        entry = self.page_table[expert_id % self.cfg.num_experts, s]
        if entry[0].item() == token_id:
            return int(entry[1].item()), True
        self.miss_count[expert_id % self.cfg.num_experts] += 1
        return s, False

    def insert(self, token_id: int, expert_id: int, addr: int) -> None:
        s = self.shard_id(token_id, expert_id) % self.cfg.shard_size
        e = expert_id % self.cfg.num_experts
        self.page_table[e, s] = torch.tensor([token_id, addr], dtype=torch.int32)


class ScoreFuseEngine(nn.Module):
    """
    Shared ScoreFuse + radix Top-k for routing variants (RT, RLB, RP, RE, RH).
    """

    def __init__(self, cfg: FPGAConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts)
        self.register_buffer("expert_load", torch.zeros(cfg.num_experts))
        self.register_buffer("miss_count", torch.zeros(cfg.num_experts))

    def forward(
        self,
        query: torch.Tensor,
        method: Optional[FPGARoutingEngine] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (expert_indices [B, k], expert_weights [B, k]).
        """
        method = method or self.cfg.engines.routing
        if query.dim() == 1:
            query = query.unsqueeze(0)
        logits = self.router(query)

        if method == FPGARoutingEngine.HASH:
            b = query.size(0)
            idx = torch.arange(b, device=query.device).unsqueeze(1) % self.cfg.num_experts
            experts = idx.expand(b, self.cfg.top_k)
            weights = torch.ones(b, self.cfg.top_k, device=query.device) / self.cfg.top_k
            return experts, weights

        if method == FPGARoutingEngine.LOAD_BALANCE:
            mu = self.expert_load.mean()
            penalty = 0.1 * (self.expert_load - mu)
            logits = logits - penalty.unsqueeze(0)

        elif method == FPGARoutingEngine.CACHE_AWARE:
            penalty = 0.05 * torch.log1p(self.miss_count.float())
            logits = logits - penalty.unsqueeze(0)

        elif method == FPGARoutingEngine.ENTROPY_LB:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
            logits = logits - 0.1 * entropy

        elif method == FPGARoutingEngine.HIERARCHICAL:
            coarse_k = max(2, self.cfg.top_k)
            _, coarse_idx = torch.topk(logits, coarse_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, coarse_idx, 0.0)
            logits = logits + mask

        weights, indices = torch.topk(
            F.softmax(logits, dim=-1),
            self.cfg.top_k,
            dim=-1,
        )
        for idx in indices.view(-1):
            self.expert_load[idx] += 1
        return indices, weights


class CodecRhoEngine(nn.Module):
    """Codecρ: on-chip compression/decompression (LoRA, Pyramid, FastV, …)."""

    def __init__(self, cfg: FPGAConfig):
        super().__init__()
        self.cfg = cfg
        d, d_prime, r = cfg.hidden_size, cfg.compressed_dim, cfg.lora_rank
        self.method = cfg.engines.compression

        if self.method == FPGACompressionEngine.LORA:
            self.lora_down = nn.Linear(d, r, bias=False)
            self.lora_up = nn.Linear(r, d_prime, bias=False)
        elif self.method == FPGACompressionEngine.PYRAMID:
            self.encoder = nn.Linear(d, d_prime)
            self.decoder = nn.Linear(d_prime, d)
        elif self.method in (FPGACompressionEngine.CHUNK, FPGACompressionEngine.FASTV):
            self.proj = nn.Linear(d, d_prime)
        else:
            self.mask = nn.Parameter(torch.ones(d))

    def compress(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.method == FPGACompressionEngine.LORA:
            k_hat = self.lora_up(self.lora_down(k))
            v_hat = self.lora_up(self.lora_down(v))
        elif self.method == FPGACompressionEngine.PYRAMID:
            k_hat = self.encoder(k)
            v_hat = self.encoder(v)
        elif self.method == FPGACompressionEngine.FASTV:
            k_hat = self.proj(k[..., : self.cfg.compressed_dim])
            v_hat = self.proj(v[..., : self.cfg.compressed_dim])
        elif self.method == FPGACompressionEngine.PRUNE:
            m = (self.mask > 0).float()
            k_hat = k * m.unsqueeze(0).unsqueeze(0)
            v_hat = v * m.unsqueeze(0).unsqueeze(0)
            k_hat = k_hat[..., : self.cfg.compressed_dim]
            v_hat = v_hat[..., : self.cfg.compressed_dim]
        else:
            k_hat = self.proj(k)
            v_hat = self.proj(v)
        return k_hat, v_hat

    def decompress(self, k_hat: torch.Tensor, v_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.method == FPGACompressionEngine.PYRAMID and hasattr(self, "decoder"):
            return self.decoder(k_hat), self.decoder(v_hat)
        if k_hat.shape[-1] < self.cfg.hidden_size:
            pad = self.cfg.hidden_size - k_hat.shape[-1]
            k = F.pad(k_hat, (0, pad))
            v = F.pad(v_hat, (0, pad))
            return k, v
        return k_hat, v_hat


class SchedulerFPGAEngine:
    """Scheduling engine: score pages ui, evict when ui < θ."""

    def __init__(self, cfg: FPGAConfig):
        self.cfg = cfg
        self.theta = 0.0
        self.method = cfg.engines.scheduling
        self.recency = {}
        self.attention_sum: Dict[int, float] = {}
        if self.method == FPGASchedulingEngine.QUEST:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.hidden_size * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.mlp = None

    def score(
        self,
        page_id: int,
        k_vec: torch.Tensor,
        v_vec: torch.Tensor,
        attention: float = 0.0,
        age: int = 0,
    ) -> float:
        m = self.method
        if m == FPGASchedulingEngine.H2O:
            return attention
        if m == FPGASchedulingEngine.SLIDING:
            return 1.0 if age < self.cfg.shard_size else 0.0
        if m == FPGASchedulingEngine.LRU:
            return -float(self.recency.get(page_id, 0))
        if m == FPGASchedulingEngine.ADAKV:
            r = self.recency.get(page_id, 0)
            return 0.5 * attention + 0.3 * (1.0 / (1 + r)) + 0.2 * self.attention_sum.get(page_id, 0.0)
        if m == FPGASchedulingEngine.DUO:
            return self.attention_sum.get(page_id, attention)
        if m == FPGASchedulingEngine.QUEST and self.mlp is not None:
            with torch.no_grad():
                x = torch.cat([k_vec.flatten()[: self.cfg.hidden_size],
                               v_vec.flatten()[: self.cfg.hidden_size]])
                if x.numel() < self.cfg.hidden_size * 2:
                    x = F.pad(x, (0, self.cfg.hidden_size * 2 - x.numel()))
                return self.mlp(x[: self.cfg.hidden_size * 2].unsqueeze(0)).item()
        return attention

    def should_retain(self, ui: float) -> bool:
        return ui >= self.theta

    def update_theta(self, target_hit_rate: float, observed_hit_rate: float, gamma: float = 0.01) -> None:
        """Adaptive SAdaKV / SQUEST: θ ← θ + γ(η* − η) via MMIO."""
        self.theta += gamma * (target_hit_rate - observed_hit_rate)

    def touch(self, page_id: int, attention: float = 0.0) -> None:
        self.recency[page_id] = self.recency.get(page_id, 0) + 1
        self.attention_sum[page_id] = self.attention_sum.get(page_id, 0.0) + attention
