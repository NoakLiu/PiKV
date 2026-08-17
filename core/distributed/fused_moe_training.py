"""
PiKV Fused MoE Training Systems
================================
Integrates three paper-inspired training stacks into one runtime:

  * LAER-MoE  (ASPLOS 2026) — Fully Sharded Expert Parallel (FSEP) +
    load-adaptive expert re-layout
  * MoEBlaze  (MLSys 2026)  — activation cache + expert-weight access pipelines
    against the MoE memory wall
  * FSMoE     (ASPLOS 2025) — elastic route / comm / compute stage scheduling

Use ``create_fused_moe_training`` for a single entry point that can enable any
subset of the three subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .laer_moe import (
    FSEPShardedExpertStore,
    LAERMoELayer,
    LoadAdaptiveRelayoutPlanner,
)
from .moeblaze import ActivationCachePipeline, ExpertWeightAccessPipeline, MoEBlazeLayer
from .fsmoe import ElasticMoEStageScheduler, FSMoELayer, MoEStage, TokenPacket


@dataclass
class FusedMoETrainingConfig:
    hidden_size: int = 512
    num_experts: int = 8
    top_k: int = 2
    world_size: int = 1
    rank: int = 0
    enable_laer: bool = True
    enable_moeblaze: bool = True
    enable_fsmoe: bool = True
    relayout_interval: int = 32
    prefetch_depth: int = 2
    max_gpu_activation_frames: int = 2
    fsmoe_adapt_interval: int = 4


class FusedMoETrainingLayer(nn.Module):
    """
    One MoE layer that fuses LAER FSEP restore, MoEBlaze pipelines, and FSMoE
    elastic staging. When a subsystem is disabled, that path is skipped.
    """

    def __init__(self, cfg: FusedMoETrainingConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k

        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)

        # Shared dense experts used when FSEP is off or as compute kernels
        inter = cfg.hidden_size * 4
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.hidden_size, inter),
                    nn.GELU(),
                    nn.Linear(inter, cfg.hidden_size),
                )
                for _ in range(cfg.num_experts)
            ]
        )

        self.fsep: Optional[FSEPShardedExpertStore] = None
        self.planner: Optional[LoadAdaptiveRelayoutPlanner] = None
        if cfg.enable_laer:
            self.fsep = FSEPShardedExpertStore(
                num_experts=cfg.num_experts,
                hidden_size=cfg.hidden_size,
                world_size=cfg.world_size,
                rank=cfg.rank,
            )
            self.planner = LoadAdaptiveRelayoutPlanner(
                num_experts=cfg.num_experts,
                world_size=cfg.world_size,
                relayout_interval=cfg.relayout_interval,
            )

        self.act_pipe: Optional[ActivationCachePipeline] = None
        self.weight_pipe: Optional[ExpertWeightAccessPipeline] = None
        if cfg.enable_moeblaze:
            self.act_pipe = ActivationCachePipeline(
                max_gpu_frames=cfg.max_gpu_activation_frames
            )
            self.weight_pipe = ExpertWeightAccessPipeline(
                self.experts, prefetch_depth=cfg.prefetch_depth
            )

        self.scheduler: Optional[ElasticMoEStageScheduler] = None
        if cfg.enable_fsmoe:
            self.scheduler = ElasticMoEStageScheduler(
                adapt_interval=cfg.fsmoe_adapt_interval
            )

        self._step = 0

    def _expert_compute(self, tokens: torch.Tensor, expert_id: int) -> torch.Tensor:
        if self.cfg.enable_laer and self.fsep is not None:
            # MoEBlaze weight pipe still tracks prefetch residency for FSEP experts
            if self.weight_pipe is not None:
                if expert_id in self.weight_pipe._staged:
                    self.weight_pipe.hits += 1
                else:
                    self.weight_pipe.misses += 1
                    self.weight_pipe.prefetch([expert_id])
            return self.fsep.expert_forward(tokens, expert_id)
        if self.cfg.enable_moeblaze and self.weight_pipe is not None:
            expert = self.weight_pipe.acquire(expert_id)
            y = expert(tokens)
            self.weight_pipe.release(expert_id)
            return y
        return self.experts[expert_id](tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)

        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # --- FSMoE: record route/comm/compute pressure for elastic adapt ---
        if self.scheduler is not None:
            self.scheduler.enqueue_tokens(x)
            self.scheduler.route_q.clear()
            for k in range(self.top_k):
                self.scheduler.comm_q.append(
                    TokenPacket(
                        tokens=flat,
                        expert_ids=top_i[:, k],
                        weights=top_p[:, k],
                    )
                )
            # Move a worker-budget of packets into compute queue (A2A stand-in)
            self.scheduler.drain_comm(simulate_a2a=True)
            self.scheduler.maybe_adapt()

        if self.planner is not None:
            self.planner.observe(top_i, top_p)
            self.planner.maybe_relayout(self.fsep)

        if self.act_pipe is not None:
            self.act_pipe.push(flat, top_i, top_p, self._step)

        uniq = top_i.reshape(-1).unique().tolist()
        if self.weight_pipe is not None:
            self.weight_pipe.prefetch(uniq)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            eid_col = top_i[:, k]
            w_col = top_p[:, k].unsqueeze(-1)
            for e in eid_col.unique().tolist():
                mask = eid_col == e
                if self.weight_pipe is not None:
                    self.weight_pipe.prefetch([u for u in uniq if u != e])
                y = self._expert_compute(flat[mask], int(e))
                out[mask] = out[mask] + w_col[mask] * y
                if self.scheduler is not None and self.scheduler.compute_q:
                    self.scheduler.compute_q.popleft()

        if self.scheduler is not None:
            self.scheduler.maybe_adapt()

        self._step += 1
        return out.reshape(shape)

    def get_metrics(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"step": self._step, "config": {
            "laer": self.cfg.enable_laer,
            "moeblaze": self.cfg.enable_moeblaze,
            "fsmoe": self.cfg.enable_fsmoe,
        }}
        if self.fsep is not None:
            m["laer_fsep"] = self.fsep.stats()
        if self.planner is not None:
            m["laer_relayout"] = self.planner.stats()
        if self.act_pipe is not None:
            m["moeblaze_activation"] = self.act_pipe.stats()
        if self.weight_pipe is not None:
            m["moeblaze_weight"] = self.weight_pipe.stats()
        if self.scheduler is not None:
            m["fsmoe"] = self.scheduler.stats()
        return m


class FusedMoETrainingSystem(nn.Module):
    """Multi-layer wrapper + convenience train_step for demos / microbench."""

    def __init__(self, cfg: FusedMoETrainingConfig, num_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [FusedMoETrainingLayer(cfg) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)

    def train_step(
        self,
        batch: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        self.train()
        out = self.forward(batch)
        if target is None:
            target = batch.detach()
        loss = F.mse_loss(out, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = self.get_metrics()
        metrics["loss"] = float(loss.detach().item())
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        return {"layers": [layer.get_metrics() for layer in self.layers]}


def create_fused_moe_training(
    hidden_size: int = 512,
    num_experts: int = 8,
    top_k: int = 2,
    num_layers: int = 2,
    world_size: int = 1,
    rank: int = 0,
    enable_laer: bool = True,
    enable_moeblaze: bool = True,
    enable_fsmoe: bool = True,
    **kwargs: Any,
) -> FusedMoETrainingSystem:
    """
    Factory: fused LAER-MoE + MoEBlaze + FSMoE training system.

    Examples
    --------
    >>> model = create_fused_moe_training(hidden_size=256, num_experts=8)
    >>> y = model(torch.randn(2, 16, 256))
    """
    cfg = FusedMoETrainingConfig(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size,
        rank=rank,
        enable_laer=enable_laer,
        enable_moeblaze=enable_moeblaze,
        enable_fsmoe=enable_fsmoe,
        relayout_interval=kwargs.get("relayout_interval", 32),
        prefetch_depth=kwargs.get("prefetch_depth", 2),
        max_gpu_activation_frames=kwargs.get("max_gpu_activation_frames", 2),
        fsmoe_adapt_interval=kwargs.get("fsmoe_adapt_interval", 4),
    )
    return FusedMoETrainingSystem(cfg, num_layers=num_layers)


def create_laer_moe(**kwargs: Any) -> LAERMoELayer:
    return LAERMoELayer(
        hidden_size=kwargs.get("hidden_size", 512),
        num_experts=kwargs.get("num_experts", 8),
        top_k=kwargs.get("top_k", 2),
        world_size=kwargs.get("world_size", 1),
        rank=kwargs.get("rank", 0),
        relayout_interval=kwargs.get("relayout_interval", 32),
    )


def create_moeblaze(**kwargs: Any) -> MoEBlazeLayer:
    return MoEBlazeLayer(
        hidden_size=kwargs.get("hidden_size", 512),
        num_experts=kwargs.get("num_experts", 8),
        top_k=kwargs.get("top_k", 2),
        prefetch_depth=kwargs.get("prefetch_depth", 2),
        max_gpu_activation_frames=kwargs.get("max_gpu_activation_frames", 2),
    )


def create_fsmoe(**kwargs: Any) -> FSMoELayer:
    return FSMoELayer(
        hidden_size=kwargs.get("hidden_size", 512),
        num_experts=kwargs.get("num_experts", 8),
        top_k=kwargs.get("top_k", 2),
        adapt_interval=kwargs.get("adapt_interval", 4),
    )


__all__ = [
    "FusedMoETrainingConfig",
    "FusedMoETrainingLayer",
    "FusedMoETrainingSystem",
    "create_fused_moe_training",
    "create_laer_moe",
    "create_moeblaze",
    "create_fsmoe",
    "LAERMoELayer",
    "MoEBlazeLayer",
    "FSMoELayer",
]
