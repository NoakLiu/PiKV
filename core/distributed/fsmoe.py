"""
FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
ASPLOS 2025

Core ideas implemented here:
  - Elastic multi-stage MoE training pipeline: Token Routing ↔ Inter-node
    Communication ↔ Expert Compute are decoupled stages with buffered queues.
  - Flexible scaling: each stage's concurrency / batching adapts to backlog so
    fixed parallel layouts no longer leave GPUs idle under routing skew.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEStage(str, Enum):
    ROUTE = "route"
    COMM = "comm"
    COMPUTE = "compute"


@dataclass
class TokenPacket:
    """Tokens destined for a set of experts after routing."""

    tokens: torch.Tensor  # [T, H]
    expert_ids: torch.Tensor  # [T]
    weights: torch.Tensor  # [T]
    src_rank: int = 0


@dataclass
class StageBacklog:
    route: int = 0
    comm: int = 0
    compute: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"route": self.route, "comm": self.comm, "compute": self.compute}


class ElasticMoEStageScheduler:
    """
    FSMoE elastic scheduler: watches queue depths and reallocates stage
    capacity (logical workers / micro-batch sizes) to the bottleneck stage.
    """

    def __init__(
        self,
        base_route_workers: int = 1,
        base_comm_workers: int = 1,
        base_compute_workers: int = 2,
        max_workers_per_stage: int = 8,
        adapt_interval: int = 8,
    ):
        self.workers = {
            MoEStage.ROUTE: base_route_workers,
            MoEStage.COMM: base_comm_workers,
            MoEStage.COMPUTE: base_compute_workers,
        }
        self.max_workers = max_workers_per_stage
        self.adapt_interval = adapt_interval
        self.step = 0
        self.adaptations = 0
        self.history: List[Dict[str, int]] = []

        self.route_q: Deque[torch.Tensor] = deque()
        self.comm_q: Deque[TokenPacket] = deque()
        self.compute_q: Deque[TokenPacket] = deque()

    def backlog(self) -> StageBacklog:
        return StageBacklog(
            route=len(self.route_q),
            comm=len(self.comm_q),
            compute=len(self.compute_q),
        )

    def enqueue_tokens(self, x: torch.Tensor):
        self.route_q.append(x)

    def maybe_adapt(self) -> Dict[str, int]:
        """Move worker slots toward the stage with the deepest backlog."""
        self.step += 1
        b = self.backlog()
        self.history.append(b.as_dict())
        if self.step % self.adapt_interval != 0:
            return dict(self.workers)

        depths = {
            MoEStage.ROUTE: b.route,
            MoEStage.COMM: b.comm,
            MoEStage.COMPUTE: b.compute,
        }
        bottleneck = max(depths, key=depths.get)
        # Prefer donating from a light-backlog stage that still has spare workers
        donors = sorted(
            (s for s in depths if self.workers[s] > 1),
            key=lambda s: (depths[s], -self.workers[s]),
        )
        if not donors:
            return {k.value: v for k, v in self.workers.items()}
        donor = donors[0]
        if (
            depths[bottleneck] > depths[donor]
            and bottleneck != donor
            and self.workers[bottleneck] < self.max_workers
        ):
            self.workers[donor] -= 1
            self.workers[bottleneck] += 1
            self.adaptations += 1
        return {k.value: v for k, v in self.workers.items()}

    def drain_route(
        self,
        gate: nn.Module,
        top_k: int,
        num_experts: int,
    ) -> List[TokenPacket]:
        """Route stage: produce per-expert packets (up to route worker budget)."""
        packets: List[TokenPacket] = []
        budget = self.workers[MoEStage.ROUTE]
        for _ in range(budget):
            if not self.route_q:
                break
            x = self.route_q.popleft()
            flat = x.reshape(-1, x.size(-1))
            probs = F.softmax(gate(flat), dim=-1)
            top_p, top_i = torch.topk(probs, top_k, dim=-1)
            top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            for k in range(top_k):
                packets.append(
                    TokenPacket(
                        tokens=flat,
                        expert_ids=top_i[:, k],
                        weights=top_p[:, k],
                    )
                )
        for p in packets:
            self.comm_q.append(p)
        return packets

    def drain_comm(self, simulate_a2a: bool = True) -> List[TokenPacket]:
        """
        Communication stage: regroup tokens by destination expert
        (stands in for inter-node all-to-all dispatch).
        """
        out: List[TokenPacket] = []
        budget = self.workers[MoEStage.COMM]
        for _ in range(budget):
            if not self.comm_q:
                break
            pkt = self.comm_q.popleft()
            if simulate_a2a:
                # Identity transform: in multi-node, this would be dist.all_to_all
                out.append(pkt)
            else:
                out.append(pkt)
            self.compute_q.append(out[-1])
        return out

    def drain_compute(self, experts: nn.ModuleList) -> torch.Tensor:
        """Compute stage: run expert MLPs for queued packets."""
        budget = self.workers[MoEStage.COMPUTE]
        acc: Optional[torch.Tensor] = None
        for _ in range(budget):
            if not self.compute_q:
                break
            pkt = self.compute_q.popleft()
            partial = torch.zeros_like(pkt.tokens)
            for e in pkt.expert_ids.unique().tolist():
                mask = pkt.expert_ids == e
                y = experts[int(e)](pkt.tokens[mask])
                w = pkt.weights[mask].unsqueeze(-1)
                partial[mask] = partial[mask] + w * y
            acc = partial if acc is None else acc + partial
        return acc if acc is not None else torch.tensor([])

    def stats(self) -> Dict[str, Any]:
        b = self.backlog()
        return {
            "workers": {k.value: v for k, v in self.workers.items()},
            "backlog": b.as_dict(),
            "adaptations": self.adaptations,
            "step": self.step,
        }


class FSMoELayer(nn.Module):
    """Sparse MoE trained under FSMoE flexible multi-stage scheduling."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        intermediate_size: Optional[int] = None,
        adapt_interval: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        inter = intermediate_size or hidden_size * 4
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, inter),
                    nn.GELU(),
                    nn.Linear(inter, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )
        self.scheduler = ElasticMoEStageScheduler(adapt_interval=adapt_interval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        self.scheduler.enqueue_tokens(x)
        self.scheduler.drain_route(self.gate, self.top_k, self.num_experts)
        self.scheduler.drain_comm(simulate_a2a=True)
        out = self.scheduler.drain_compute(self.experts)
        self.scheduler.maybe_adapt()
        if out.numel() == 0:
            return torch.zeros(shape, device=x.device, dtype=x.dtype)
        return out.reshape(shape)

    def get_fsmoe_metrics(self) -> Dict[str, Any]:
        return self.scheduler.stats()
