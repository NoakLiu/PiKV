"""
LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training
ASPLOS 2026

Core ideas implemented here:
  - Fully Sharded Expert Parallel (FSEP): expert parameters are sliced across ranks;
    at dispatch time, All-to-All (or gather) restores only the experts needed for the
    current token set at expert granularity.
  - Load-adaptive re-layout planner: runtime remaps expert ownership / shard affinity
    to reduce imbalance from skewed token routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertLayout:
    """Maps each expert to owning rank(s) and local shard ids."""

    num_experts: int
    world_size: int
    # expert_id -> list of ranks that hold a shard (FSEP: typically all ranks)
    shard_owners: Dict[int, List[int]] = field(default_factory=dict)
    # expert_id -> preferred compute rank after re-layout (for local affinity)
    compute_home: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def balanced(cls, num_experts: int, world_size: int) -> "ExpertLayout":
        layout = cls(num_experts=num_experts, world_size=max(1, world_size))
        ws = layout.world_size
        for e in range(num_experts):
            # FSEP: every rank keeps a parameter shard of every expert
            layout.shard_owners[e] = list(range(ws))
            layout.compute_home[e] = e % ws
        return layout

    def remapped_homes(self, new_homes: Dict[int, int]) -> "ExpertLayout":
        out = ExpertLayout(
            num_experts=self.num_experts,
            world_size=self.world_size,
            shard_owners={k: list(v) for k, v in self.shard_owners.items()},
            compute_home=dict(new_homes),
        )
        return out


class FSEPShardedExpertStore(nn.Module):
    """
    Fully Sharded Expert Parallel parameter store.

    Each expert's weight matrix is partitioned along the intermediate dim across
    ``world_size`` shards. ``materialize_expert`` restores a full expert via
    all_gather / simulated all-to-all for the ranks that need it.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.world_size = max(1, world_size)
        self.rank = rank
        self.device = device or torch.device("cpu")

        # Per-expert local shard: [hidden, inter // world_size] (+ remainder on last rank)
        shard_w = self.intermediate_size // self.world_size
        rem = self.intermediate_size % self.world_size
        self._local_inter = shard_w + (rem if self.rank == self.world_size - 1 else 0)

        self.w1_shards = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, self._local_inter) * 0.02)
                for _ in range(num_experts)
            ]
        )
        self.w2_shards = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self._local_inter, hidden_size) * 0.02)
                for _ in range(num_experts)
            ]
        )
        self.layout = ExpertLayout.balanced(num_experts, self.world_size)
        self._materialize_count = 0
        self._bytes_restored = 0

    def _all_gather_shard(self, local: torch.Tensor) -> torch.Tensor:
        """Restore full tensor along dim=-1 (or dim=0 for w2) via all_gather."""
        if self.world_size == 1 or not dist.is_available() or not dist.is_initialized():
            return local
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        # Uneven shards: pad to max local size for NCCL, then trim
        max_elems = local.numel()
        # Use all_gather on flattened padded buffers when sizes may differ
        sizes = [self.intermediate_size // self.world_size] * self.world_size
        sizes[-1] += self.intermediate_size % self.world_size
        # For simplicity with equal-ish shards, gather then cat on last dim
        dist.all_gather(gathered, local.contiguous())
        return torch.cat(gathered, dim=-1 if local.dim() == 2 and local.size(0) == self.hidden_size else 0)

    def materialize_expert(self, expert_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FSEP restore: All-to-All / all_gather expert shards → full w1, w2.
        Returns (w1 [H, I], w2 [I, H]).
        """
        w1_local = self.w1_shards[expert_id]
        w2_local = self.w2_shards[expert_id]

        if self.world_size == 1 or not dist.is_available() or not dist.is_initialized():
            w1, w2 = w1_local, w2_local
        else:
            # Expert-granularity restore (paper: A2A by expert; here all_gather shards)
            parts_w1 = [torch.zeros_like(w1_local) for _ in range(self.world_size)]
            parts_w2 = [torch.zeros_like(w2_local) for _ in range(self.world_size)]
            dist.all_gather(parts_w1, w1_local.contiguous())
            dist.all_gather(parts_w2, w2_local.contiguous())
            w1 = torch.cat(parts_w1, dim=-1)
            w2 = torch.cat(parts_w2, dim=0)

        self._materialize_count += 1
        self._bytes_restored += (w1.numel() + w2.numel()) * w1.element_size()
        return w1, w2

    def expert_forward(self, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        """Compute expert MLP after FSEP materialize: GELU(x @ w1) @ w2."""
        w1, w2 = self.materialize_expert(expert_id)
        return F.gelu(x @ w1) @ w2

    def stats(self) -> Dict[str, Any]:
        return {
            "materialize_count": self._materialize_count,
            "bytes_restored": self._bytes_restored,
            "world_size": self.world_size,
            "local_intermediate": self._local_inter,
            "layout_homes": dict(self.layout.compute_home),
        }


class LoadAdaptiveRelayoutPlanner:
    """
    LAER runtime planner: watches per-expert token load and remaps ``compute_home``
    so hot experts prefer less-loaded ranks (adaptive dynamic re-layout).
    """

    def __init__(
        self,
        num_experts: int,
        world_size: int,
        relayout_interval: int = 32,
        imbalance_threshold: float = 0.15,
    ):
        self.num_experts = num_experts
        self.world_size = max(1, world_size)
        self.relayout_interval = relayout_interval
        self.imbalance_threshold = imbalance_threshold
        self.token_counts = torch.zeros(num_experts)
        self.step = 0
        self.relayout_events = 0
        self.layout = ExpertLayout.balanced(num_experts, self.world_size)

    def observe(self, expert_ids: torch.Tensor, token_weights: Optional[torch.Tensor] = None):
        """Accumulate routing statistics. ``expert_ids``: [..., k] expert indices."""
        flat = expert_ids.reshape(-1).detach().cpu()
        if token_weights is None:
            for e in flat.tolist():
                if 0 <= int(e) < self.num_experts:
                    self.token_counts[int(e)] += 1.0
        else:
            w = token_weights.reshape(-1).detach().cpu()
            for e, wt in zip(flat.tolist(), w.tolist()):
                if 0 <= int(e) < self.num_experts:
                    self.token_counts[int(e)] += float(wt)
        self.step += 1

    def load_imbalance(self) -> float:
        if self.token_counts.sum() <= 0:
            return 0.0
        mean = self.token_counts.mean().clamp_min(1e-6)
        return float((self.token_counts.std() / mean).item())

    def maybe_relayout(self, store: Optional[FSEPShardedExpertStore] = None) -> Optional[ExpertLayout]:
        """
        Periodically reassign compute homes: sort experts by load, assign to
        ranks in round-robin from hottest → coldest (greedy load packing).
        """
        if self.step == 0 or self.step % self.relayout_interval != 0:
            return None
        imb = self.load_imbalance()
        if imb < self.imbalance_threshold:
            return None

        order = torch.argsort(self.token_counts, descending=True).tolist()
        rank_load = [0.0] * self.world_size
        new_homes: Dict[int, int] = {}
        for e in order:
            # Place on currently lightest rank
            home = min(range(self.world_size), key=lambda r: rank_load[r])
            new_homes[e] = home
            rank_load[home] += float(self.token_counts[e])

        self.layout = self.layout.remapped_homes(new_homes)
        self.relayout_events += 1
        if store is not None:
            store.layout = self.layout
        # Decay history so planner stays reactive
        self.token_counts.mul_(0.5)
        return self.layout

    def stats(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "load_imbalance": self.load_imbalance(),
            "relayout_events": self.relayout_events,
            "token_counts": self.token_counts.tolist(),
            "compute_home": dict(self.layout.compute_home),
        }


class LAERMoELayer(nn.Module):
    """Sparse MoE layer using FSEP store + load-adaptive re-layout (LAER-MoE)."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        world_size: int = 1,
        rank: int = 0,
        relayout_interval: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.store = FSEPShardedExpertStore(
            num_experts=num_experts,
            hidden_size=hidden_size,
            world_size=world_size,
            rank=rank,
        )
        self.planner = LoadAdaptiveRelayoutPlanner(
            num_experts=num_experts,
            world_size=world_size,
            relayout_interval=relayout_interval,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H] or [T, H]
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        self.planner.observe(top_i, top_p)
        self.planner.maybe_relayout(self.store)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            eid = top_i[:, k]
            w = top_p[:, k].unsqueeze(-1)
            # Group by expert for fewer materializations
            for e in eid.unique().tolist():
                mask = eid == e
                if not mask.any():
                    continue
                y = self.store.expert_forward(flat[mask], int(e))
                out[mask] = out[mask] + w[mask] * y
        return out.reshape(shape)

    def get_laer_metrics(self) -> Dict[str, Any]:
        return {
            "fsep": self.store.stats(),
            "relayout": self.planner.stats(),
        }
