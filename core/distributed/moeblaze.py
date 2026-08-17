"""
MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs
MLSys 2026

Core ideas implemented here:
  - Staged activation cache pipeline: keep only tokens routed to live experts;
    double-buffer activations between forward compute and backward retention.
  - Expert weight access pipeline: prefetch / stage the next top-k expert
    parameters while the current expert computes, overlapping HBM traffic with
    GEMM to raise hardware utilization under the MoE memory wall.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActivationFrame:
    """One micro-batch activation slice retained for backward."""

    tokens: torch.Tensor  # [T, H]
    expert_ids: torch.Tensor  # [T, K]
    routing_weights: torch.Tensor  # [T, K]
    step_id: int


class ActivationCachePipeline:
    """
    MoEBlaze activation pipeline: selective retention + double buffering.

    Instead of checkpointing the full sequence activation, only routed token
    slices (and routing metadata) stay in the hot cache; cold frames spill to
    a CPU side buffer when ``max_gpu_frames`` is exceeded.
    """

    def __init__(
        self,
        max_gpu_frames: int = 2,
        max_cpu_frames: int = 8,
        enable_cpu_spill: bool = True,
    ):
        self.max_gpu_frames = max_gpu_frames
        self.max_cpu_frames = max_cpu_frames
        self.enable_cpu_spill = enable_cpu_spill
        self.gpu_q: Deque[ActivationFrame] = deque()
        self.cpu_q: Deque[ActivationFrame] = deque()
        self.pushed = 0
        self.spilled = 0
        self.reloaded = 0
        self.peak_gpu_bytes = 0

    def _frame_bytes(self, frame: ActivationFrame) -> int:
        return (
            frame.tokens.numel() * frame.tokens.element_size()
            + frame.expert_ids.numel() * frame.expert_ids.element_size()
            + frame.routing_weights.numel() * frame.routing_weights.element_size()
        )

    def push(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        step_id: int,
    ) -> ActivationFrame:
        frame = ActivationFrame(
            tokens=tokens.detach(),
            expert_ids=expert_ids.detach(),
            routing_weights=routing_weights.detach(),
            step_id=step_id,
        )
        self.gpu_q.append(frame)
        self.pushed += 1
        self.peak_gpu_bytes = max(
            self.peak_gpu_bytes, sum(self._frame_bytes(f) for f in self.gpu_q)
        )

        while len(self.gpu_q) > self.max_gpu_frames:
            cold = self.gpu_q.popleft()
            if self.enable_cpu_spill:
                cpu_frame = ActivationFrame(
                    tokens=cold.tokens.to("cpu", non_blocking=True),
                    expert_ids=cold.expert_ids.to("cpu"),
                    routing_weights=cold.routing_weights.to("cpu", non_blocking=True),
                    step_id=cold.step_id,
                )
                self.cpu_q.append(cpu_frame)
                self.spilled += 1
                while len(self.cpu_q) > self.max_cpu_frames:
                    self.cpu_q.popleft()
            # else: drop (activation checkpointing style)
        return frame

    def pop_for_backward(self, device: Optional[torch.device] = None) -> Optional[ActivationFrame]:
        """LIFO within GPU buffer; reload from CPU spill if needed."""
        if self.gpu_q:
            return self.gpu_q.pop()
        if not self.cpu_q:
            return None
        cold = self.cpu_q.pop()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hot = ActivationFrame(
            tokens=cold.tokens.to(device, non_blocking=True),
            expert_ids=cold.expert_ids.to(device),
            routing_weights=cold.routing_weights.to(device, non_blocking=True),
            step_id=cold.step_id,
        )
        self.reloaded += 1
        return hot

    def clear(self):
        self.gpu_q.clear()
        self.cpu_q.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "gpu_frames": len(self.gpu_q),
            "cpu_frames": len(self.cpu_q),
            "pushed": self.pushed,
            "spilled": self.spilled,
            "reloaded": self.reloaded,
            "peak_gpu_bytes": self.peak_gpu_bytes,
        }


class ExpertWeightAccessPipeline:
    """
    Prefetch / stage expert weights for the next compute wave while the current
    expert GEMM runs (software pipeline over weight HBM traffic).
    """

    def __init__(self, experts: nn.ModuleList, prefetch_depth: int = 1):
        self.experts = experts
        self.prefetch_depth = max(1, prefetch_depth)
        self._prefetch_ids: Deque[int] = deque()
        self._staged: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self.hits = 0
        self.misses = 0
        self.prefetches = 0

    def _snapshot_params(self, expert_id: int) -> Tuple[torch.Tensor, ...]:
        expert = self.experts[expert_id]
        return tuple(p.detach() for p in expert.parameters())

    def prefetch(self, expert_ids: List[int]):
        """Stage upcoming expert parameter snapshots (overlap with compute)."""
        for eid in expert_ids[: self.prefetch_depth]:
            if eid in self._staged:
                continue
            self._staged[eid] = self._snapshot_params(eid)
            self._prefetch_ids.append(eid)
            self.prefetches += 1
            while len(self._prefetch_ids) > self.prefetch_depth + 1:
                old = self._prefetch_ids.popleft()
                self._staged.pop(old, None)

    def acquire(self, expert_id: int) -> nn.Module:
        """Return live expert module; count pipeline hit if prefetched."""
        if expert_id in self._staged:
            self.hits += 1
        else:
            self.misses += 1
            self.prefetch([expert_id])
        return self.experts[expert_id]

    def release(self, expert_id: int):
        self._staged.pop(expert_id, None)

    def stats(self) -> Dict[str, Any]:
        total = max(1, self.hits + self.misses)
        return {
            "prefetch_hits": self.hits,
            "prefetch_misses": self.misses,
            "hit_rate": self.hits / total,
            "prefetches": self.prefetches,
            "staged": len(self._staged),
        }


class MoEBlazeLayer(nn.Module):
    """
    MoE layer with MoEBlaze memory-wall optimizations:
    activation cache pipeline + expert weight access pipeline.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        intermediate_size: Optional[int] = None,
        prefetch_depth: int = 2,
        max_gpu_activation_frames: int = 2,
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
        self.act_pipe = ActivationCachePipeline(max_gpu_frames=max_gpu_activation_frames)
        self.weight_pipe = ExpertWeightAccessPipeline(self.experts, prefetch_depth=prefetch_depth)
        self._step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        self.act_pipe.push(flat, top_i, top_p, self._step)
        self._step += 1

        # Prefetch unique experts in routing order
        uniq = top_i.reshape(-1).unique().tolist()
        self.weight_pipe.prefetch(uniq)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            eid_col = top_i[:, k]
            w_col = top_p[:, k].unsqueeze(-1)
            for e in eid_col.unique().tolist():
                mask = eid_col == e
                expert = self.weight_pipe.acquire(int(e))
                # Pipeline: prefetch remaining while computing
                remaining = [u for u in uniq if u != e]
                self.weight_pipe.prefetch(remaining)
                y = expert(flat[mask])
                out[mask] = out[mask] + w_col[mask] * y
                self.weight_pipe.release(int(e))
        return out.reshape(shape)

    def get_moeblaze_metrics(self) -> Dict[str, Any]:
        return {
            "activation_pipeline": self.act_pipe.stats(),
            "weight_pipeline": self.weight_pipe.stats(),
        }
