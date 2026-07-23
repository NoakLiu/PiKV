"""
Adaptive routing strategy used by PiKV MoE models.

Provides AdaptiveRouter with the interface expected by pikv_moe and
enhanced_pikv_moe:
  routing_weights, expert_indices, top_k_weights, lb_loss, importance = router(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptiveRouter(nn.Module):
    """Top-k adaptive router with optional temperature scaling."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.temperature = max(temperature, 1e-6)

        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        )

        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.register_buffer("expert_loads", torch.zeros(num_experts))
        self.load_balance_weight = 0.01

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            routing_weights: [B, S, E] full softmax probabilities
            expert_indices: [B, S, K] selected expert indices
            top_k_weights: [B, S, K] normalized top-k weights
            lb_loss: scalar load-balancing loss
            importance: [B, S] token importance scores
        """
        logits = self.router(x) / self.temperature
        routing_weights = F.softmax(logits, dim=-1)

        top_k_weights, expert_indices = torch.topk(
            routing_weights, k=self.top_k, dim=-1
        )
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        importance = self.importance_predictor(x).squeeze(-1)

        # Track loads for diagnostics
        with torch.no_grad():
            batch_loads = routing_weights.mean(dim=[0, 1])
            self.expert_loads.mul_(0.9).add_(batch_loads, alpha=0.1)

        # Load-balance auxiliary loss (encourage uniform expert usage)
        mean_prob = routing_weights.mean(dim=[0, 1])
        lb_loss = self.load_balance_weight * self.num_experts * torch.sum(
            mean_prob * torch.log(mean_prob * self.num_experts + 1e-9)
        )

        return routing_weights, expert_indices, top_k_weights, lb_loss, importance
