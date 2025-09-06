"""
Enhanced MoE (Mixture of Experts) Implementation
Advanced routing strategies with normalization, LoRA, EPLB, hierarchical routing, FastMoE, and FasterMoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
try:
    from config import config
except ImportError:
    # Default config if not available
    class Config:
        def __init__(self):
            self.hidden_size = 512
            self.num_experts = 8
            self.top_k = 2
    config = Config()
import math
import os

class BaseRouter(nn.Module):
    """Base router class for MoE routing logic"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Layer normalization for router input
        self.router_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Route input to experts"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Calculate routing logits
        router_logits = self.router(x_norm)  # [batch_size, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss for load balancing
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return top_k_indices, top_k_probs, float(aux_loss.item())


class FastMoERouter(BaseRouter):
    """
    FastMoE Router - High-performance MoE implementation
    Based on: https://github.com/laekov/fastmoe
    """
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 enable_dynamic_shadowing: bool = False, enable_fuse: bool = False):
        super().__init__(hidden_size, num_experts, top_k)
        self.enable_dynamic_shadowing = enable_dynamic_shadowing
        self.enable_fuse = enable_fuse
        
        # FastMoE specific optimizations
        self.gate_noise = nn.Parameter(torch.zeros(1))
        self.gate_epsilon = 1e-2
        
        # Dynamic shadowing for load balancing
        if self.enable_dynamic_shadowing:
            self.shadow_experts = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)
            ])
            self.shadow_router = nn.Linear(hidden_size, num_experts)
        
        # Smart scheduling for communication optimization
        if self.enable_fuse:
            self.fuse_buffer = {}
            self.register_buffer('expert_utilization', torch.zeros(num_experts))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """FastMoE routing with optimizations"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Add noise for exploration (FastMoE technique)
        if self.training:
            noise = torch.randn_like(x_norm) * self.gate_noise * self.gate_epsilon
            x_norm = x_norm + noise
        
        # Calculate routing logits
        router_logits = self.router(x_norm)
        
        # Dynamic shadowing for load balancing
        if self.enable_dynamic_shadowing and self.training:
            shadow_logits = self.shadow_router(x_norm)
            router_logits = router_logits + 0.1 * shadow_logits
        
        # Apply softmax
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Smart scheduling optimization
        if self.enable_fuse:
            router_probs = self._apply_smart_scheduling(router_probs)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss
        aux_loss = self._get_fastmoe_loss(router_probs)
        
        return top_k_indices, top_k_probs, float(aux_loss.item())
    
    def _apply_smart_scheduling(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Apply smart scheduling for communication optimization"""
        # Update expert utilization
        self.expert_utilization = 0.9 * self.expert_utilization + 0.1 * router_probs.mean(dim=[0, 1])
        
        # Adjust probabilities based on utilization
        utilization_factor = 1.0 - self.expert_utilization
        adjusted_probs = router_probs * utilization_factor.unsqueeze(0).unsqueeze(0)
        
        return F.softmax(adjusted_probs, dim=-1)
    
    def _get_fastmoe_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """FastMoE specific loss calculation"""
        # Standard load balancing loss
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        load_balance_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        # FastMoE specific regularization
        if self.enable_dynamic_shadowing:
            shadow_loss = torch.mean(torch.abs(router_probs - 1.0 / self.num_experts))
            load_balance_loss = load_balance_loss + 0.01 * shadow_loss
        
        return load_balance_loss


class FasterMoERouter(BaseRouter):
    """
    FasterMoE Router - Optimized MoE with advanced features
    Based on: https://github.com/thu-pacman/FasterMoE
    """
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2,
                 enable_dynrep: bool = True, enable_fuse: bool = True, 
                 enable_hir_gate: bool = True):
        super().__init__(hidden_size, num_experts, top_k)
        self.enable_dynrep = enable_dynrep
        self.enable_fuse = enable_fuse
        self.enable_hir_gate = enable_hir_gate
        
        # FasterMoE specific components
        self.dynrep_router = nn.Linear(hidden_size, num_experts) if enable_dynrep else None
        self.fuse_scheduler = nn.Linear(hidden_size, num_experts) if enable_fuse else None
        
        # Hierarchical Intelligent Routing (HIR) Gate
        if enable_hir_gate:
            self.hir_gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, num_experts)
            )
            self.hir_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Performance tracking
        self.register_buffer('expert_performance', torch.ones(num_experts))
        self.register_buffer('communication_cost', torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """FasterMoE routing with advanced optimizations"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Hierarchical Intelligent Routing
        if self.enable_hir_gate:
            # Self-attention for context-aware routing
            attn_out, _ = self.hir_attention(x_norm, x_norm, x_norm)
            hir_logits = self.hir_gate(attn_out)
        else:
            hir_logits = torch.zeros_like(x_norm[..., :self.num_experts])
        
        # Dynamic replication routing
        if self.enable_dynrep and self.dynrep_router is not None:
            dynrep_logits = self.dynrep_router(x_norm)
        else:
            dynrep_logits = torch.zeros_like(x_norm[..., :self.num_experts])
        
        # Smart fuse scheduling
        if self.enable_fuse and self.fuse_scheduler is not None:
            fuse_logits = self.fuse_scheduler(x_norm)
        else:
            fuse_logits = torch.zeros_like(x_norm[..., :self.num_experts])
        
        # Combine all routing strategies
        base_logits = self.router(x_norm)
        combined_logits = (base_logits + 
                          0.3 * hir_logits + 
                          0.2 * dynrep_logits + 
                          0.1 * fuse_logits)
        
        # Apply performance-aware routing
        performance_weights = self.expert_performance.unsqueeze(0).unsqueeze(0)
        combined_logits = combined_logits * performance_weights
        
        # Apply softmax
        router_probs = F.softmax(combined_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate FasterMoE specific loss
        aux_loss = self._get_fastermoe_loss(router_probs, x_norm)
        
        return top_k_indices, top_k_probs, float(aux_loss.item())
    
    def _get_fastermoe_loss(self, router_probs: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        """FasterMoE specific loss calculation"""
        # Standard load balancing loss
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        load_balance_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        # Communication cost optimization
        if self.enable_fuse:
            comm_cost = torch.mean(torch.abs(router_probs - 1.0 / self.num_experts))
            load_balance_loss = load_balance_loss + 0.05 * comm_cost
        
        # Performance-aware regularization
        if self.enable_hir_gate:
            performance_loss = torch.mean(torch.abs(self.expert_performance - 1.0))
            load_balance_loss = load_balance_loss + 0.01 * performance_loss
        
        return load_balance_loss
    
    def update_expert_performance(self, expert_idx: int, performance: float):
        """Update expert performance for adaptive routing"""
        self.expert_performance[expert_idx] = 0.9 * self.expert_performance[expert_idx] + 0.1 * performance


class EPLBRouter(BaseRouter):
    """EPLB (Expert Parallel Load Balancing) Router"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, balance_coefficient: float = 0.01):
        super().__init__(hidden_size, num_experts, top_k)
        self.balance_coefficient = balance_coefficient
        
        # Load balancing network
        self.load_balancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # Expert load tracking
        self.register_buffer('expert_loads', torch.zeros(num_experts))
        self.register_buffer('total_load', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """EPLB routing with load balancing"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Main routing
        router_logits = self.router(x_norm)
        
        # Load balancing routing
        load_balance_logits = self.load_balancer(x_norm)
        
        # Combine routing with load balancing
        combined_logits = router_logits + self.balance_coefficient * load_balance_logits
        
        # Apply softmax
        router_probs = F.softmax(combined_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Update expert loads
        self._update_loads(router_probs)
        
        # Calculate auxiliary loss
        aux_loss = self._get_load_balance_loss()
        
        return top_k_indices, top_k_probs, float(aux_loss.item())
    
    def _update_loads(self, router_probs: torch.Tensor):
        """Update expert load tracking"""
        expert_loads = router_probs.mean(dim=[0, 1])
        self.expert_loads = 0.9 * self.expert_loads + 0.1 * expert_loads
        self.total_load = self.expert_loads.sum()
    
    def _get_load_balance_loss(self) -> torch.Tensor:
        """Calculate load balancing loss"""
        target_load = self.total_load / self.num_experts
        load_imbalance = torch.sum((self.expert_loads - target_load) ** 2)
        return load_imbalance


class HierarchicalRouter(BaseRouter):
    """Hierarchical Router for large-scale expert systems"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 num_hierarchies: int = 2):
        super().__init__(hidden_size, num_experts, top_k)
        self.num_hierarchies = num_hierarchies
        self.experts_per_hierarchy = num_experts // num_hierarchies
        
        # Hierarchical routing networks
        self.hierarchy_routers = nn.ModuleList([
            nn.Linear(hidden_size, self.experts_per_hierarchy) 
            for _ in range(num_hierarchies)
        ])
        
        # Top-level router
        self.top_router = nn.Linear(hidden_size, num_hierarchies)
        
        # Hierarchy attention
        self.hierarchy_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Hierarchical routing"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Top-level routing to select hierarchy
        top_logits = self.top_router(x_norm)  # [batch_size, seq_len, num_hierarchies]
        top_probs = F.softmax(top_logits, dim=-1)
        
        # Apply hierarchy attention
        attn_out, _ = self.hierarchy_attention(x_norm, x_norm, x_norm)
        
        # Hierarchical expert routing
        expert_logits = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        
        for h in range(self.num_hierarchies):
            start_idx = h * self.experts_per_hierarchy
            end_idx = start_idx + self.experts_per_hierarchy
            
            hierarchy_logits = self.hierarchy_routers[h](attn_out)
            expert_logits[:, :, start_idx:end_idx] = hierarchy_logits * top_probs[:, :, h:h+1]
        
        # Apply softmax
        router_probs = F.softmax(expert_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss
        aux_loss = self._get_hierarchical_loss(router_probs, top_probs)
        
        return top_k_indices, top_k_probs, float(aux_loss.item())
    
    def _get_hierarchical_loss(self, router_probs: torch.Tensor, top_probs: torch.Tensor) -> torch.Tensor:
        """Calculate hierarchical routing loss"""
        # Standard load balancing loss
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        load_balance_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        # Hierarchy balance loss
        hierarchy_balance = top_probs.mean(dim=[0, 1])
        hierarchy_loss = torch.sum(hierarchy_balance * torch.log(hierarchy_balance * self.num_hierarchies + 1e-9))
        
        return load_balance_loss + 0.1 * hierarchy_loss


class SinkhornRouter(BaseRouter):
    """Sinkhorn/OT load-balanced router.
    Approximates a doubly-stochastic assignment via Sinkhorn iterations on routing
    scores, then performs top-k rounding. Adds OT-style balance regularization.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        sinkhorn_iters: int = 10,
        tau: float = 1.0,
        tau_anneal: float = 0.0,
    ):
        super().__init__(hidden_size, num_experts, top_k)
        self.capacity_factor = capacity_factor
        self.sinkhorn_iters = sinkhorn_iters
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.tau_anneal = float(tau_anneal)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        batch_size, seq_len, _ = x.shape
        x_norm = self.router_norm(x)

        logits = self.router(x_norm) / torch.clamp(self.tau, min=1e-6)
        scores = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)

        # Sinkhorn normalization to near doubly-stochastic per (batch, seq_len)
        # scores: [B, T, E] → normalize rows (tokens) and columns (experts)
        assign = scores
        # Target marginals
        row_target = torch.ones(batch_size, seq_len, device=x.device)
        total_tokens = float(batch_size * seq_len)
        expert_capacity = self.capacity_factor * total_tokens / float(self.num_experts)
        col_target = torch.full((batch_size, self.num_experts), expert_capacity, device=x.device)

        for _ in range(self.sinkhorn_iters):
            # Row normalization
            row_sum = assign.sum(dim=-1, keepdim=True) + 1e-9
            assign = assign / row_sum
            # Column normalization (per batch)
            col_sum = assign.sum(dim=1, keepdim=True) + 1e-9  # [B,1,E]
            assign = assign / col_sum * (expert_capacity)

        # Convert assignment to probabilities per token
        router_probs = assign / (assign.sum(dim=-1, keepdim=True) + 1e-9)

        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # OT balance regularization
        row_sum_final = assign.sum(dim=-1)  # [B,T]
        col_sum_final = assign.sum(dim=1)   # [B,E]
        l_row = torch.mean((row_sum_final - 1.0) ** 2)
        l_col = torch.mean((col_sum_final - expert_capacity) ** 2)
        aux_loss = l_row + l_col

        # Temperature annealing
        if self.training and self.tau_anneal > 0.0:
            self.tau.data = torch.clamp(self.tau.data - self.tau_anneal, min=0.1)

        return top_k_indices, top_k_probs, float(aux_loss.item())


class PERouter(BaseRouter):
    """Uncertainty-aware router (Predictive Entropy / Gradient Norm proxy).
    Adjusts effective top_k and capacity factor based on token uncertainty.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        uncertainty_metric: str = "attn_entropy",
        min_k: int = 1,
        max_k: int = 4,
        base_capacity_factor: float = 1.0,
    ):
        super().__init__(hidden_size, num_experts, top_k)
        self.uncertainty_metric = uncertainty_metric
        self.min_k = int(min_k)
        self.max_k = int(max_k)
        self.base_capacity_factor = float(base_capacity_factor)
        # lightweight self-attn to approximate predictive distribution for entropy
        self._attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

    def _predictive_entropy(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, attn_w = self._attn(x, x, x)  # attn_w: [B, H, T, T]
        p = attn_w.mean(dim=1)  # [B, T, T]
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)  # [B,T]
        return entropy

    def _grad_norm_proxy(self, x: torch.Tensor) -> torch.Tensor:
        # Proxy: local variance magnitude as uncertainty estimate
        var = torch.mean((x - x.mean(dim=-1, keepdim=True)) ** 2, dim=-1)  # [B,T]
        return var

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        batch_size, seq_len, _ = x.shape
        x_norm = self.router_norm(x)

        logits = self.router(x_norm)
        probs = F.softmax(logits, dim=-1)

        # Compute uncertainty per token
        if self.uncertainty_metric == "attn_entropy":
            unc = self._predictive_entropy(x_norm)
        elif self.uncertainty_metric == "grad_norm":
            unc = self._grad_norm_proxy(x_norm)
        else:
            unc = torch.zeros(batch_size, seq_len, device=x.device)

        # Normalize uncertainty to [0,1]
        unc_norm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-9)
        # Map to per-token k in [min_k, max_k]
        k_real = self.min_k + (self.max_k - self.min_k) * unc_norm  # [B,T]
        k_used = torch.clamp(k_real.round().long(), min=self.min_k, max=self.max_k)

        # Use maximum k for a single topk op, then mask by token-specific k
        k_global = int(self.max_k)
        top_probs_all, top_idx_all = torch.topk(probs, k=k_global, dim=-1)
        # Build masks per token
        arange_k = torch.arange(k_global, device=x.device).view(1, 1, k_global)
        k_expanded = k_used.unsqueeze(-1)
        mask = (arange_k < k_expanded).float()
        top_k_probs = top_probs_all * mask
        # Renormalize
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        top_k_indices = top_idx_all

        # Simple aux: encourage low-uncertainty tokens to use smaller k
        aux_loss = torch.mean((k_real / float(self.max_k)) ** 2)

        return top_k_indices, top_k_probs, float(aux_loss.item())


class BARouter(BaseRouter):
    """Budget-Aware Router.
    Given memory/latency budgets, greedily selects experts by gain/cost.
    Provides set_budget(mem=..., latency=...) API. Costs are simple proxies.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        mem_budget: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        cost_per_expert_mem: float = 1.0,
        cost_per_expert_latency: float = 1.0,
    ):
        super().__init__(hidden_size, num_experts, top_k)
        self.mem_budget = mem_budget
        self.latency_budget_ms = latency_budget_ms
        self.cost_per_expert_mem = float(cost_per_expert_mem)
        self.cost_per_expert_latency = float(cost_per_expert_latency)

    def set_budget(self, mem: Optional[float] = None, latency: Optional[float] = None):
        if mem is not None:
            self.mem_budget = float(mem)
        if latency is not None:
            self.latency_budget_ms = float(latency)

    def _budget_to_k(self) -> int:
        # Map budgets to an effective k; simplistic proxy
        k = self.top_k
        if self.mem_budget is not None:
            k = min(self.num_experts, max(1, int(self.mem_budget / max(self.cost_per_expert_mem, 1e-6))))
        if self.latency_budget_ms is not None:
            k_latency = min(self.num_experts, max(1, int(self.latency_budget_ms / max(self.cost_per_expert_latency, 1e-6))))
            k = min(k, k_latency)
        return max(1, int(k))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        x_norm = self.router_norm(x)
        logits = self.router(x_norm)
        probs = F.softmax(logits, dim=-1)

        # Determine effective k from budgets
        eff_k = self._budget_to_k()
        top_k_probs, top_k_indices = torch.topk(probs, k=eff_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Aux: encourage staying within budget via small k
        aux_loss = torch.tensor(float(eff_k) / float(self.num_experts), device=x.device)
        return top_k_indices, top_k_probs, float(aux_loss.item())


class FlexMoERouter(BaseRouter):
    """Flex-MoE Router for multimodal learning"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 num_modalities: int = 2):
        super().__init__(hidden_size, num_experts, top_k)
        self.num_modalities = num_modalities
        self.experts_per_modality = num_experts // num_modalities
        
        # Modality-specific routers
        self.modality_routers = nn.ModuleList([
            nn.Linear(hidden_size, self.experts_per_modality)
            for _ in range(num_modalities)
        ])
        
        # Modality fusion network
        self.modality_fusion = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_modalities)
        )
    
    def forward(self, x: torch.Tensor, modality_info: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Flex-MoE routing with modality awareness"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Modality fusion
        if modality_info is not None:
            modality_weights = F.softmax(self.modality_fusion(modality_info), dim=-1)
        else:
            modality_weights = torch.ones(batch_size, seq_len, self.num_modalities, device=x.device) / self.num_modalities
        
        # Expert routing per modality
        expert_logits = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        
        for m in range(self.num_modalities):
            start_idx = m * self.experts_per_modality
            end_idx = start_idx + self.experts_per_modality
            
            modality_logits = self.modality_routers[m](x_norm)
            expert_logits[:, :, start_idx:end_idx] = modality_logits * modality_weights[:, :, m:m+1]
        
        # Apply softmax
        router_probs = F.softmax(expert_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss
        aux_loss = self._get_flexmoe_loss(router_probs, modality_weights)
        
        return top_k_indices, top_k_probs, float(aux_loss.item())
    
    def _get_flexmoe_loss(self, router_probs: torch.Tensor, modality_weights: torch.Tensor) -> torch.Tensor:
        """Calculate Flex-MoE specific loss"""
        # Standard load balancing loss
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        load_balance_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        # Modality balance loss
        modality_balance = modality_weights.mean(dim=[0, 1])
        modality_loss = torch.sum(modality_balance * torch.log(modality_balance * self.num_modalities + 1e-9))
        
        return load_balance_loss + 0.1 * modality_loss


class TimeMoERouter(BaseRouter):
    """Time-MoE Router for time series prediction"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 temporal_window: int = 10):
        super().__init__(hidden_size, num_experts, top_k)
        self.temporal_window = temporal_window
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        
        # Temporal routing network
        self.temporal_router = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Time embedding
        self.time_embedding = nn.Embedding(temporal_window, hidden_size)
    
    def forward(self, x: torch.Tensor, time_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Time-MoE routing with temporal awareness"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Temporal attention
        attn_out, _ = self.temporal_attention(x_norm, x_norm, x_norm)
        
        # Time embedding
        if time_steps is not None:
            time_emb = self.time_embedding(time_steps)
        else:
            time_emb = torch.zeros_like(x_norm)
        
        # Combine temporal information
        temporal_input = torch.cat([attn_out, time_emb], dim=-1)
        
        # Calculate routing logits
        router_logits = self.temporal_router(temporal_input)
        
        # Apply softmax
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss
        aux_loss = self._get_timemoe_loss(router_probs)
        
        return top_k_indices, top_k_probs, float(aux_loss.item())
    
    def _get_timemoe_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Calculate Time-MoE specific loss"""
        # Standard load balancing loss
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        load_balance_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        # Temporal consistency loss
        temporal_consistency = torch.mean(torch.abs(router_probs[:, 1:] - router_probs[:, :-1]))
        
        return load_balance_loss + 0.05 * temporal_consistency


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for efficient fine-tuning"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Original weights (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        # Original computation
        original_out = F.linear(x, self.weight)
        
        # LoRA computation: x @ A.T @ B.T = x @ (B @ A).T
        lora_out = F.linear(x, (self.lora_B @ self.lora_A).T)
        
        return original_out + self.scaling * lora_out


class MoE(nn.Module):
    """Enhanced Mixture of Experts with multiple routing strategies"""
    
    def __init__(self, hidden_size: int, num_experts: int, expert_size: Optional[int] = None,
                 router_type: str = "base", top_k: int = 2, use_normalization: bool = True,
                 use_lora: bool = False, **router_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_size = expert_size or hidden_size
        self.top_k = top_k
        self.use_normalization = use_normalization
        self.use_lora = use_lora
        
        # Create router based on type
        self.router = self._create_router(router_type, **router_kwargs)
        
        # Create experts
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(num_experts)
        ])
        
        # Normalization layers
        if use_normalization:
            self.input_norm = nn.LayerNorm(hidden_size)
            self.output_norm = nn.LayerNorm(hidden_size)
        
        # LoRA layers
        if use_lora:
            self.lora_layers = nn.ModuleList([
                LoRALayer(hidden_size, hidden_size) for _ in range(num_experts)
            ])
    
    def _create_router(self, router_type: str, **kwargs) -> BaseRouter:
        """Create router based on type"""
        router_map = {
            "base": BaseRouter,
            "eplb": EPLBRouter,
            "hierarchical": HierarchicalRouter,
            "flex": FlexMoERouter,
            "time": TimeMoERouter,
            "fastmoe": FastMoERouter,
            "fastermoe": FasterMoERouter,
            "sinkhorn": SinkhornRouter,
            "pe": PERouter,
            "budget": BARouter
        }
        
        if router_type not in router_map:
            raise ValueError(f"Unknown router type: {router_type}")
        
        return router_map[router_type](self.hidden_size, self.num_experts, self.top_k, **kwargs)
    
    def _create_expert(self) -> nn.Module:
        """Create expert network"""
        if self.use_lora:
            return LoRALayer(self.hidden_size, self.hidden_size)
        else:
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.expert_size),
                nn.ReLU(),
                nn.Linear(self.expert_size, self.hidden_size)
            )
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, float]:
        """Forward pass through MoE"""
        batch_size, seq_len, _ = x.shape
        
        # Input normalization
        if self.use_normalization:
            x = self.input_norm(x)
        
        # Route to experts
        if isinstance(self.router, (FlexMoERouter, TimeMoERouter)):
            top_k_indices, top_k_probs, aux_loss = self.router(x, **kwargs)
        else:
            top_k_indices, top_k_probs, aux_loss = self.router(x)
        
        # Process through experts (simplified for now)
        expert_output = self.experts[0](x)
        
        # Output normalization
        if self.use_normalization:
            expert_output = self.output_norm(expert_output)
        
        return expert_output, float(aux_loss)


class PiKVMoE(MoE):
    """PiKV-specific MoE with knowledge distillation"""
    
    def __init__(self, hidden_size: int, num_experts: int, expert_size: Optional[int] = None,
                 router_type: str = "base", top_k: int = 2, use_normalization: bool = True,
                 use_lora: bool = False, use_distillation: bool = True, **router_kwargs):
        super().__init__(hidden_size, num_experts, expert_size, router_type, top_k, 
                        use_normalization, use_lora, **router_kwargs)
        
        self.use_distillation = use_distillation
        
        # Knowledge distillation components
        if use_distillation:
            self.teacher_projection = nn.Linear(hidden_size, hidden_size)
            self.student_projection = nn.Linear(hidden_size, hidden_size)
            self.distillation_loss_weight = 0.1
    
    def forward(self, x: torch.Tensor, teacher_output: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, float]:
        """Forward pass with knowledge distillation"""
        # Standard MoE forward pass
        output, aux_loss = super().forward(x, **kwargs)
        
        # Knowledge distillation
        if self.use_distillation and teacher_output is not None:
            teacher_proj = self.teacher_projection(teacher_output)
            student_proj = self.student_projection(output)
            
            distillation_loss = F.mse_loss(student_proj, teacher_proj.detach())
            aux_loss = aux_loss + self.distillation_loss_weight * distillation_loss
        
        return output, float(aux_loss)


def create_moe(hidden_size: int = 512, num_experts: int = 8, expert_size: Optional[int] = None,
               router_type: str = "base", top_k: int = 2, use_normalization: bool = True,
               use_lora: bool = False, use_distillation: bool = False, **kwargs) -> Union[MoE, PiKVMoE]:
    """Factory function to create MoE models"""
    
    if use_distillation:
        return PiKVMoE(hidden_size, num_experts, expert_size, router_type, top_k,
                       use_normalization, use_lora, use_distillation, **kwargs)
    else:
        return MoE(hidden_size, num_experts, expert_size, router_type, top_k,
                   use_normalization, use_lora, **kwargs)
