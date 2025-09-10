"""
SmartMoE: Efficiently Training Sparsely-Activated Models through Combining 
Offline and Online Parallelization

Based on the paper:
SmartMoE: Efficiently Training Sparsely-Activated Models through Combining 
Offline and Online Parallelization
Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, and Jidong Zhai
USENIX ATC 2023

This implementation provides automatic parallelization for MoE models with:
- Offline pool construction with data-sensitive performance modeling
- Online adaptive parallelization with light-weight searching
- Hybrid parallelism support (data, tensor, pipeline, expert)
- Expert placement optimization based on workload patterns
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from .config import config


class ParallelStrategy(Enum):
    """Parallel strategy types"""
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    EXPERT = "expert"


class NetworkTopology(Enum):
    """Network topology types"""
    FULLY_CONNECTED = "fully_connected"
    RING = "ring"
    MESH = "mesh"
    TREE = "tree"


@dataclass
class ExpertSlot:
    """Configuration of expert slot for parallel strategies"""
    capacity: float  # Fraction between 0 and 1
    num_slots: int   # Number of slots per worker
    num_layers: int  # Number of MoE layers per worker


@dataclass
class ExecutionPlan:
    """Parallel execution plan"""
    expert_slots: List[ExpertSlot]
    parallel_strategies: List[ParallelStrategy]
    expert_placement: Dict[int, List[int]]  # expert_id -> device_ids
    performance_estimate: float


class WorkloadPredictor:
    """
    Data-sensitive workload prediction for MoE models.
    Estimates expert selection distribution without actual training.
    """
    
    def __init__(self, num_experts: int, capacity_factor: float = 1.2):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
    def predict_expert_selection(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Predict expert selection distribution based on gating network design.
        Uses capacity factor to estimate load balancing.
        """
        # GShard-style load balancing prediction
        total_tokens = batch_size * seq_len
        tokens_per_expert = total_tokens / self.num_experts
        
        # Apply capacity factor constraint
        max_tokens_per_expert = int(tokens_per_expert * self.capacity_factor)
        
        # Simulate expert selection with load balancing
        expert_selections = torch.zeros(self.num_experts)
        
        # Distribute tokens with capacity constraint
        remaining_tokens = total_tokens
        for expert_id in range(self.num_experts):
            if remaining_tokens <= 0:
                break
                
            # Calculate tokens for this expert
            expert_tokens = min(remaining_tokens, max_tokens_per_expert)
            expert_selections[expert_id] = expert_tokens
            remaining_tokens -= expert_tokens
        
        # Normalize to get selection probabilities
        expert_selections = expert_selections / expert_selections.sum()
        
        return expert_selections
    
    def predict_topology_aware_selection(self, batch_size: int, seq_len: int, 
                                       topology: NetworkTopology) -> torch.Tensor:
        """
        Predict expert selection for topology-aware gating networks.
        """
        expert_selections = self.predict_expert_selection(batch_size, seq_len)
        
        if topology == NetworkTopology.MESH:
            # Prefer local experts in mesh topology
            # This is a simplified model - real implementation would be more complex
            local_bias = 0.7  # 70% local, 30% remote
            expert_selections = expert_selections * local_bias + (1 - local_bias) / self.num_experts
            
        return expert_selections


class PerformanceModel:
    """
    Workload-aware performance model for MoE training.
    Estimates computation and communication costs.
    """
    
    def __init__(self, hidden_size: int, num_experts: int, world_size: int):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.world_size = world_size
        
        # Hardware parameters (can be configured)
        self.compute_cost_per_token = 1.0  # Normalized compute cost
        self.communication_cost_per_byte = 0.1  # Normalized communication cost
        self.all_to_all_latency = 0.001  # All-to-all latency in seconds
        
    def estimate_computation_cost(self, expert_selections: torch.Tensor, 
                                execution_plan: ExecutionPlan) -> float:
        """Estimate computation cost for given expert selections"""
        total_cost = 0.0
        
        for expert_id, selection_prob in enumerate(expert_selections):
            if selection_prob > 0:
                # Find devices hosting this expert
                device_ids = execution_plan.expert_placement.get(expert_id, [])
                if device_ids:
                    # Cost is proportional to selection probability and number of replicas
                    cost = selection_prob * self.compute_cost_per_token / len(device_ids)
                    total_cost += cost
                    
        return total_cost
    
    def estimate_communication_cost(self, expert_selections: torch.Tensor,
                                  execution_plan: ExecutionPlan) -> float:
        """Estimate communication cost for all-to-all operations"""
        # All-to-all communication cost depends on expert selection distribution
        # More imbalanced selection leads to higher communication cost
        
        # Calculate load imbalance
        mean_load = expert_selections.mean()
        load_variance = ((expert_selections - mean_load) ** 2).mean()
        
        # Communication cost increases with load imbalance
        base_cost = self.all_to_all_latency
        imbalance_factor = 1.0 + load_variance * 2.0  # Scale with variance
        
        return base_cost * imbalance_factor
    
    def estimate_total_cost(self, expert_selections: torch.Tensor,
                          execution_plan: ExecutionPlan) -> float:
        """Estimate total training cost (computation + communication)"""
        comp_cost = self.estimate_computation_cost(expert_selections, execution_plan)
        comm_cost = self.estimate_communication_cost(expert_selections, execution_plan)
        
        return comp_cost + comm_cost


class OfflinePoolConstructor:
    """
    Offline pool construction with data-sensitive performance modeling.
    Creates optimal pools of execution plans before training.
    """
    
    def __init__(self, num_experts: int, world_size: int, hidden_size: int):
        self.num_experts = num_experts
        self.world_size = world_size
        self.hidden_size = hidden_size
        self.performance_model = PerformanceModel(hidden_size, num_experts, world_size)
        self.workload_predictor = WorkloadPredictor(num_experts)
        
    def generate_hybrid_parallelism_plans(self) -> List[ExecutionPlan]:
        """Generate candidate execution plans with hybrid parallelism"""
        plans = []
        
        # Generate different combinations of parallel strategies
        strategies_combinations = [
            [ParallelStrategy.EXPERT],
            [ParallelStrategy.EXPERT, ParallelStrategy.DATA],
            [ParallelStrategy.EXPERT, ParallelStrategy.TENSOR],
            [ParallelStrategy.EXPERT, ParallelStrategy.PIPELINE],
            [ParallelStrategy.EXPERT, ParallelStrategy.DATA, ParallelStrategy.TENSOR],
        ]
        
        for strategies in strategies_combinations:
            # Create expert slot configuration
            expert_slots = self._create_expert_slots(strategies)
            
            # Generate expert placement plans
            placement_plans = self._generate_expert_placements()
            
            for placement in placement_plans:
                plan = ExecutionPlan(
                    expert_slots=expert_slots,
                    parallel_strategies=strategies,
                    expert_placement=placement,
                    performance_estimate=0.0  # Will be calculated later
                )
                plans.append(plan)
                
        return plans
    
    def _create_expert_slots(self, strategies: List[ParallelStrategy]) -> List[ExpertSlot]:
        """Create expert slot configuration based on parallel strategies"""
        # Calculate capacity and number of slots based on strategies
        capacity = 1.0
        num_slots = self.num_experts // self.world_size
        num_layers = 1
        
        for strategy in strategies:
            if strategy == ParallelStrategy.DATA:
                capacity *= 1.0  # Data parallelism doesn't affect capacity
            elif strategy == ParallelStrategy.TENSOR:
                capacity /= 2.0  # Tensor parallelism reduces capacity per slot
            elif strategy == ParallelStrategy.PIPELINE:
                num_layers = self.world_size  # Pipeline parallelism across layers
        
        return [ExpertSlot(capacity=capacity, num_slots=num_slots, num_layers=num_layers)]
    
    def _generate_expert_placements(self) -> List[Dict[int, List[int]]]:
        """Generate different expert placement plans"""
        placements = []
        
        # Simple round-robin placement
        placement = {}
        for expert_id in range(self.num_experts):
            device_id = expert_id % self.world_size
            placement[expert_id] = [device_id]
        placements.append(placement)
        
        # Load-balanced placement (simplified)
        if self.num_experts >= self.world_size:
            placement = {}
            experts_per_device = self.num_experts // self.world_size
            for device_id in range(self.world_size):
                start_expert = device_id * experts_per_device
                end_expert = start_expert + experts_per_device
                for expert_id in range(start_expert, end_expert):
                    placement[expert_id] = [device_id]
            placements.append(placement)
        
        return placements
    
    def construct_optimal_pool(self, batch_size: int = 32, seq_len: int = 128) -> List[ExecutionPlan]:
        """Construct optimal pool of execution plans"""
        # Generate candidate plans
        candidate_plans = self.generate_hybrid_parallelism_plans()
        
        # Predict workload
        expert_selections = self.workload_predictor.predict_expert_selection(batch_size, seq_len)
        
        # Evaluate each plan
        for plan in candidate_plans:
            cost = self.performance_model.estimate_total_cost(expert_selections, plan)
            plan.performance_estimate = cost
        
        # Sort by performance (lower cost is better)
        candidate_plans.sort(key=lambda x: x.performance_estimate)
        
        # Return top-k plans as the pool
        pool_size = min(10, len(candidate_plans))
        return candidate_plans[:pool_size]


class OnlineAdaptiveOptimizer:
    """
    Online adaptive parallelization with light-weight searching.
    Performs runtime optimization within the constructed pool.
    """
    
    def __init__(self, pool: List[ExecutionPlan], world_size: int):
        self.pool = pool
        self.world_size = world_size
        self.current_plan = pool[0] if pool else None
        self.performance_history = deque(maxlen=100)
        
    def greedy_expert_placement(self, expert_workloads: torch.Tensor) -> Dict[int, List[int]]:
        """
        Greedy algorithm for expert placement optimization.
        Algorithm 1 from the SmartMoE paper.
        """
        num_experts = len(expert_workloads)
        samples_per_device = [0.0] * self.world_size
        experts_per_device = [0] * self.world_size
        placement = {i: [] for i in range(num_experts)}
        
        # Sort experts by workload (descending)
        sorted_experts = torch.argsort(expert_workloads, descending=True)
        
        for expert_id in sorted_experts:
            expert_id = expert_id.item()
            workload = expert_workloads[expert_id].item()
            
            # Find device with minimum current load
            min_load = float('inf')
            best_device = 0
            
            for device_id in range(self.world_size):
                if experts_per_device[device_id] < num_experts // self.world_size:
                    if samples_per_device[device_id] < min_load:
                        min_load = samples_per_device[device_id]
                        best_device = device_id
            
            # Place expert on best device
            placement[expert_id].append(best_device)
            samples_per_device[best_device] += workload
            experts_per_device[best_device] += 1
            
        return placement
    
    def hybrid_expert_placement(self, expert_workloads: torch.Tensor, 
                              devices_per_node: int = 4) -> Dict[int, List[int]]:
        """
        Hybrid approach combining greedy and dynamic programming.
        Algorithm 2 from the SmartMoE paper.
        """
        num_experts = len(expert_workloads)
        num_nodes = self.world_size // devices_per_node
        
        # Step 1: Greedy placement across nodes
        node_workloads = torch.zeros(num_nodes)
        node_experts = [0] * num_nodes
        node_placement = {i: [] for i in range(num_experts)}
        
        sorted_experts = torch.argsort(expert_workloads, descending=True)
        
        for expert_id in sorted_experts:
            expert_id = expert_id.item()
            workload = expert_workloads[expert_id].item()
            
            # Find node with minimum load
            min_load = float('inf')
            best_node = 0
            
            for node_id in range(num_nodes):
                if node_experts[node_id] < num_experts // num_nodes:
                    if node_workloads[node_id] < min_load:
                        min_load = node_workloads[node_id]
                        best_node = node_id
            
            node_placement[expert_id].append(best_node)
            node_workloads[best_node] += workload
            node_experts[best_node] += 1
        
        # Step 2: Dynamic programming within each node
        final_placement = {i: [] for i in range(num_experts)}
        
        for node_id in range(num_nodes):
            # Get experts assigned to this node
            node_experts_list = [eid for eid, nodes in node_placement.items() if node_id in nodes]
            
            if not node_experts_list:
                continue
                
            # Simple round-robin within node
            for i, expert_id in enumerate(node_experts_list):
                device_id = node_id * devices_per_node + (i % devices_per_node)
                final_placement[expert_id].append(device_id)
        
        return final_placement
    
    def adapt_execution_plan(self, expert_workloads: torch.Tensor, 
                           switching_threshold: float = 0.1) -> Optional[ExecutionPlan]:
        """
        Adapt execution plan based on current workload.
        Returns new plan if switching is beneficial.
        """
        if not self.pool:
            return None
        
        # Calculate current performance
        current_performance = self._calculate_plan_performance(expert_workloads, self.current_plan)
        
        # Find best plan in pool
        best_plan = None
        best_performance = float('inf')
        
        for plan in self.pool:
            performance = self._calculate_plan_performance(expert_workloads, plan)
            if performance < best_performance:
                best_performance = performance
                best_plan = plan
        
        # Check if switching is beneficial
        improvement = (current_performance - best_performance) / current_performance
        
        if improvement > switching_threshold:
            return best_plan
        
        return None
    
    def _calculate_plan_performance(self, expert_workloads: torch.Tensor, 
                                  plan: ExecutionPlan) -> float:
        """Calculate performance of a specific execution plan"""
        # Simplified performance calculation
        # In practice, this would use the performance model
        
        # Calculate load imbalance
        device_loads = [0.0] * self.world_size
        
        for expert_id, workload in enumerate(expert_workloads):
            device_ids = plan.expert_placement.get(expert_id, [])
            for device_id in device_ids:
                device_loads[device_id] += workload.item()
        
        # Performance is inversely related to load imbalance
        mean_load = sum(device_loads) / len(device_loads)
        load_variance = sum((load - mean_load) ** 2 for load in device_loads) / len(device_loads)
        
        return mean_load + load_variance  # Higher variance = worse performance


class SmartMoE(nn.Module):
    """
    SmartMoE: Automatic parallelization for MoE models.
    
    Combines offline pool construction and online adaptive optimization
    for efficient training of sparsely-activated models.
    """
    
    def __init__(self, 
                 hidden_size: int = 512,
                 num_experts: int = 8,
                 world_size: int = 1,
                 capacity_factor: float = 1.2,
                 enable_offline_optimization: bool = True,
                 enable_online_adaptation: bool = True):
        super(SmartMoE, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.world_size = world_size
        self.capacity_factor = capacity_factor
        self.enable_offline_optimization = enable_offline_optimization
        self.enable_online_adaptation = enable_online_adaptation
        
        # Initialize components
        self.workload_predictor = WorkloadPredictor(num_experts, capacity_factor)
        self.performance_model = PerformanceModel(hidden_size, num_experts, world_size)
        
        # Offline pool construction
        self.pool = []
        if enable_offline_optimization:
            self.pool_constructor = OfflinePoolConstructor(num_experts, world_size, hidden_size)
            self.pool = self.pool_constructor.construct_optimal_pool()
        
        # Online adaptive optimization
        self.adaptive_optimizer = None
        if enable_online_adaptation and self.pool:
            self.adaptive_optimizer = OnlineAdaptiveOptimizer(self.pool, world_size)
        
        # Expert models (simplified)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Workload tracking
        self.register_buffer('expert_workloads', torch.zeros(num_experts))
        self.register_buffer('iteration_count', torch.tensor(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic parallelization"""
        batch_size, seq_len, hidden_size = x.shape
        
        # Calculate gating weights
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        # Track expert workloads
        with torch.no_grad():
            expert_usage = gate_probs.sum(dim=[0, 1])  # [num_experts]
            self.expert_workloads = 0.9 * self.expert_workloads + 0.1 * expert_usage
            self.iteration_count += 1
        
        # Online adaptation
        if (self.enable_online_adaptation and 
            self.adaptive_optimizer and 
            self.iteration_count % 10 == 0):  # Adapt every 10 iterations
            
            new_plan = self.adaptive_optimizer.adapt_execution_plan(self.expert_workloads)
            if new_plan:
                self.adaptive_optimizer.current_plan = new_plan
                print(f"Switched to new execution plan at iteration {self.iteration_count}")
        
        # Process through experts (simplified)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # [batch_size, seq_len, hidden_size]
            expert_outputs.append(expert_output)
        
        # Combine expert outputs
        output = torch.zeros_like(x)
        for i, expert_output in enumerate(expert_outputs):
            output += expert_output * gate_probs[:, :, i:i+1]
        
        return output
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        metrics = {
            'expert_workloads': self.expert_workloads.tolist(),
            'load_imbalance': self.expert_workloads.var().item(),
            'iteration_count': self.iteration_count.item(),
            'pool_size': len(self.pool),
            'current_plan': self.adaptive_optimizer.current_plan if self.adaptive_optimizer else None
        }
        
        return metrics
    
    def optimize_parallelization(self, workload_data: torch.Tensor, 
                               hardware_config: Dict[str, Any]) -> ExecutionPlan:
        """Optimize parallelization for given workload and hardware"""
        if not self.pool:
            raise ValueError("No execution plans available. Enable offline optimization.")
        
        # Use current workload for optimization
        current_workloads = self.expert_workloads
        
        if self.adaptive_optimizer:
            best_plan = self.adaptive_optimizer.adapt_execution_plan(current_workloads, switching_threshold=0.0)
            if best_plan:
                return best_plan
        
        # Return best plan from pool
        return self.pool[0] if self.pool else None


# Factory function
def create_smartmoe(hidden_size: int = 512, 
                   num_experts: int = 8,
                   world_size: int = 1,
                   capacity_factor: float = 1.2,
                   enable_offline_optimization: bool = True,
                   enable_online_adaptation: bool = True) -> SmartMoE:
    """
    Factory function to create SmartMoE models.
    
    Args:
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        world_size: Number of distributed processes
        capacity_factor: Capacity factor for load balancing
        enable_offline_optimization: Enable offline pool construction
        enable_online_adaptation: Enable online adaptive optimization
    
    Returns:
        SmartMoE instance
    """
    return SmartMoE(
        hidden_size=hidden_size,
        num_experts=num_experts,
        world_size=world_size,
        capacity_factor=capacity_factor,
        enable_offline_optimization=enable_offline_optimization,
        enable_online_adaptation=enable_online_adaptation
    )
