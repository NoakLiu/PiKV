import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union
import math
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import numpy as np

from .enhanced_config import enhanced_config as config
from .kv_cache_compression import KVCacheCompressor
from .routing_strategy import AdaptiveRouter
from .lora import LoRALayer, LoRAExpert, LoRAKVCache
from .distillation import PiKVDistillation, create_teacher_model, distillation_training_step
from .shared import ExternalMemoryCache
from .cache_scheduling import CacheSchedulingManager, SchedulingPolicy
from .smartmoe import SmartMoE, create_smartmoe
from .pikv_moe import KVCache


class _ToggleFlag:
    """Boolean flag that is also callable (enable) for API compatibility."""

    __slots__ = ("_value", "_on_enable")

    def __init__(self, value: bool = True, on_enable=None):
        self._value = bool(value)
        self._on_enable = on_enable

    def __bool__(self):
        return self._value

    def __eq__(self, other):
        return self._value == bool(other)

    def __repr__(self):
        return f"{self._value}"

    def __call__(self):
        self._value = True
        if self._on_enable is not None:
            self._on_enable()
        return True

    def set(self, value: bool):
        self._value = bool(value)
        return self._value



class DynamicLoadBalancer:
    """
    Dynamic Load Balancer for solving load imbalance in MoE systems.
    Implements real-time expert selection with adaptive routing.
    """
    
    def __init__(self, num_experts: int, hidden_size: int, 
                 load_balance_threshold: float = 0.1,
                 adaptation_rate: float = 0.01):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.load_balance_threshold = load_balance_threshold
        self.adaptation_rate = adaptation_rate
        
        # Real-time load tracking
        self.expert_loads = torch.zeros(num_experts)
        self.expert_utilization = torch.zeros(num_experts)
        self.expert_performance = torch.ones(num_experts)  # Performance scores
        self.load_history = deque(maxlen=100)  # Recent load history
        
        # Adaptive routing parameters
        self.routing_weights = torch.ones(num_experts)
        self.load_penalties = torch.zeros(num_experts)
        self.performance_rewards = torch.ones(num_experts)
        
        # Load balancing network
        self.load_balancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert capacity management
        self.expert_capacities = torch.ones(num_experts) * 1.0
        self.overload_penalty = 2.0
        
    def update_load(self, expert_idx: int, load: float, performance: float = 1.0):
        """Update expert load and performance metrics"""
        self.expert_loads[expert_idx] = 0.9 * self.expert_loads[expert_idx] + 0.1 * load
        self.expert_performance[expert_idx] = 0.9 * self.expert_performance[expert_idx] + 0.1 * performance
        
        # Update utilization
        self.expert_utilization[expert_idx] = self.expert_loads[expert_idx] / self.expert_capacities[expert_idx]
        
        # Update load history
        self.load_history.append(self.expert_loads.clone())
        
    def compute_load_balance_adjustment(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute load balancing adjustments for routing"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute current load distribution
        current_loads = self.expert_loads
        target_load = current_loads.mean()
        load_imbalance = torch.abs(current_loads - target_load)
        
        # Compute load balancing adjustments
        adjustments = torch.zeros_like(current_loads)
        
        # Penalize overloaded experts
        overload_mask = self.expert_utilization > 1.0
        adjustments[overload_mask] = -self.overload_penalty * (self.expert_utilization[overload_mask] - 1.0)
        
        # Reward underloaded experts
        underload_mask = self.expert_utilization < 0.5
        adjustments[underload_mask] = self.adaptation_rate * (0.5 - self.expert_utilization[underload_mask])
        
        # Apply performance-based adjustments
        performance_adjustments = self.performance_rewards * self.expert_performance
        adjustments += performance_adjustments * self.adaptation_rate
        
        # Expand to batch dimensions
        adjustments = adjustments.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        return adjustments
    
    def adaptive_routing(self, router_logits: torch.Tensor, 
                        hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply adaptive routing with load balancing"""
        # Get load balancing adjustments
        load_adjustments = self.compute_load_balance_adjustment(hidden_states)
        
        # Apply adjustments to router logits
        adjusted_logits = router_logits + load_adjustments
        
        return adjusted_logits
    
    def get_load_balance_loss(self) -> torch.Tensor:
        """Compute load balancing loss"""
        if len(self.load_history) < 2:
            return torch.tensor(0.0)
        
        # Compute load variance as imbalance measure
        recent_loads = torch.stack(list(self.load_history))
        load_variance = recent_loads.var(dim=0).mean()
        
        # Compute utilization imbalance
        utilization_variance = self.expert_utilization.var()
        
        # Combined load balancing loss
        lb_loss = load_variance + utilization_variance
        
        return lb_loss


class AsyncExecutionManager:
    """
    Asynchronous Execution Manager for solving inefficient synchronous operations.
    Implements pipeline parallelism and async communication.
    """
    
    def __init__(self, num_experts: int, world_size: int = 1):
        self.num_experts = num_experts
        self.world_size = world_size
        self.executor = ThreadPoolExecutor(max_workers=num_experts)
        
        # Pipeline stages
        self.pipeline_stages = {
            'routing': 0,
            'expert_computation': 1,
            'communication': 2,
            'aggregation': 3
        }
        
        # Async queues for each stage
        self.stage_queues = {
            stage: asyncio.Queue() for stage in self.pipeline_stages.keys()
        }
        
        # Expert dependency tracking
        self.expert_dependencies = defaultdict(list)
        self.computation_graph = {}
        
        # Async communication buffers
        self.communication_buffers = {}
        self.pending_communications = {}
        
    def add_expert_dependency(self, expert_id: int, depends_on: List[int]):
        """Add dependency information for expert computation"""
        self.expert_dependencies[expert_id] = depends_on
        
    def async_expert_computation(self, expert_id: int, input_data: torch.Tensor, 
                               expert_model: nn.Module) -> asyncio.Future:
        """Execute expert computation asynchronously"""
        async def _compute():
            # Check dependencies
            if expert_id in self.expert_dependencies:
                for dep_expert in self.expert_dependencies[expert_id]:
                    # Wait for dependency completion
                    if dep_expert in self.pending_communications:
                        await self.pending_communications[dep_expert]
            
            # Execute expert computation
            with torch.no_grad():
                result = expert_model(input_data)
            
            return result
        
        return asyncio.create_task(_compute())
    
    def async_communication(self, expert_id: int, data: torch.Tensor, 
                          target_ranks: List[int]) -> asyncio.Future:
        """Execute communication asynchronously"""
        async def _communicate():
            if self.world_size > 1:
                # Non-blocking all-to-all communication
                communication_tasks = []
                for target_rank in target_ranks:
                    if target_rank != dist.get_rank():
                        task = asyncio.create_task(
                            self._send_data_async(data, target_rank)
                        )
                        communication_tasks.append(task)
                
                # Wait for all communications to complete
                await asyncio.gather(*communication_tasks)
            
            return data
        
        future = asyncio.create_task(_communicate())
        self.pending_communications[expert_id] = future
        return future
    
    async def _send_data_async(self, data: torch.Tensor, target_rank: int):
        """Send data to target rank asynchronously"""
        # This would be implemented with actual async communication
        # For now, we'll simulate with a small delay
        await asyncio.sleep(0.001)
        return data
    
    async def pipeline_execution(self, input_data: torch.Tensor, 
                          expert_models: List[nn.Module]) -> List[torch.Tensor]:
        """Execute expert computation with pipeline parallelism"""
        results = []
        
        # Stage 1: Routing (synchronous)
        routing_results = self._execute_routing_stage(input_data)
        
        # Stage 2: Expert computation (asynchronous)
        expert_tasks = []
        for i, expert_model in enumerate(expert_models):
            task = self.async_expert_computation(i, input_data, expert_model)
            expert_tasks.append(task)
        
        # Stage 3: Communication (asynchronous)
        communication_tasks = []
        for i, task in enumerate(expert_tasks):
            expert_result = await task
            comm_task = self.async_communication(i, expert_result, list(range(self.world_size)))
            communication_tasks.append(comm_task)
        
        # Stage 4: Aggregation (synchronous)
        for task in communication_tasks:
            result = await task
            results.append(result)
        
        return results
    
    def _execute_routing_stage(self, input_data: torch.Tensor) -> torch.Tensor:
        """Execute routing stage synchronously"""
        # This would contain the actual routing logic
        return input_data


class CommunicationAwarePlacer:
    """
    Communication-Aware Expert Placement for solving congested all-to-all communication.
    Implements topology-aware routing and expert placement.
    """
    
    def __init__(self, num_experts: int, world_size: int, 
                 network_topology: Optional[Dict] = None):
        self.num_experts = num_experts
        self.world_size = world_size
        self.network_topology = network_topology or self._default_topology()
        
        # Expert placement mapping
        self.expert_placement = {}
        self.communication_costs = {}
        
        # Network topology analysis
        self.rank_distances = self._compute_rank_distances()
        self.communication_bandwidth = self._compute_bandwidth_matrix()
        
        # Communication optimization
        self.communication_schedule = {}
        self.batched_communications = defaultdict(list)
        
    def _default_topology(self) -> Dict:
        """Default network topology (fully connected)"""
        return {
            'type': 'fully_connected',
            'bandwidth': 1.0,
            'latency': 0.001
        }
    
    def _compute_rank_distances(self) -> torch.Tensor:
        """Compute distances between ranks based on topology"""
        distances = torch.zeros(self.world_size, self.world_size)
        
        if self.network_topology['type'] == 'fully_connected':
            # All ranks are equally distant
            distances.fill_(1.0)
            distances.fill_diagonal_(0.0)
        elif self.network_topology['type'] == 'ring':
            # Ring topology
            for i in range(self.world_size):
                for j in range(self.world_size):
                    distances[i, j] = min(abs(i - j), self.world_size - abs(i - j))
        elif self.network_topology['type'] == 'mesh':
            # 2D mesh topology
            mesh_size = int(math.sqrt(self.world_size))
            for i in range(self.world_size):
                for j in range(self.world_size):
                    i_pos = (i // mesh_size, i % mesh_size)
                    j_pos = (j // mesh_size, j % mesh_size)
                    distances[i, j] = abs(i_pos[0] - j_pos[0]) + abs(i_pos[1] - j_pos[1])
        
        return distances
    
    def _compute_bandwidth_matrix(self) -> torch.Tensor:
        """Compute bandwidth matrix based on topology"""
        base_bandwidth = self.network_topology.get('bandwidth', 1.0)
        latency = self.network_topology.get('latency', 0.001)
        
        # Bandwidth decreases with distance
        bandwidth = base_bandwidth / (1 + self.rank_distances * latency)
        bandwidth.fill_diagonal_(float('inf'))  # Same rank has infinite bandwidth
        
        return bandwidth
    
    def optimize_expert_placement(self, expert_communication_patterns: Dict[int, List[int]]):
        """Optimize expert placement based on communication patterns"""
        # This is a simplified placement algorithm
        # In practice, this would be a complex optimization problem
        
        experts_per_rank = self.num_experts // self.world_size
        
        for expert_id, communication_partners in expert_communication_patterns.items():
            # Find the rank that minimizes communication cost
            best_rank = 0
            min_cost = float('inf')
            
            for rank in range(self.world_size):
                cost = 0
                for partner in communication_partners:
                    if partner in self.expert_placement:
                        partner_rank = self.expert_placement[partner]
                        cost += self.rank_distances[rank, partner_rank]
                
                if cost < min_cost:
                    min_cost = cost
                    best_rank = rank
            
            self.expert_placement[expert_id] = best_rank
    
    def schedule_communication(self, expert_outputs: Dict[int, torch.Tensor]) -> Dict:
        """Schedule communication to minimize congestion"""
        # Group communications by target ranks
        communication_groups = defaultdict(list)
        
        for expert_id, output in expert_outputs.items():
            expert_rank = self.expert_placement.get(expert_id, 0)
            
            # Determine which ranks need this expert's output
            target_ranks = self._get_target_ranks(expert_id, expert_rank)
            
            for target_rank in target_ranks:
                communication_groups[target_rank].append({
                    'expert_id': expert_id,
                    'data': output,
                    'source_rank': expert_rank
                })
        
        # Schedule batched communications
        communication_schedule = {}
        for target_rank, communications in communication_groups.items():
            # Batch communications to the same target
            batched_data = torch.cat([comm['data'] for comm in communications], dim=0)
            communication_schedule[target_rank] = {
                'data': batched_data,
                'expert_ids': [comm['expert_id'] for comm in communications]
            }
        
        return communication_schedule
    
    def _get_target_ranks(self, expert_id: int, source_rank: int) -> List[int]:
        """Get ranks that need this expert's output"""
        # This would be determined by the routing strategy
        # For now, return all ranks except the source
        return [rank for rank in range(self.world_size) if rank != source_rank]
    
    def compute_communication_cost(self, expert_id: int, target_rank: int) -> float:
        """Compute communication cost for expert output"""
        source_rank = self.expert_placement.get(expert_id, 0)
        distance = self.rank_distances[source_rank, target_rank]
        bandwidth = self.communication_bandwidth[source_rank, target_rank]
        
        # Cost is inversely proportional to bandwidth and proportional to distance
        cost = distance / bandwidth
        return cost


class EnhancedPiKVMoE(nn.Module):
    """
    Enhanced PiKV MoE implementation that addresses the three key issues:
    1. Dynamic load imbalance - Dynamic expert selection
    2. Inefficient synchronous execution - Async execution mode
    3. Congested all-to-all communication - Communication-aware placement
    """
    
    def __init__(self, rank=4, alpha=1.0, use_distillation=False, teacher_hidden_size=None,
                 use_cache_scheduling=False, cache_scheduling_policy=SchedulingPolicy.NONE,
                 enable_dynamic_balancing=True, enable_async_execution=True, 
                 enable_communication_optimization=True, enable_smartmoe=True, world_size=1):
        super(EnhancedPiKVMoE, self).__init__()
        
        self.use_distillation = use_distillation
        self.use_cache_scheduling = use_cache_scheduling
        self.cache_scheduling_policy = cache_scheduling_policy
        self.world_size = world_size
        
        # Enable optional features (private attrs avoid method/attribute collisions)
        self._enable_dynamic_balancing = enable_dynamic_balancing
        self._enable_async_execution = enable_async_execution
        self._enable_communication_optimization = enable_communication_optimization
        self.enable_smartmoe = enable_smartmoe
        
        # Add embedding layer
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        
        # Initialize experts
        self.experts = nn.ModuleList([
            LoRAExpert(config['hidden_size'], rank=rank, alpha=alpha)
            for _ in range(config['num_experts'])
        ])
        
        # Initialize enhanced router with dynamic load balancing
        if self._enable_dynamic_balancing:
            self.router = AdaptiveRouter(
                hidden_size=config['hidden_size'],
                num_experts=config['num_experts'],
                top_k=2,
                temperature=1.0
            )
            self.load_balancer = DynamicLoadBalancer(
                num_experts=config['num_experts'],
                hidden_size=config['hidden_size']
            )
        else:
            self.router = AdaptiveRouter(
                hidden_size=config['hidden_size'],
                num_experts=config['num_experts'],
                top_k=2,
                temperature=1.0
            )
        
        # Initialize async execution manager
        if self._enable_async_execution:
            self.async_manager = AsyncExecutionManager(
                num_experts=config['num_experts'],
                world_size=world_size
            )
        
        # Initialize communication-aware placer
        if self._enable_communication_optimization:
            self.communication_placer = CommunicationAwarePlacer(
                num_experts=config['num_experts'],
                world_size=world_size
            )
        
        # Initialize SmartMoE
        if self.enable_smartmoe:
            self.smartmoe = create_smartmoe(
                hidden_size=config['hidden_size'],
                num_experts=config['num_experts'],
                world_size=world_size,
                enable_offline_optimization=True,
                enable_online_adaptation=True
            )
        
        # Query-aware KV cache selection with LoRA
        self.query_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.key_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.query_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
        self.key_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
        
        # Cache size allocation (one cache per expert)
        self.cache_sizes = self.pyramidal_cache_allocation()
        while len(self.cache_sizes) < config['num_experts']:
            self.cache_sizes.append(self.cache_sizes[-1] if self.cache_sizes else config['kv_cache_size'])
        self.cache_sizes = self.cache_sizes[:config['num_experts']]
        
        # Initialize KV caches
        self.kv_caches = nn.ModuleList([
            KVCache(size, use_scheduling=use_cache_scheduling, 
                   scheduling_policy=cache_scheduling_policy,
                   hidden_size=config['hidden_size']) 
            for size in self.cache_sizes
        ])
        
        # Cache pointers
        self.register_buffer('cache_ptrs', torch.zeros(config['num_experts'], dtype=torch.long))
        
        # Projection to vocabulary size
        self.vocab_proj = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Knowledge Distillation Setup
        if use_distillation:
            self.teacher_hidden_size = teacher_hidden_size or config['hidden_size'] * 2
            self.distillation_module = PiKVDistillation(
                student_hidden_size=config['hidden_size'],
                teacher_hidden_size=self.teacher_hidden_size,
                num_experts=config['num_experts'],
                temperature=4.0,
                expert_distill_weight=0.4,
                cache_distill_weight=0.3
            )
            
            self.teacher_model = create_teacher_model(
                hidden_size=self.teacher_hidden_size,
                num_experts=config['num_experts']
            )
            
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            print(f"Knowledge Distillation enabled with teacher hidden size: {self.teacher_hidden_size}")
        
        print(f"Enhanced PiKV MoE initialized with:")
        print(f"  - Dynamic Load Balancing: {self._enable_dynamic_balancing}")
        print(f"  - Async Execution: {self._enable_async_execution}")
        print(f"  - Communication Optimization: {self._enable_communication_optimization}")
        print(f"  - SmartMoE Integration: {self.enable_smartmoe}")

        # Callable flag so both `model.enable_communication_optimization`
        # (truthiness) and `model.enable_communication_optimization()` work.
        self.enable_communication_optimization = _ToggleFlag(
            self._enable_communication_optimization,
            on_enable=self._ensure_communication_placer,
        )
    
    @property
    def enable_dynamic_balancing(self):
        return self._enable_dynamic_balancing
    
    @enable_dynamic_balancing.setter
    def enable_dynamic_balancing(self, value):
        self._enable_dynamic_balancing = bool(value)
    
    @property
    def enable_async_execution(self):
        return self._enable_async_execution
    
    @enable_async_execution.setter
    def enable_async_execution(self, value):
        self._enable_async_execution = bool(value)

    def _ensure_communication_placer(self):
        self._enable_communication_optimization = True
        if not hasattr(self, 'communication_placer'):
            self.communication_placer = CommunicationAwarePlacer(
                num_experts=config['num_experts'],
                world_size=self.world_size
            )
        print("Communication optimization enabled")

    def pyramidal_cache_allocation(self):
        """Calculate cache sizes using pyramidal allocation"""
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]
    
    def compute_token_importance(self, query, key):
        """Compute token importance scores"""
        query_base = self.query_proj(query)
        key_base = self.key_proj(key)
        
        query_lora = self.query_lora(query)
        key_lora = self.key_lora(key)
        
        query = query_base + query_lora
        key = key_base + key_lora
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config['hidden_size'])
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        importance = attention_probs.sum(dim=1)
        return importance
    
    def update_cache(self, expert_idx, key, value, importance):
        """Update KV cache for specific expert"""
        cache = self.kv_caches[expert_idx]
        ptr = self.cache_ptrs[expert_idx]
        
        cache.update(ptr, key, value, importance)
        
        if not self.use_cache_scheduling:
            self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x, query=None, return_loss=False, targets=None, use_teacher=False):
        """Enhanced forward pass with optional optimizations"""
        # Handle input format
        if len(x.shape) == 2:
            x = self.embedding(x)
        elif len(x.shape) == 3:
            pass
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape: {x.shape}")
        
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        # Calculate routing weights
        if self.enable_dynamic_balancing:
            # Use dynamic load balancing
            routing_weights, expert_indices, top_k_weights, lb_loss, importance = self.router(x)
            
            # Apply load balancing adjustments
            router_logits = self.router.router(x)
            adjusted_logits = self.load_balancer.adaptive_routing(router_logits, x)
            routing_probs = F.softmax(adjusted_logits, dim=-1)
            
            # Update load balancer
            for i in range(config['num_experts']):
                expert_load = (expert_indices == i).sum().float()
                self.load_balancer.update_load(i, expert_load.item())
            
            # Get enhanced load balancing loss
            enhanced_lb_loss = self.load_balancer.get_load_balance_loss()
            lb_loss = lb_loss + enhanced_lb_loss
        else:
            routing_weights, expert_indices, top_k_weights, lb_loss, importance = self.router(x)
        
        # Compute token importance
        if query is not None:
            if query.dtype != torch.float32:
                query = query.to(torch.float32)
            importance = self.compute_token_importance(query, x)
        
        # Initialize output
        batch_size, seq_len, hidden_size = x.shape
        expert_output = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        
        # Process experts
        if self.enable_async_execution:
            # Use async execution
            expert_outputs = self._async_expert_processing(x, routing_weights, importance)
        else:
            # Use synchronous execution
            expert_outputs = self._sync_expert_processing(x, routing_weights, importance)
        
        # Aggregate expert outputs
        for i, expert_output_i in enumerate(expert_outputs):
            expert_output += expert_output_i * routing_weights[:, :, i].unsqueeze(-1)
        
        # Project to vocabulary size
        logits = self.vocab_proj(expert_output)
        
        # Knowledge Distillation
        distill_loss = torch.tensor(0.0, device=x.device)
        if self.use_distillation and self.training and use_teacher:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(x)
            
            distill_loss, distill_loss_dict = self.distillation_module(
                student_logits=logits,
                teacher_logits=teacher_outputs['logits'],
                student_features=expert_output,
                teacher_features=teacher_outputs['features'],
                student_expert_outputs=expert_outputs,
                teacher_expert_outputs=teacher_outputs.get('expert_outputs'),
                teacher_routing_weights=teacher_outputs.get('routing_weights'),
                targets=targets
            )
        
        if return_loss:
            total_loss = lb_loss + distill_loss
            return logits, total_loss
        return logits
    
    def _sync_expert_processing(self, x, routing_weights, importance):
        """Synchronous expert processing"""
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            expert_output_i = expert(x)
            expert_outputs.append(expert_output_i)
            
            cached_values = self.kv_caches[i].get_all()
            if cached_values is not None:
                expert_output_i = expert_output_i + cached_values.unsqueeze(0).unsqueeze(0)
            
            self.update_cache(i, x.detach(), expert_output_i.detach(), importance.detach())
        
        return expert_outputs
    
    def _async_expert_processing(self, x, routing_weights, importance):
        """Asynchronous expert processing"""
        # This would implement the actual async processing
        # For now, fall back to sync processing
        return self._sync_expert_processing(x, routing_weights, importance)
    
    def enable_dynamic_load_balancing(self):
        """Enable dynamic load balancing"""
        self.enable_dynamic_balancing = True
        if not hasattr(self, 'load_balancer'):
            self.load_balancer = DynamicLoadBalancer(
                num_experts=config['num_experts'],
                hidden_size=config['hidden_size']
            )
        print("Dynamic load balancing enabled")
    
    def disable_dynamic_load_balancing(self):
        """Disable dynamic load balancing"""
        self.enable_dynamic_balancing = False
        print("Dynamic load balancing disabled")
    
    def enable_async_execution_mode(self):
        """Enable async execution mode"""
        self.enable_async_execution = True
        if not hasattr(self, 'async_manager'):
            self.async_manager = AsyncExecutionManager(
                num_experts=config['num_experts'],
                world_size=self.world_size
            )
        print("Async execution mode enabled")
    
    def disable_async_execution_mode(self):
        """Disable async execution mode"""
        self.enable_async_execution = False
        print("Async execution mode disabled")
    
    def disable_communication_optimization(self):
        """Disable communication optimization"""
        self._enable_communication_optimization = False
        if isinstance(self.enable_communication_optimization, _ToggleFlag):
            self.enable_communication_optimization.set(False)
        print("Communication optimization disabled")
    
    def get_performance_metrics(self):
        """Get performance metrics for all optimizations"""
        metrics = {}
        
        if self.enable_dynamic_balancing:
            metrics['load_balancing'] = {
                'expert_loads': self.load_balancer.expert_loads.tolist(),
                'expert_utilization': self.load_balancer.expert_utilization.tolist(),
                'load_imbalance': self.load_balancer.get_load_balance_loss().item()
            }
        
        if self.enable_communication_optimization:
            metrics['communication'] = {
                'expert_placement': dict(self.communication_placer.expert_placement),
                'communication_costs': dict(self.communication_placer.communication_costs)
            }
        
        if self.enable_smartmoe:
            smartmoe_metrics = self.smartmoe.get_performance_metrics()
            metrics['smartmoe'] = smartmoe_metrics
        
        return metrics


# Factory function for creating enhanced MoE models
def create_enhanced_pikv_moe(
    rank=4, 
    alpha=1.0, 
    use_distillation=False, 
    teacher_hidden_size=None,
    use_cache_scheduling=False, 
    cache_scheduling_policy=SchedulingPolicy.NONE,
    enable_dynamic_balancing=True,
    enable_async_execution=True,
    enable_communication_optimization=True,
    enable_smartmoe=True,
    world_size=1,
    **kwargs
) -> EnhancedPiKVMoE:
    """
    Factory function to create enhanced PiKV MoE models with optional optimizations.
    
    Args:
        rank: LoRA rank
        alpha: LoRA alpha
        use_distillation: Enable knowledge distillation
        teacher_hidden_size: Teacher model hidden size
        use_cache_scheduling: Enable cache scheduling
        cache_scheduling_policy: Cache scheduling policy
        enable_dynamic_balancing: Enable dynamic load balancing
        enable_async_execution: Enable async execution mode
        enable_communication_optimization: Enable communication optimization
        enable_smartmoe: Enable SmartMoE integration
        world_size: Number of distributed processes
        **kwargs: Accepted for README compatibility (e.g. load_balancing_strategy,
                  execution_mode, communication_strategy, network_topology)
    
    Returns:
        EnhancedPiKVMoE instance
    """
    # Map documented strategy kwargs onto enable flags when provided
    if 'load_balancing_strategy' in kwargs:
        strategy = kwargs['load_balancing_strategy']
        if strategy in (None, 'none', 'None'):
            enable_dynamic_balancing = False
    if 'execution_mode' in kwargs:
        mode = kwargs['execution_mode']
        enable_async_execution = mode not in (None, 'sync', 'Sync')
    if 'communication_strategy' in kwargs:
        strategy = kwargs['communication_strategy']
        if strategy in (None, 'none', 'None'):
            enable_communication_optimization = False

    return EnhancedPiKVMoE(
        rank=rank,
        alpha=alpha,
        use_distillation=use_distillation,
        teacher_hidden_size=teacher_hidden_size,
        use_cache_scheduling=use_cache_scheduling,
        cache_scheduling_policy=cache_scheduling_policy,
        enable_dynamic_balancing=enable_dynamic_balancing,
        enable_async_execution=enable_async_execution,
        enable_communication_optimization=enable_communication_optimization,
        enable_smartmoe=enable_smartmoe,
        world_size=world_size
    )
