import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import logging
import time
import os
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager
from .distributed_config import distributed_config as dconfig
from .config import config
from ..single.lora import LoRALayer, LoRAExpert, LoRAKVCache
from ..single.kv_cache_compression import KVCacheCompressor
from ..single.routing_strategy import AdaptiveRouter
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def distributed_context(backend: str = 'nccl', timeout: int = 1800):
    """Context manager for distributed training with proper cleanup"""
    try:
        if not dist.is_initialized():
            # Get environment variables from torchrun
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Initialize process group
            dist.init_process_group(
                backend=backend,
                init_method=dconfig['dist_url'],
                world_size=world_size,
                rank=rank,
                timeout=torch.distributed.constants.default_pg_timeout
            )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
        
        yield
        
    except Exception as e:
        logger.error(f"Distributed context error: {e}")
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")


class DistributedPerformanceMonitor:
    """Monitor distributed training performance"""
    
    def __init__(self):
        self.metrics = {
            'forward_time': [],
            'backward_time': [],
            'communication_time': [],
            'memory_usage': [],
            'throughput': []
        }
        self.start_time = None
    
    def start_timing(self):
        """Start timing a training step"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def end_timing(self, step_type: str = 'forward'):
        """End timing and record metrics"""
        if self.start_time is None:
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - self.start_time
        self.metrics[f'{step_type}_time'].append(elapsed)
        
        # Record memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.metrics['memory_usage'].append(memory_used)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[f'{metric}_mean'] = sum(values) / len(values)
                stats[f'{metric}_max'] = max(values)
                stats[f'{metric}_min'] = min(values)
                stats[f'{metric}_total'] = sum(values)
        
        return stats
    
    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics:
            self.metrics[metric] = []

class DistributedExpert(nn.Module):
    """Enhanced distributed expert with error handling and performance monitoring"""
    
    def __init__(self, expert_id: int, world_size: int, rank: int = 4, alpha: float = 1.0):
        super(DistributedExpert, self).__init__()
        self.expert_id = expert_id
        self.world_size = world_size
        self.rank = rank
        self.alpha = alpha
        
        try:
            self.expert = LoRAExpert(config['hidden_size'], rank=rank, alpha=alpha)
            self.register_buffer('expert_utilization', torch.tensor(0.0))
            self.register_buffer('forward_count', torch.tensor(0))
            logger.info(f"Initialized DistributedExpert {expert_id} on rank {rank}")
        except Exception as e:
            logger.error(f"Failed to initialize DistributedExpert {expert_id}: {e}")
            raise
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with performance monitoring"""
        try:
            start_time = time.time()
            output = self.expert(x)
            
            # Update statistics
            self.forward_count += 1
            self.expert_utilization += time.time() - start_time
            
            return output
        except Exception as e:
            logger.error(f"Forward pass failed for expert {self.expert_id}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get expert performance statistics"""
        return {
            'expert_id': self.expert_id,
            'forward_count': self.forward_count.item(),
            'avg_forward_time': self.expert_utilization.item() / max(self.forward_count.item(), 1),
            'utilization': self.expert_utilization.item()
        }

class DistributedKVCache(nn.Module):
    """Enhanced distributed KV cache with error handling and performance monitoring"""
    
    def __init__(self, size: int, expert_id: int, world_size: int):
        super(DistributedKVCache, self).__init__()
        self.size = size
        self.hidden_size = config['hidden_size']
        self.expert_id = expert_id
        self.world_size = world_size
        
        try:
            # Initialize tensors
            self.register_buffer('keys', torch.zeros(size, self.hidden_size))
            self.register_buffer('values', torch.zeros(size, self.hidden_size))
            self.register_buffer('importance', torch.zeros(size))
            self.register_buffer('hit_count', torch.tensor(0))
            self.register_buffer('miss_count', torch.tensor(0))
            self.register_buffer('cache_usage', torch.zeros(size))
            
            # Initialize compressor
            self.compressor = KVCacheCompressor(
                hidden_size=self.hidden_size,
                compression_type='pyramid',
                compression_ratio=0.5
            )
            
            # Initialize LoRA for cache values
            self.value_lora = LoRALayer(
                self.hidden_size,
                self.hidden_size,
                rank=4,
                alpha=1.0
            )
            
            logger.info(f"Initialized DistributedKVCache for expert {expert_id} with size {size}")
        except Exception as e:
            logger.error(f"Failed to initialize DistributedKVCache for expert {expert_id}: {e}")
            raise
    
    def update(self, idx: int, key: torch.Tensor, value: torch.Tensor, importance: torch.Tensor):
        """Update cache with error handling and performance monitoring"""
        try:
            # Validate inputs
            if idx >= self.size or idx < 0:
                raise ValueError(f"Index {idx} out of range [0, {self.size})")
            
            # Reshape input if needed
            if len(key.shape) == 3:  # [batch_size, seq_len, hidden_size]
                key = key.mean(dim=0).mean(dim=0)  # [hidden_size]
                value = value.mean(dim=0).mean(dim=0)  # [hidden_size]
            elif len(key.shape) == 2:  # [batch_size, hidden_size]
                key = key.mean(dim=0)  # [hidden_size]
                value = value.mean(dim=0)  # [hidden_size]
            elif len(key.shape) == 1:  # [hidden_size]
                pass  # Already in correct shape
            else:
                raise ValueError(f"Invalid key shape: {key.shape}")
            
            # Update cache at the specified index
            self.keys[idx] = key.detach()
            self.values[idx] = value.detach()
            self.importance[idx] = importance.mean().item() if importance is not None else 1.0
            self.cache_usage[idx] = 1.0  # Mark as used
            
        except Exception as e:
            logger.error(f"Failed to update cache at index {idx}: {e}")
            raise
    
    def get_all(self) -> torch.Tensor:
        """Get all cached values with compression and performance tracking"""
        try:
            # Check if cache has any data
            if self.cache_usage.sum() == 0:
                self.miss_count += 1
                return torch.zeros(self.hidden_size, device=self.keys.device)
            
            self.hit_count += 1
            
            # Apply compression to cached values
            compressed_keys, compressed_values = self.compressor(
                self.keys.unsqueeze(0),
                self.values.unsqueeze(0),
                self.importance.unsqueeze(0)
            )
            
            # Apply LoRA to compressed values
            compressed_values = compressed_values + self.value_lora(compressed_values)
            
            return compressed_values.squeeze(0).mean(dim=0)  # Return average of compressed values
            
        except Exception as e:
            logger.error(f"Failed to get cached values: {e}")
            self.miss_count += 1
            return torch.zeros(self.hidden_size, device=self.keys.device)
    
    def set_all(self, data: torch.Tensor):
        """Set all cache values with error handling"""
        try:
            if data is not None:
                if data.shape[-1] != self.hidden_size:
                    raise ValueError(f"Data shape {data.shape} incompatible with hidden_size {self.hidden_size}")
                
                self.values.copy_(data.unsqueeze(0).expand(self.size, -1))
                self.cache_usage.fill_(1.0)  # Mark all as used
        except Exception as e:
            logger.error(f"Failed to set cache values: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count.item() + self.miss_count.item()
        hit_rate = self.hit_count.item() / max(total_requests, 1)
        
        return {
            'expert_id': self.expert_id,
            'cache_size': self.size,
            'hit_count': self.hit_count.item(),
            'miss_count': self.miss_count.item(),
            'hit_rate': hit_rate,
            'usage_rate': self.cache_usage.mean().item(),
            'total_requests': total_requests
        }

class DistributedPiKVMoE(nn.Module):
    """Enhanced distributed PiKV MoE with comprehensive error handling and performance monitoring"""
    
    def __init__(self, rank: int = 4, alpha: float = 1.0):
        super(DistributedPiKVMoE, self).__init__()
        
        try:
            self.world_size = dconfig['world_size']
            self.rank = dconfig['rank']
            self.alpha = alpha
            
            # Validate configuration
            if config['num_experts'] % self.world_size != 0:
                raise ValueError(f"num_experts ({config['num_experts']}) must be divisible by world_size ({self.world_size})")
            
            # Expert parallel: each GPU handles a subset of experts
            experts_per_gpu = config['num_experts'] // self.world_size
            self.experts_per_gpu = experts_per_gpu
            
            # Initialize local experts with error handling
            self.local_experts = nn.ModuleList()
            for i in range(experts_per_gpu):
                expert_id = i + self.rank * experts_per_gpu
                try:
                    expert = DistributedExpert(expert_id, self.world_size, rank=rank, alpha=alpha)
                    self.local_experts.append(expert)
                except Exception as e:
                    logger.error(f"Failed to create expert {expert_id}: {e}")
                    raise
            
            # Initialize adaptive router with LoRA
            self.router = AdaptiveRouter(
                hidden_size=config['hidden_size'],
                num_experts=config['num_experts'],
                top_k=2,
                temperature=1.0
            )
            
            # Query-aware KV cache selection with LoRA
            self.query_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
            self.key_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
            self.query_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
            self.key_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
            
            # Cache size allocation
            self.cache_sizes = self.pyramidal_cache_allocation()
            
            # Initialize KV caches for each local expert
            self.kv_caches = nn.ModuleList()
            for i, size in enumerate(self.cache_sizes[:experts_per_gpu]):
                expert_id = i + self.rank * experts_per_gpu
                try:
                    cache = DistributedKVCache(size, expert_id, self.world_size)
                    self.kv_caches.append(cache)
                except Exception as e:
                    logger.error(f"Failed to create KV cache for expert {expert_id}: {e}")
                    raise
            
            # Cache pointers for each expert
            self.register_buffer('cache_ptrs', torch.zeros(experts_per_gpu, dtype=torch.long))
            
            # Performance monitoring
            self.performance_monitor = DistributedPerformanceMonitor()
            self.register_buffer('forward_count', torch.tensor(0))
            self.register_buffer('total_loss', torch.tensor(0.0))
            
            # Mixed precision training
            self.use_mixed_precision = dconfig['use_mixed_precision']
            
            logger.info(f"Initialized DistributedPiKVMoE with {experts_per_gpu} experts per GPU on rank {self.rank}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DistributedPiKVMoE: {e}")
            raise
        
    def pyramidal_cache_allocation(self):
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]
    
    def compute_token_importance(self, query, key):
        # Project query and key with LoRA
        query_base = self.query_proj(query)
        key_base = self.key_proj(key)
        
        query_lora = self.query_lora(query)
        key_lora = self.key_lora(key)
        
        query = query_base + query_lora
        key = key_base + key_lora
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config['hidden_size'])
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Compute importance as sum of attention weights
        importance = attention_probs.sum(dim=1)  # [batch_size, seq_len]
        
        return importance
    
    def update_cache(self, expert_idx, key, value, importance):
        cache = self.kv_caches[expert_idx]
        ptr = self.cache_ptrs[expert_idx]
        
        # Update cache with new key-value pair
        cache.update(ptr, key, value, importance)
        
        # Update pointer
        self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x: torch.Tensor, query: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with comprehensive error handling and performance monitoring"""
        try:
            self.performance_monitor.start_timing()
            
            # Validate input
            if x.dim() != 3:
                raise ValueError(f"Expected input with 3 dimensions, got {x.dim()}")
            
            # Calculate routing weights using adaptive router
            routing_weights, expert_indices, top_k_weights, lb_loss, importance = self.router(x)
            
            # If query is provided, compute token importance
            if query is not None:
                importance = self.compute_token_importance(query, x)
            
            # Initialize output tensor
            expert_output = torch.zeros_like(x)
            
            # Process each local expert
            for i, expert in enumerate(self.local_experts):
                try:
                    # Get expert output with LoRA
                    expert_output_i = expert(x)
                    
                    # Get cached values
                    cached_values = self.kv_caches[i].get_all()
                    
                    # Combine with cached values
                    if cached_values is not None and cached_values.norm() > 0:
                        expert_output_i = expert_output_i + cached_values.detach()
                    
                    # Update cache with new values
                    self.update_cache(i, x.detach(), expert_output_i.detach(), importance.detach())
                    
                    # Add to final output weighted by routing probabilities
                    local_expert_id = i + self.rank * self.experts_per_gpu
                    if local_expert_id < routing_weights.size(-1):
                        expert_output += expert_output_i * routing_weights[:, :, local_expert_id].unsqueeze(-1)
                    
                except Exception as e:
                    logger.error(f"Error processing expert {i}: {e}")
                    # Continue with other experts
                    continue
            
            # Synchronize expert outputs across GPUs
            if dconfig.get('expert_parallel', True) and dist.is_initialized():
                dist.all_reduce(expert_output, op=dist.ReduceOp.SUM)
            
            # Update performance metrics
            self.performance_monitor.end_timing('forward')
            self.forward_count += 1
            self.total_loss += lb_loss.detach()
            
            return expert_output, lb_loss
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            # Return zero output and loss to prevent training crash
            return torch.zeros_like(x), torch.tensor(0.0, device=x.device)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'forward_count': self.forward_count.item(),
            'total_loss': self.total_loss.item(),
            'average_loss': self.total_loss.item() / max(self.forward_count.item(), 1),
            'rank': self.rank,
            'world_size': self.world_size,
            'experts_per_gpu': self.experts_per_gpu
        }
        
        # Add performance monitor stats
        metrics.update(self.performance_monitor.get_stats())
        
        # Add expert statistics
        expert_stats = []
        for i, expert in enumerate(self.local_experts):
            expert_stats.append(expert.get_stats())
        metrics['expert_stats'] = expert_stats
        
        # Add cache statistics
        cache_stats = []
        for i, cache in enumerate(self.kv_caches):
            cache_stats.append(cache.get_stats())
        metrics['cache_stats'] = cache_stats
        
        return metrics
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.forward_count.zero_()
        self.total_loss.zero_()
        self.performance_monitor.reset()
        
        # Reset expert and cache metrics
        for expert in self.local_experts:
            expert.forward_count.zero_()
            expert.expert_utilization.zero_()
        
        for cache in self.kv_caches:
            cache.hit_count.zero_()
            cache.miss_count.zero_()
            cache.cache_usage.zero_()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if not torch.cuda.is_available():
            return {'gpu_memory': 0.0, 'cpu_memory': 0.0}
        
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        cpu_memory = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return {
            'gpu_memory_allocated': gpu_memory,
            'gpu_memory_reserved': cpu_memory,
            'gpu_memory_available': torch.cuda.get_device_properties(0).total_memory / 1024**3 - gpu_memory
        }

class DistributedPiKVManager:
    """Enhanced distributed PiKV manager with comprehensive error handling and monitoring"""
    
    def __init__(self, rank: int = 4, alpha: float = 1.0):
        try:
            self.world_size = dconfig['world_size']
            self.rank = dconfig['rank']
            self.device = dconfig['device']
            self.alpha = alpha
            
            # Initialize distributed environment with error handling
            self._init_distributed()
            
            # Create model with error handling
            self.model = DistributedPiKVMoE(rank=rank, alpha=alpha).to(self.device)
            
            # Wrap model with DDP if using expert parallel
            if dconfig.get('expert_parallel', True):
                self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            
            # Optimizer with error handling
            try:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=config['learning_rate'],
                    weight_decay=0.01,
                    betas=(0.9, 0.999)
                )
            except Exception as e:
                logger.error(f"Failed to create optimizer: {e}")
                # Fallback to basic Adam
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
            
            # Mixed precision training
            self.scaler = GradScaler() if self.model.use_mixed_precision else None
            
            # Training statistics
            self.training_stats = {
                'total_steps': 0,
                'total_loss': 0.0,
                'peak_memory': 0.0,
                'start_time': time.time()
            }
            
            logger.info(f"Initialized DistributedPiKVManager on rank {self.rank}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DistributedPiKVManager: {e}")
            raise
    
    def _init_distributed(self):
        """Initialize distributed environment with proper error handling"""
        try:
            if not dist.is_initialized():
                # Get environment variables from torchrun
                rank = int(os.environ.get('RANK', self.rank))
                world_size = int(os.environ.get('WORLD_SIZE', self.world_size))
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                
                # Initialize process group
                dist.init_process_group(
                    backend=dconfig['dist_backend'],
                    init_method=dconfig['dist_url'],
                    world_size=world_size,
                    rank=rank,
                    timeout=torch.distributed.constants.default_pg_timeout
                )
                
                # Update rank and world_size from environment
                self.rank = rank
                self.world_size = world_size
                
                # Set device
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    self.device = torch.device(f'cuda:{local_rank}')
                
                logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
                
        except Exception as e:
            logger.error(f"Failed to initialize distributed environment: {e}")
            raise
        
    def train_step(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """Enhanced training step with comprehensive error handling and monitoring"""
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Validate inputs
            if data.shape != target.shape:
                raise ValueError(f"Data shape {data.shape} doesn't match target shape {target.shape}")
            
            # Use mixed precision training
            if dconfig.get('use_mixed_precision', False) and self.scaler is not None:
                with autocast():
                    output, lb_loss = self.model(data)
                    loss = F.mse_loss(output, target) + lb_loss
                
                # Scale loss and backward
                self.scaler.scale(loss).backward()
                
                # Unscale gradients and check for NaN/Inf
                self.scaler.unscale_(self.optimizer)
                if torch.isfinite(loss):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logger.warning("Loss is not finite, skipping optimizer step")
            else:
                output, lb_loss = self.model(data)
                loss = F.mse_loss(output, target) + lb_loss
                
                # Check for NaN/Inf before backward
                if torch.isfinite(loss):
                    loss.backward()
                    self.optimizer.step()
                else:
                    logger.warning("Loss is not finite, skipping backward pass")
            
            # Update training statistics
            loss_value = loss.item()
            self.training_stats['total_steps'] += 1
            self.training_stats['total_loss'] += loss_value
            
            # Update peak memory
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                self.training_stats['peak_memory'] = max(self.training_stats['peak_memory'], current_memory)
            
            return loss_value
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            # Return a high loss to indicate failure
            return float('inf')
    
    def save_checkpoint(self, path: str, tag: str = None):
        """Save checkpoint with comprehensive error handling"""
        try:
            if self.rank == 0:  # Only save checkpoint on main process
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                # Get model state dict (handle DDP wrapper)
                model_state_dict = self.model.state_dict()
                if hasattr(self.model, 'module'):
                    model_state_dict = self.model.module.state_dict()
                
                checkpoint = {
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'training_stats': self.training_stats,
                    'config': {
                        'rank': self.rank,
                        'world_size': self.world_size,
                        'alpha': self.alpha
                    },
                    'timestamp': time.time()
                }
                
                # Add model performance metrics
                if hasattr(self.model, 'get_performance_metrics'):
                    checkpoint['performance_metrics'] = self.model.get_performance_metrics()
                
                torch.save(checkpoint, path)
                logger.info(f"Checkpoint saved to {path}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, path: str):
        """Load checkpoint with comprehensive error handling"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state dict (handle DDP wrapper)
            model_state_dict = checkpoint['model_state_dict']
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scaler state
            if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training statistics
            if 'training_stats' in checkpoint:
                self.training_stats.update(checkpoint['training_stats'])
            
            logger.info(f"Checkpoint loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including model and training metrics"""
        stats = {
            'training_stats': self.training_stats.copy(),
            'memory_usage': self.get_memory_usage(),
            'rank': self.rank,
            'world_size': self.world_size,
            'device': str(self.device)
        }
        
        # Add model performance metrics
        if hasattr(self.model, 'get_performance_metrics'):
            stats['model_metrics'] = self.model.get_performance_metrics()
        
        # Add training time
        if 'start_time' in self.training_stats:
            stats['training_time'] = time.time() - self.training_stats['start_time']
        
        return stats
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if not torch.cuda.is_available():
            return {'gpu_memory': 0.0, 'cpu_memory': 0.0}
        
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return {
            'gpu_memory_allocated': gpu_memory,
            'gpu_memory_reserved': gpu_reserved,
            'gpu_memory_available': torch.cuda.get_device_properties(0).total_memory / 1024**3 - gpu_memory
        }
    
    def cleanup(self):
        """Cleanup resources and destroy process group"""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            logger.info("DistributedPiKVManager cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}") 