import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config
from .kv_cache_compression import KVCacheCompressor
from .routing_strategy import AdaptiveRouter
from .lora import LoRALayer, LoRAExpert, LoRAKVCache
from .distillation import PiKVDistillation, create_teacher_model, distillation_training_step
from .shared import ExternalMemoryCache
from .cache_scheduling import CacheSchedulingManager, SchedulingPolicy
import math

class KVCache(nn.Module):
    """
    KV Cache implementation with compression and streaming support.
    """
    def __init__(self, size, use_scheduling=False, scheduling_policy=SchedulingPolicy.NONE, hidden_size=None):
        super(KVCache, self).__init__()
        self.size = size
        self.hidden_size = hidden_size if hidden_size is not None else config['hidden_size']
        self.use_scheduling = use_scheduling
        
        # Initialize tensors
        self.register_buffer('keys', torch.zeros(size, self.hidden_size))
        self.register_buffer('values', torch.zeros(size, self.hidden_size))
        self.register_buffer('importance', torch.zeros(size))
        
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
        
        # Initialize cache scheduling manager if enabled
        if self.use_scheduling:
            self.scheduling_manager = CacheSchedulingManager(
                cache_size=size,
                hidden_size=self.hidden_size,
                policy=scheduling_policy
            )
        else:
            self.scheduling_manager = None
    
    def update(self, idx, key, value, importance):
        # Reshape input if needed
        if len(key.shape) == 3:  # [batch_size, seq_len, hidden_size]
            key = key.mean(dim=0).mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0).mean(dim=0)  # [hidden_size]
        elif len(key.shape) == 2:  # [seq_len, hidden_size]
            key = key.mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0)  # [hidden_size]
        
        # Use scheduling manager if enabled
        if self.use_scheduling and self.scheduling_manager is not None:
            # Prepare data for scheduling manager
            keys_batch = key.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            values_batch = value.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            # Update cache through scheduling manager
            metadata = {
                'importance': importance.unsqueeze(0) if importance.dim() == 0 else importance,
                'timestamp': torch.tensor([idx], device=key.device)
            }
            self.scheduling_manager.update_cache(keys_batch, values_batch, metadata)
        else:
            # Traditional cache update
            self.keys[idx] = key
            self.values[idx] = value
            self.importance[idx] = importance.mean().item()
    
    def get_all(self):
        if self.use_scheduling and self.scheduling_manager is not None:
            # Get cached values from scheduling manager
            current_size = self.scheduling_manager.cache_size_current.item()
            if current_size > 0:
                cached_keys = self.scheduling_manager.cache_keys[:current_size]
                cached_values = self.scheduling_manager.cache_values[:current_size]
                
                # Apply compression to cached values
                compressed_keys, compressed_values = self.compressor(
                    cached_keys.unsqueeze(0),
                    cached_values.unsqueeze(0),
                    torch.ones(1, current_size, device=cached_keys.device)
                )
                
                # Apply LoRA to compressed values
                compressed_values = compressed_values + self.value_lora(compressed_values)
                
                return compressed_values.squeeze(0).mean(dim=0)  # Return average of compressed values
            else:
                return torch.zeros(self.hidden_size, device=self.keys.device)
        else:
            # Traditional cache retrieval
            # Apply compression to cached values
            compressed_keys, compressed_values = self.compressor(
                self.keys.unsqueeze(0),
                self.values.unsqueeze(0),
                self.importance.unsqueeze(0)
            )
            
            # Apply LoRA to compressed values
            compressed_values = compressed_values + self.value_lora(compressed_values)
            
            return compressed_values.squeeze(0).mean(dim=0)  # Return average of compressed values
    
    def set_all(self, data):
        if data is not None:
            if self.use_scheduling and self.scheduling_manager is not None:
                # Reset scheduling manager cache
                self.scheduling_manager.reset_cache()
            else:
                self.values.copy_(data.unsqueeze(0).expand(self.size, -1))
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        if self.use_scheduling and self.scheduling_manager is not None:
            return self.scheduling_manager.get_cache_stats()
        else:
            return {
                'cache_size': self.size,
                'cache_utilization': 1.0,
                'policy': 'none'
            }
    
    def change_scheduling_policy(self, new_policy: SchedulingPolicy):
        """更改调度策略"""
        if self.use_scheduling and self.scheduling_manager is not None:
            self.scheduling_manager.change_policy(new_policy)

class PiKVMoE(nn.Module):
    def __init__(self, rank=4, alpha=1.0, use_distillation=False, teacher_hidden_size=None,
                 use_cache_scheduling=False, cache_scheduling_policy=SchedulingPolicy.NONE):
        super(PiKVMoE, self).__init__()
        self.use_distillation = use_distillation
        self.use_cache_scheduling = use_cache_scheduling
        self.cache_scheduling_policy = cache_scheduling_policy
        
        # Add embedding layer to handle token IDs
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        
        self.experts = nn.ModuleList([
            LoRAExpert(config['hidden_size'], rank=rank, alpha=alpha)
            for _ in range(config['num_experts'])
        ])
        
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
        
        # Cache size allocation for each layer
        self.cache_sizes = self.pyramidal_cache_allocation()
        
        # Initialize KV caches for each expert with optional scheduling
        self.kv_caches = nn.ModuleList([
            KVCache(size, use_scheduling=use_cache_scheduling, 
                   scheduling_policy=cache_scheduling_policy) 
            for size in self.cache_sizes
        ])
        
        # Cache pointers for each expert
        self.register_buffer('cache_ptrs', torch.zeros(config['num_experts'], dtype=torch.long))
        
        # Projection to vocabulary size
        self.vocab_proj = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Memory expansion flag
        self.use_memory_expansion = config.get('use_memory_expansion', False)
        if self.use_memory_expansion:
            self.external_cache = ExternalMemoryCache()
        
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
            
            # Create teacher model (can be loaded from checkpoint)
            self.teacher_model = create_teacher_model(
                hidden_size=self.teacher_hidden_size,
                num_experts=config['num_experts']
            )
            
            # Freeze teacher model parameters
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            print(f"Knowledge Distillation enabled with teacher hidden size: {self.teacher_hidden_size}")
        
        # Print cache scheduling info
        if self.use_cache_scheduling:
            print(f"Cache Scheduling enabled with policy: {cache_scheduling_policy.value}")

    def pyramidal_cache_allocation(self):
        """
        Calculate the cache size for each layer using the pyramidal allocation policy.
        """
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]
    
    def compute_token_importance(self, query, key):
        """
        Compute importance scores for tokens based on query-key attention.
        """
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
        """
        Update KV cache for a specific expert with new key-value pairs.
        """
        cache = self.kv_caches[expert_idx]
        ptr = self.cache_ptrs[expert_idx]
        
        # Update cache with new key-value pair
        cache.update(ptr, key, value, importance)
        
        # Update pointer (only for non-scheduling caches)
        if not self.use_cache_scheduling:
            self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x, query=None, return_loss=False, targets=None, use_teacher=False):
        # Handle both token IDs (2D) and embeddings (3D) as input
        if len(x.shape) == 2:  # Token IDs: [batch_size, seq_len]
            x = self.embedding(x)  # Convert to embeddings: [batch_size, seq_len, hidden_size]
        elif len(x.shape) == 3:  # Already embeddings: [batch_size, seq_len, hidden_size]
            pass
        else:
            raise ValueError(f"Input must be 2D (token IDs) or 3D (embeddings), got shape: {x.shape}")
        
        # Convert input to float if it's not already
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        # Calculate routing weights using adaptive router
        routing_weights, expert_indices, top_k_weights, lb_loss, importance = self.router(x)
        
        # If query is provided, compute token importance
        if query is not None:
            if query.dtype != torch.float32:
                query = query.to(torch.float32)
            importance = self.compute_token_importance(query, x)
        
        # Initialize output tensor
        batch_size, seq_len, hidden_size = x.shape
        expert_output = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        
        # Collect expert outputs for distillation
        expert_outputs_list = []
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get expert output with LoRA
            expert_output_i = expert(x)  # [batch_size, seq_len, hidden_size]
            expert_outputs_list.append(expert_output_i)
            
            # Get cached values
            cached_values = self.kv_caches[i].get_all()
            
            # Combine with cached values
            if cached_values is not None:
                expert_output_i = expert_output_i + cached_values.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            # Update cache with new values
            self.update_cache(i, x.detach(), expert_output_i.detach(), importance.detach())
            
            # Add to final output weighted by routing probabilities
            expert_output += expert_output_i * routing_weights[:, :, i].unsqueeze(-1)
        
        # Project to vocabulary size if needed
        logits = self.vocab_proj(expert_output)
        
        # Knowledge Distillation
        distill_loss = torch.tensor(0.0, device=x.device)
        if self.use_distillation and self.training and use_teacher:
            # Get teacher outputs
            with torch.no_grad():
                teacher_outputs = self.teacher_model(x)
            
            # Compute distillation loss
            distill_loss, distill_loss_dict = self.distillation_module(
                student_logits=logits,
                teacher_logits=teacher_outputs['logits'],
                student_features=expert_output,
                teacher_features=teacher_outputs['features'],
                student_expert_outputs=expert_outputs_list,
                teacher_expert_outputs=teacher_outputs.get('expert_outputs'),
                teacher_routing_weights=teacher_outputs.get('routing_weights'),
                targets=targets
            )
        
        if return_loss:
            total_loss = lb_loss + distill_loss
            return logits, total_loss
        return logits
    
    def enable_cache_scheduling(self, policy: SchedulingPolicy = SchedulingPolicy.LRU):
        """启用缓存调度"""
        self.use_cache_scheduling = True
        self.cache_scheduling_policy = policy
        
        # 为所有缓存启用调度
        for cache in self.kv_caches:
            # Use setattr to avoid PyTorch module parameter checking
            setattr(cache, 'use_scheduling', True)
            if cache.scheduling_manager is None:
                cache.scheduling_manager = CacheSchedulingManager(
                    cache_size=cache.size,
                    hidden_size=cache.hidden_size,
                    policy=policy
                )
            else:
                cache.scheduling_manager.change_policy(policy)
        
        print(f"Cache scheduling enabled with policy: {policy.value}")
    
    def disable_cache_scheduling(self):
        """禁用缓存调度"""
        self.use_cache_scheduling = False
        
        # 为所有缓存禁用调度
        for cache in self.kv_caches:
            # Use setattr to avoid PyTorch module parameter checking
            setattr(cache, 'use_scheduling', False)
            if cache.scheduling_manager is not None:
                cache.scheduling_manager.reset_cache()
        
        print("Cache scheduling disabled")
    
    def change_cache_scheduling_policy(self, new_policy: SchedulingPolicy):
        """更改缓存调度策略"""
        if self.use_cache_scheduling:
            self.cache_scheduling_policy = new_policy
            for cache in self.kv_caches:
                cache.change_scheduling_policy(new_policy)
            print(f"Cache scheduling policy changed to: {new_policy.value}")
        else:
            print("Cache scheduling is not enabled. Use enable_cache_scheduling() first.")
    
    def get_cache_stats(self):
        """获取所有缓存的统计信息"""
        stats = {}
        for i, cache in enumerate(self.kv_caches):
            stats[f'expert_{i}'] = cache.get_cache_stats()
        return stats
    
    def print_cache_stats(self):
        """打印缓存统计信息"""
        if self.use_cache_scheduling:
            print("\n===== Cache Scheduling Statistics =====")
            stats = self.get_cache_stats()
            for expert_name, expert_stats in stats.items():
                print(f"\n{expert_name.upper()}:")
                for stat_name, value in expert_stats.items():
                    if isinstance(value, float):
                        print(f"  {stat_name}: {value:.4f}")
                    else:
                        print(f"  {stat_name}: {value}")
            print("\n======================================")
        else:
            print("Cache scheduling is not enabled.")

    def enable_distillation(self, teacher_model_path=None):
        """
        Enable knowledge distillation and optionally load teacher model
        """
        if not self.use_distillation:
            print("Distillation not initialized. Please set use_distillation=True during initialization.")
            return
        
        if teacher_model_path:
            self.load_teacher_model(teacher_model_path)
        
        print("Knowledge distillation enabled")
    
    def disable_distillation(self):
        """
        Disable knowledge distillation for inference
        """
        self.use_distillation = False
        print("Knowledge distillation disabled")
    
    def load_teacher_model(self, model_path):
        """
        Load pre-trained teacher model
        """
        if not self.use_distillation:
            print("Distillation not initialized")
            return
        
        try:
            checkpoint = torch.load(model_path, map_location=next(self.parameters()).device)
            self.teacher_model.load_state_dict(checkpoint)
            print(f"Teacher model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load teacher model: {e}")
    
    def distillation_step(self, input_data, targets=None, optimizer=None):
        """
        Perform a single distillation training step
        """
        if not self.use_distillation:
            print("Distillation not enabled")
            return {}
        
        return distillation_training_step(
            student_model=self,
            teacher_model=self.teacher_model,
            distillation_module=self.distillation_module,
            input_data=input_data,
            targets=targets,
            optimizer=optimizer
        )
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint with KV caches and LoRA parameters.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'cache_ptrs': self.cache_ptrs,
            'kv_caches': [cache.get_all() for cache in self.kv_caches],
            'use_distillation': self.use_distillation,
            'use_cache_scheduling': self.use_cache_scheduling,
            'cache_scheduling_policy': self.cache_scheduling_policy.value if self.use_cache_scheduling else None
        }
        
        # Save teacher model if distillation is enabled
        if self.use_distillation:
            checkpoint['teacher_state_dict'] = self.teacher_model.state_dict()
            checkpoint['distillation_state_dict'] = self.distillation_module.state_dict()
        
        # Save cache scheduling stats if enabled
        if self.use_cache_scheduling:
            checkpoint['cache_stats'] = self.get_cache_stats()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint with KV caches and LoRA parameters.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.cache_ptrs.copy_(checkpoint['cache_ptrs'])
        for i, cache_data in enumerate(checkpoint['kv_caches']):
            if cache_data is not None:
                self.kv_caches[i].set_all(cache_data)
        
        # Load distillation components if available
        if checkpoint.get('use_distillation', False) and self.use_distillation:
            if 'teacher_state_dict' in checkpoint:
                self.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
            if 'distillation_state_dict' in checkpoint:
                self.distillation_module.load_state_dict(checkpoint['distillation_state_dict'])
        
        # Load cache scheduling settings if available
        if checkpoint.get('use_cache_scheduling', False):
            policy_name = checkpoint.get('cache_scheduling_policy', 'none')
            try:
                policy = SchedulingPolicy(policy_name)
                self.enable_cache_scheduling(policy)
            except ValueError:
                print(f"Unknown scheduling policy in checkpoint: {policy_name}")
