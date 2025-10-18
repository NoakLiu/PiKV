import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
import json
import os
from typing import Dict, Any, Optional, Union
from .distributed_config import distributed_config as dconfig
from .config import config
from ..single.moe import create_moe
from ..single.pikv_compression import create_compressor
from ..single.kvcache_centric_system import create_kvcache_centric_system


class PiKVDeepSpeedConfig:
    """DeepSpeed configuration for PiKV"""
    
    def __init__(self, 
                 zero_stage: int = 3,
                 offload_optimizer: bool = True,
                 offload_param: bool = True,
                 cpu_offload: bool = True,
                 pin_memory: bool = True,
                 overlap_comm: bool = True,
                 contiguous_gradients: bool = True,
                 sub_group_size: int = 1e9,
                 reduce_bucket_size: int = 5e8,
                 stage3_prefetch_bucket_size: int = 5e7,
                 stage3_param_persistence_threshold: int = 1e6,
                 stage3_max_live_parameters: int = 1e9,
                 stage3_max_reuse_distance: int = 1e9,
                 stage3_gather_16bit_weights_on_model_save: bool = True,
                 moe_expert_count: int = None,
                 moe_loss_coeff: float = 0.1,
                 moe_top_k: int = 2,
                 enable_moe: bool = False):
        
        self.zero_stage = zero_stage
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param
        self.cpu_offload = cpu_offload
        self.pin_memory = pin_memory
        self.overlap_comm = overlap_comm
        self.contiguous_gradients = contiguous_gradients
        self.sub_group_size = sub_group_size
        self.reduce_bucket_size = reduce_bucket_size
        self.stage3_prefetch_bucket_size = stage3_prefetch_bucket_size
        self.stage3_param_persistence_threshold = stage3_param_persistence_threshold
        self.stage3_max_live_parameters = stage3_max_live_parameters
        self.stage3_max_reuse_distance = stage3_max_reuse_distance
        self.stage3_gather_16bit_weights_on_model_save = stage3_gather_16bit_weights_on_model_save
        self.moe_expert_count = moe_expert_count or config['num_experts']
        self.moe_loss_coeff = moe_loss_coeff
        self.moe_top_k = moe_top_k
        self.enable_moe = enable_moe
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to DeepSpeed configuration dictionary"""
        ds_config = {
            "train_batch_size": dconfig.get('batch_size', 32),
            "gradient_accumulation_steps": dconfig.get('gradient_accumulation_steps', 1),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config['learning_rate'],
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": config['learning_rate'],
                    "warmup_num_steps": 100
                }
            },
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.cpu_offload else "none",
                    "pin_memory": self.pin_memory
                },
                "offload_param": {
                    "device": "cpu" if self.cpu_offload else "none",
                    "pin_memory": self.pin_memory
                },
                "overlap_comm": self.overlap_comm,
                "contiguous_gradients": self.contiguous_gradients,
                "sub_group_size": self.sub_group_size,
                "reduce_bucket_size": self.reduce_bucket_size,
                "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.stage3_param_persistence_threshold,
                "stage3_max_live_parameters": self.stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.stage3_max_reuse_distance,
                "stage3_gather_16bit_weights_on_model_save": self.stage3_gather_16bit_weights_on_model_save
            },
            "fp16": {
                "enabled": dconfig.get('use_mixed_precision', True),
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "wall_clock_breakdown": False,
            "memory_breakdown": False
        }
        
        # Add MoE configuration if enabled
        if self.enable_moe:
            ds_config["moe"] = {
                "enabled": True,
                "expert_count": self.moe_expert_count,
                "loss_coeff": self.moe_loss_coeff,
                "top_k": self.moe_top_k
            }
        
        return ds_config
    
    def save_config(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class PiKVDeepSpeedModel(nn.Module):
    """PiKV model optimized for DeepSpeed"""
    
    def __init__(self, 
                 model_type: str = 'pikv',
                 hidden_size: int = None,
                 num_experts: int = None,
                 top_k: int = 2,
                 enable_compression: bool = True,
                 enable_scheduling: bool = True,
                 enable_kvcache_centric: bool = True,
                 **kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size or config['hidden_size']
        self.num_experts = num_experts or config['num_experts']
        self.top_k = top_k
        self.enable_compression = enable_compression
        self.enable_scheduling = enable_scheduling
        self.enable_kvcache_centric = enable_kvcache_centric
        
        # Create PiKV MoE model
        self.moe_model = create_moe(
            model_type,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            use_normalization=True,
            use_lora=True,
            use_distillation=True,
            **kwargs
        )
        
        # Add compression if enabled
        if self.enable_compression:
            self.compressor = create_compressor(
                'pikv',
                hidden_size=self.hidden_size,
                compression_methods=['lora', 'pyramid', 'svd', 'quantized']
            )
        
        # Add KVCache-centric system if enabled
        if self.enable_kvcache_centric:
            self.kvcache_system = create_kvcache_centric_system(
                world_size=dconfig['world_size'],
                enable_rdma=True,
                ttft_slo=0.1,
                tbt_slo=0.05
            )
        
        # Initialize loss tracking
        self.register_buffer('total_loss', torch.tensor(0.0))
        self.register_buffer('step_count', torch.tensor(0))
    
    def forward(self, x, query=None, return_loss=True):
        """Forward pass with optional loss computation"""
        # Forward through MoE model
        output, aux_loss = self.moe_model(x)
        
        # Apply compression if enabled
        if self.enable_compression and hasattr(self, 'compressor'):
            # Extract keys and values for compression
            keys = x  # Use input as keys
            values = output  # Use output as values
            importance = torch.ones(x.size(0), x.size(1), device=x.device)
            
            compressed_keys, compressed_values = self.compressor(keys, values, importance)
            output = compressed_values
        
        # Update KVCache-centric system if enabled
        if self.enable_kvcache_centric and hasattr(self, 'kvcache_system'):
            # Register cache in distributed pool
            cache_data = output.detach()
            self.kvcache_system.distributed_pool.register_cache(f"cache_{self.step_count.item()}", cache_data)
        
        # Compute total loss
        if return_loss:
            total_loss = aux_loss
            if hasattr(self, 'compressor'):
                # Add compression loss if available
                compression_stats = self.compressor.get_compression_stats()
                if 'loss' in compression_stats:
                    total_loss = total_loss + compression_stats['loss']
            
            # Update loss tracking
            self.total_loss += total_loss.detach()
            self.step_count += 1
            
            return output, total_loss
        
        return output
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        metrics = {
            'total_loss': self.total_loss.item(),
            'step_count': self.step_count.item(),
            'average_loss': self.total_loss.item() / max(self.step_count.item(), 1)
        }
        
        if hasattr(self, 'compressor'):
            metrics.update(self.compressor.get_compression_stats())
        
        if hasattr(self, 'kvcache_system'):
            metrics.update(self.kvcache_system.get_system_stats())
        
        return metrics


class PiKVDeepSpeedManager:
    """Manager for PiKV with DeepSpeed integration"""
    
    def __init__(self, 
                 model_config: PiKVDeepSpeedConfig,
                 model_args: Dict[str, Any] = None,
                 deepspeed_config_path: str = None):
        
        self.model_config = model_config
        self.model_args = model_args or {}
        self.deepspeed_config_path = deepspeed_config_path
        
        # Initialize distributed environment
        self._init_distributed()
        
        # Create model
        self.model = PiKVDeepSpeedModel(**self.model_args)
        
        # Initialize DeepSpeed
        self._init_deepspeed()
        
        # Performance tracking
        self.training_stats = {
            'total_steps': 0,
            'total_loss': 0.0,
            'peak_memory': 0.0,
            'throughput': 0.0
        }
    
    def _init_distributed(self):
        """Initialize distributed environment"""
        if not dist.is_initialized():
            # Get environment variables set by torchrun
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Initialize process group
            dist.init_process_group(
                backend=dconfig['dist_backend'],
                init_method=dconfig['dist_url'],
                world_size=world_size,
                rank=rank
            )
            
            # Set device
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device(f'cuda:{dist.get_rank()}')
    
    def _init_deepspeed(self):
        """Initialize DeepSpeed engine"""
        # Create or load DeepSpeed configuration
        if self.deepspeed_config_path and os.path.exists(self.deepspeed_config_path):
            with open(self.deepspeed_config_path, 'r') as f:
                ds_config = json.load(f)
        else:
            ds_config = self.model_config.to_dict()
            # Save config for reference
            config_path = 'deepspeed_config.json'
            with open(config_path, 'w') as f:
                json.dump(ds_config, f, indent=2)
        
        # Initialize DeepSpeed engine
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config
        )
        
        # Enable activation checkpointing if configured
        if ds_config.get('activation_checkpointing', {}).get('enabled', False):
            deepspeed.checkpointing.configure(
                None,
                partition_activations=ds_config['activation_checkpointing']['partition_activations'],
                cpu_checkpointing=ds_config['activation_checkpointing']['cpu_checkpointing']
            )
    
    def train_step(self, batch_data, batch_target=None):
        """Single training step"""
        self.model_engine.train()
        
        # Forward pass
        if batch_target is not None:
            output, loss = self.model_engine(batch_data, return_loss=True)
            # Compute additional loss if target provided
            if hasattr(self.model, 'moe_model'):
                target_loss = torch.nn.functional.mse_loss(output, batch_target)
                loss = loss + target_loss
        else:
            output, loss = self.model_engine(batch_data, return_loss=True)
        
        # Backward pass
        self.model_engine.backward(loss)
        
        # Optimizer step
        self.model_engine.step()
        
        # Update statistics
        self.training_stats['total_steps'] += 1
        self.training_stats['total_loss'] += loss.item()
        
        # Update peak memory
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.training_stats['peak_memory'] = max(self.training_stats['peak_memory'], current_memory)
        
        return loss.item()
    
    def save_checkpoint(self, path: str, tag: str = None):
        """Save model checkpoint"""
        self.model_engine.save_checkpoint(path, tag=tag)
        
        # Save additional PiKV-specific data
        pikv_data = {
            'model_config': self.model_config.to_dict(),
            'training_stats': self.training_stats,
            'performance_metrics': self.model.get_performance_metrics()
        }
        
        pikv_path = os.path.join(path, 'pikv_data.json')
        with open(pikv_path, 'w') as f:
            json.dump(pikv_data, f, indent=2)
    
    def load_checkpoint(self, path: str, tag: str = None):
        """Load model checkpoint"""
        _, client_state = self.model_engine.load_checkpoint(path, tag=tag)
        
        # Load PiKV-specific data
        pikv_path = os.path.join(path, 'pikv_data.json')
        if os.path.exists(pikv_path):
            with open(pikv_path, 'r') as f:
                pikv_data = json.load(f)
                self.training_stats = pikv_data.get('training_stats', self.training_stats)
        
        return client_state
    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        metrics = self.model.get_performance_metrics()
        metrics.update(self.training_stats)
        
        # Add DeepSpeed-specific metrics
        if hasattr(self.model_engine, 'get_global_stats'):
            metrics.update(self.model_engine.get_global_stats())
        
        return metrics
    
    def optimize_system(self):
        """Run system optimization"""
        if hasattr(self.model, 'kvcache_system'):
            self.model.kvcache_system.optimize_system()
        
        # Clear cache and optimize memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def create_pikv_deepspeed(model_name: str = "microsoft/DialoGPT-medium",
                         enable_compression: bool = True,
                         enable_scheduling: bool = True,
                         enable_kvcache_centric: bool = True,
                         enable_moe: bool = True,
                         zero_stage: int = 3,
                         deepspeed_config_path: str = None,
                         **kwargs) -> PiKVDeepSpeedManager:
    """
    Create PiKV model with DeepSpeed integration
    
    Args:
        model_name: Name of the base model
        enable_compression: Enable PiKV compression
        enable_scheduling: Enable cache scheduling
        enable_kvcache_centric: Enable KVCache-centric system
        enable_moe: Enable MoE features
        zero_stage: DeepSpeed ZeRO stage (1, 2, or 3)
        deepspeed_config_path: Path to custom DeepSpeed config
        **kwargs: Additional model arguments
    
    Returns:
        PiKVDeepSpeedManager: Configured manager instance
    """
    
    # Create DeepSpeed configuration
    ds_config = PiKVDeepSpeedConfig(
        zero_stage=zero_stage,
        enable_moe=enable_moe,
        **kwargs
    )
    
    # Model arguments
    model_args = {
        'model_type': 'pikv',
        'hidden_size': config['hidden_size'],
        'num_experts': config['num_experts'],
        'top_k': 2,
        'enable_compression': enable_compression,
        'enable_scheduling': enable_scheduling,
        'enable_kvcache_centric': enable_kvcache_centric,
        **kwargs
    }
    
    # Create manager
    manager = PiKVDeepSpeedManager(
        model_config=ds_config,
        model_args=model_args,
        deepspeed_config_path=deepspeed_config_path
    )
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Create PiKV with DeepSpeed
    manager = create_pikv_deepspeed(
        enable_compression=True,
        enable_scheduling=True,
        enable_kvcache_centric=True,
        zero_stage=3
    )
    
    # Test training step
    batch_size = 4
    seq_len = 128
    hidden_size = config['hidden_size']
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, hidden_size).to(manager.device)
    target = torch.randn(batch_size, seq_len, hidden_size).to(manager.device)
    
    # Training step
    loss = manager.train_step(x, target)
    print(f"Training loss: {loss:.4f}")
    
    # Get metrics
    metrics = manager.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Save checkpoint
    manager.save_checkpoint("./checkpoints", tag="test")
    print("Checkpoint saved successfully!")
