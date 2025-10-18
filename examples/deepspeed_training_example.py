#!/usr/bin/env python3
"""
DeepSpeed Training Example for PiKV

This example demonstrates advanced DeepSpeed features:
1. ZeRO-1, ZeRO-2, and ZeRO-3 optimization
2. CPU offloading for memory efficiency
3. MoE training with DeepSpeed
4. Performance monitoring and profiling
5. Advanced checkpointing strategies

Usage:
    # ZeRO-1 training
    torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --zero_stage 1
    
    # ZeRO-2 with CPU offloading
    torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --zero_stage 2 --offload_optimizer
    
    # ZeRO-3 with full offloading
    torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --zero_stage 3 --offload_optimizer --offload_param
    
    # MoE training with DeepSpeed
    torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --enable_moe --zero_stage 3
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import logging
import json
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.distributed.deepspeed_integration import (
    create_pikv_deepspeed, 
    PiKVDeepSpeedConfig,
    PiKVDeepSpeedManager
)
from core.distributed.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PiKVDeepSpeedTrainer:
    """Advanced DeepSpeed trainer for PiKV"""
    
    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Initializing DeepSpeed trainer on rank {self.rank}/{self.world_size}")
        
        # Initialize DeepSpeed manager
        self._init_deepspeed_manager()
        
        # Training statistics
        self.training_stats = {
            'start_time': time.time(),
            'total_steps': 0,
            'total_loss': 0.0,
            'best_loss': float('inf'),
            'throughput': 0.0,
            'memory_usage': []
        }
    
    def _init_deepspeed_manager(self):
        """Initialize DeepSpeed manager with advanced configuration"""
        logger.info("Initializing DeepSpeed manager")
        
        # Create DeepSpeed configuration
        ds_config = PiKVDeepSpeedConfig(
            zero_stage=self.args.zero_stage,
            offload_optimizer=self.args.offload_optimizer,
            offload_param=self.args.offload_param,
            cpu_offload=self.args.cpu_offload,
            pin_memory=self.args.pin_memory,
            overlap_comm=self.args.overlap_comm,
            contiguous_gradients=self.args.contiguous_gradients,
            enable_moe=self.args.enable_moe,
            moe_expert_count=self.args.moe_expert_count,
            moe_loss_coeff=self.args.moe_loss_coeff,
            moe_top_k=self.args.moe_top_k
        )
        
        # Model arguments
        model_args = {
            'model_type': 'pikv',
            'hidden_size': self.args.hidden_size,
            'num_experts': self.args.num_experts,
            'top_k': self.args.moe_top_k,
            'enable_compression': self.args.enable_compression,
            'enable_scheduling': self.args.enable_scheduling,
            'enable_kvcache_centric': self.args.enable_kvcache_centric
        }
        
        # Create DeepSpeed manager
        self.manager = PiKVDeepSpeedManager(
            model_config=ds_config,
            model_args=model_args,
            deepspeed_config_path=self.args.deepspeed_config
        )
        
        logger.info(f"DeepSpeed manager initialized with ZeRO stage {self.args.zero_stage}")
    
    def generate_training_data(self, batch_size: int, seq_len: int, hidden_size: int):
        """Generate training data with realistic patterns"""
        # Create more realistic data patterns
        data = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        
        # Add some structure to make training more meaningful
        target = data + 0.1 * torch.randn_like(data)  # Add noise for regression task
        
        return data, target
    
    def train_step_with_profiling(self, data: torch.Tensor, target: torch.Tensor):
        """Training step with performance profiling"""
        step_start_time = time.time()
        
        # Record memory before step
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.max_memory_allocated() / 1024**3
        
        # Training step
        loss = self.manager.train_step(data, target)
        
        # Record memory after step
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.max_memory_allocated() / 1024**3
            memory_used = memory_after - memory_before
            self.training_stats['memory_usage'].append(memory_used)
        
        # Calculate throughput
        step_time = time.time() - step_start_time
        throughput = data.numel() / step_time  # elements per second
        self.training_stats['throughput'] = throughput
        
        return loss, step_time
    
    def train_epoch(self, num_steps: int = 100):
        """Train for one epoch with advanced monitoring"""
        logger.info(f"Starting training epoch with {num_steps} steps")
        
        epoch_start_time = time.time()
        epoch_losses = []
        
        for step in range(num_steps):
            try:
                # Generate training data
                data, target = self.generate_training_data(
                    batch_size=self.args.batch_size,
                    seq_len=self.args.seq_len,
                    hidden_size=self.args.hidden_size
                )
                
                # Training step with profiling
                loss, step_time = self.train_step_with_profiling(data, target)
                
                # Update statistics
                self.training_stats['total_steps'] += 1
                self.training_stats['total_loss'] += loss
                epoch_losses.append(loss)
                
                if loss < self.training_stats['best_loss']:
                    self.training_stats['best_loss'] = loss
                
                # Log progress with detailed metrics
                if step % self.args.log_interval == 0:
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    avg_memory = sum(self.training_stats['memory_usage'][-10:]) / min(10, len(self.training_stats['memory_usage']))
                    
                    logger.info(
                        f"Step {step}/{num_steps} | "
                        f"Loss: {loss:.6f} | "
                        f"Avg Loss: {avg_loss:.6f} | "
                        f"Best Loss: {self.training_stats['best_loss']:.6f} | "
                        f"Throughput: {self.training_stats['throughput']:.2f} elem/s | "
                        f"Memory: {avg_memory:.2f} GB | "
                        f"Step Time: {step_time:.3f}s"
                    )
                
                # Save checkpoint
                if step % self.args.save_interval == 0 and step > 0:
                    self.save_checkpoint(step)
                
                # Performance analysis
                if step % self.args.analysis_interval == 0:
                    self.analyze_performance()
                
                # Memory optimization
                if step % self.args.optimize_interval == 0:
                    self.manager.optimize_system()
                
            except Exception as e:
                logger.error(f"Training step {step} failed: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        avg_throughput = sum(self.training_stats['memory_usage'][-num_steps:]) / num_steps if self.training_stats['memory_usage'] else 0
        
        logger.info(
            f"Epoch completed | "
            f"Time: {epoch_time:.2f}s | "
            f"Avg Loss: {avg_epoch_loss:.6f} | "
            f"Avg Throughput: {avg_throughput:.2f} elem/s | "
            f"Steps: {num_steps}"
        )
    
    def analyze_performance(self):
        """Analyze and log performance metrics"""
        try:
            metrics = self.manager.get_performance_metrics()
            
            logger.info("=== Performance Analysis ===")
            
            # Training metrics
            logger.info(f"Total Steps: {self.training_stats['total_steps']}")
            logger.info(f"Average Loss: {self.training_stats['total_loss'] / max(self.training_stats['total_steps'], 1):.6f}")
            logger.info(f"Best Loss: {self.training_stats['best_loss']:.6f}")
            
            # Memory metrics
            if self.training_stats['memory_usage']:
                avg_memory = sum(self.training_stats['memory_usage']) / len(self.training_stats['memory_usage'])
                max_memory = max(self.training_stats['memory_usage'])
                logger.info(f"Average Memory Usage: {avg_memory:.2f} GB")
                logger.info(f"Peak Memory Usage: {max_memory:.2f} GB")
            
            # Model metrics
            if 'model_metrics' in metrics:
                model_metrics = metrics['model_metrics']
                logger.info(f"Model Forward Count: {model_metrics.get('forward_count', 0)}")
                logger.info(f"Model Average Loss: {model_metrics.get('average_loss', 0):.6f}")
                
                # Cache statistics
                if 'cache_stats' in model_metrics:
                    cache_stats = model_metrics['cache_stats']
                    if cache_stats:
                        avg_hit_rate = sum(cache['hit_rate'] for cache in cache_stats) / len(cache_stats)
                        logger.info(f"Average Cache Hit Rate: {avg_hit_rate:.3f}")
            
            logger.info("============================")
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
    
    def save_checkpoint(self, step: int):
        """Save checkpoint with metadata"""
        try:
            checkpoint_path = f"{self.args.checkpoint_dir}/deepspeed_checkpoint_step_{step}"
            
            # Save DeepSpeed checkpoint
            self.manager.save_checkpoint(checkpoint_path, tag=f"step_{step}")
            
            # Save additional metadata
            metadata = {
                'step': step,
                'training_stats': self.training_stats,
                'args': vars(self.args),
                'timestamp': time.time()
            }
            
            metadata_path = f"{checkpoint_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Checkpoint saved at step {step}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint with metadata"""
        try:
            # Load DeepSpeed checkpoint
            self.manager.load_checkpoint(checkpoint_path)
            
            # Load metadata
            metadata_path = f"{checkpoint_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.training_stats.update(metadata.get('training_stats', {}))
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def run_training(self):
        """Run complete DeepSpeed training process"""
        logger.info("Starting DeepSpeed training")
        
        try:
            # Create checkpoint directory
            os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            
            # Load checkpoint if specified
            if self.args.resume_from:
                self.load_checkpoint(self.args.resume_from)
            
            # Training loop
            for epoch in range(self.args.num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{self.args.num_epochs}")
                
                self.train_epoch(self.args.steps_per_epoch)
                
                # Save epoch checkpoint
                self.save_checkpoint(epoch)
                
                # Final performance analysis
                self.analyze_performance()
            
            # Final checkpoint
            self.save_checkpoint("final")
            
            # Final report
            total_time = time.time() - self.training_stats['start_time']
            logger.info(f"Training completed in {total_time:.2f} seconds")
            logger.info(f"Final best loss: {self.training_stats['best_loss']:.6f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PiKV DeepSpeed Training Example')
    
    # DeepSpeed configuration
    parser.add_argument('--zero_stage', type=int, default=3,
                       choices=[1, 2, 3],
                       help='DeepSpeed ZeRO stage')
    parser.add_argument('--offload_optimizer', action='store_true',
                       help='Offload optimizer to CPU')
    parser.add_argument('--offload_param', action='store_true',
                       help='Offload parameters to CPU')
    parser.add_argument('--cpu_offload', action='store_true', default=True,
                       help='Enable CPU offloading')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Pin memory for faster CPU-GPU transfer')
    parser.add_argument('--overlap_comm', action='store_true', default=True,
                       help='Overlap communication with computation')
    parser.add_argument('--contiguous_gradients', action='store_true', default=True,
                       help='Use contiguous gradients')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='Path to custom DeepSpeed config file')
    
    # Model configuration
    parser.add_argument('--hidden_size', type=int, default=1024,
                       help='Hidden size')
    parser.add_argument('--num_experts', type=int, default=8,
                       help='Number of experts')
    parser.add_argument('--enable_compression', action='store_true', default=True,
                       help='Enable compression')
    parser.add_argument('--enable_scheduling', action='store_true', default=True,
                       help='Enable cache scheduling')
    parser.add_argument('--enable_kvcache_centric', action='store_true', default=True,
                       help='Enable KVCache-centric system')
    
    # MoE configuration
    parser.add_argument('--enable_moe', action='store_true',
                       help='Enable MoE training')
    parser.add_argument('--moe_expert_count', type=int, default=8,
                       help='Number of MoE experts')
    parser.add_argument('--moe_loss_coeff', type=float, default=0.1,
                       help='MoE loss coefficient')
    parser.add_argument('--moe_top_k', type=int, default=2,
                       help='MoE top-k experts')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                       help='Number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length')
    
    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval')
    parser.add_argument('--save_interval', type=int, default=50,
                       help='Save checkpoint interval')
    parser.add_argument('--analysis_interval', type=int, default=25,
                       help='Performance analysis interval')
    parser.add_argument('--optimize_interval', type=int, default=100,
                       help='System optimization interval')
    parser.add_argument('--checkpoint_dir', type=str, default='./deepspeed_checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not torch.cuda.is_available():
        logger.error("DeepSpeed training requires CUDA")
        sys.exit(1)
    
    # Create and run trainer
    trainer = PiKVDeepSpeedTrainer(args)
    trainer.run_training()


if __name__ == '__main__':
    main()
