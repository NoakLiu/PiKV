#!/usr/bin/env python3
"""
Comprehensive Distributed Training Example for PiKV

This example demonstrates:
1. Basic distributed training with PiKV
2. DeepSpeed integration
3. Performance monitoring
4. Checkpoint saving/loading
5. Error handling and recovery

Usage:
    # Basic distributed training
    torchrun --nproc_per_node=4 examples/distributed_training_example.py --mode basic
    
    # DeepSpeed training
    torchrun --nproc_per_node=4 examples/distributed_training_example.py --mode deepspeed --zero_stage 3
    
    # MoE training with DeepSpeed
    torchrun --nproc_per_node=4 examples/distributed_training_example.py --mode moe --zero_stage 3
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.distributed.distributed_pikv import DistributedPiKVManager, distributed_context
from core.distributed.deepspeed_integration import create_pikv_deepspeed, PiKVDeepSpeedConfig
from core.distributed.config import config
from core.distributed.distributed_config import distributed_config as dconfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PiKVDistributedTrainer:
    """Comprehensive distributed trainer for PiKV"""
    
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
        
        logger.info(f"Initializing trainer on rank {self.rank}/{self.world_size}, device: {self.device}")
        
        # Initialize trainer based on mode
        if args.mode == 'basic':
            self._init_basic_trainer()
        elif args.mode == 'deepspeed':
            self._init_deepspeed_trainer()
        elif args.mode == 'moe':
            self._init_moe_trainer()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Training statistics
        self.training_stats = {
            'start_time': time.time(),
            'total_steps': 0,
            'total_loss': 0.0,
            'best_loss': float('inf')
        }
    
    def _init_basic_trainer(self):
        """Initialize basic distributed trainer"""
        logger.info("Initializing basic distributed trainer")
        
        with distributed_context():
            self.manager = DistributedPiKVManager(
                rank=4,  # LoRA rank
                alpha=1.0
            )
    
    def _init_deepspeed_trainer(self):
        """Initialize DeepSpeed trainer"""
        logger.info("Initializing DeepSpeed trainer")
        
        # Create DeepSpeed configuration
        ds_config = PiKVDeepSpeedConfig(
            zero_stage=self.args.zero_stage,
            offload_optimizer=self.args.offload_optimizer,
            offload_param=self.args.offload_param,
            cpu_offload=True,
            pin_memory=True
        )
        
        # Create DeepSpeed manager
        self.manager = create_pikv_deepspeed(
            model_name="microsoft/DialoGPT-medium",
            enable_compression=True,
            enable_scheduling=True,
            enable_kvcache_centric=True,
            enable_moe=False,
            zero_stage=self.args.zero_stage,
            deepspeed_config_path=self.args.deepspeed_config
        )
    
    def _init_moe_trainer(self):
        """Initialize MoE trainer with DeepSpeed"""
        logger.info("Initializing MoE trainer with DeepSpeed")
        
        # Create DeepSpeed configuration with MoE
        ds_config = PiKVDeepSpeedConfig(
            zero_stage=self.args.zero_stage,
            offload_optimizer=self.args.offload_optimizer,
            offload_param=self.args.offload_param,
            cpu_offload=True,
            pin_memory=True,
            enable_moe=True,
            moe_expert_count=config['num_experts'],
            moe_loss_coeff=0.1,
            moe_top_k=2
        )
        
        # Create DeepSpeed manager with MoE
        self.manager = create_pikv_deepspeed(
            model_name="microsoft/DialoGPT-medium",
            enable_compression=True,
            enable_scheduling=True,
            enable_kvcache_centric=True,
            enable_moe=True,
            zero_stage=self.args.zero_stage,
            deepspeed_config_path=self.args.deepspeed_config
        )
    
    def generate_dummy_data(self, batch_size: int, seq_len: int, hidden_size: int):
        """Generate dummy training data"""
        data = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        target = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        return data, target
    
    def train_epoch(self, num_steps: int = 100):
        """Train for one epoch"""
        logger.info(f"Starting training epoch with {num_steps} steps")
        
        for step in range(num_steps):
            try:
                # Generate dummy data
                data, target = self.generate_dummy_data(
                    batch_size=self.args.batch_size,
                    seq_len=self.args.seq_len,
                    hidden_size=config['hidden_size']
                )
                
                # Training step
                if self.args.mode == 'basic':
                    loss = self.manager.train_step(data, target)
                else:  # DeepSpeed modes
                    loss = self.manager.train_step(data, target)
                
                # Update statistics
                self.training_stats['total_steps'] += 1
                self.training_stats['total_loss'] += loss
                
                if loss < self.training_stats['best_loss']:
                    self.training_stats['best_loss'] = loss
                
                # Log progress
                if step % self.args.log_interval == 0:
                    avg_loss = self.training_stats['total_loss'] / self.training_stats['total_steps']
                    logger.info(
                        f"Step {step}/{num_steps}, "
                        f"Loss: {loss:.6f}, "
                        f"Avg Loss: {avg_loss:.6f}, "
                        f"Best Loss: {self.training_stats['best_loss']:.6f}"
                    )
                
                # Save checkpoint
                if step % self.args.save_interval == 0 and step > 0:
                    self.save_checkpoint(step)
                
                # Get performance metrics
                if step % self.args.metrics_interval == 0:
                    self.log_performance_metrics()
                
            except Exception as e:
                logger.error(f"Training step {step} failed: {e}")
                # Continue training despite errors
                continue
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        try:
            checkpoint_path = f"{self.args.checkpoint_dir}/checkpoint_step_{step}.pt"
            
            if self.args.mode == 'basic':
                self.manager.save_checkpoint(checkpoint_path)
            else:  # DeepSpeed modes
                self.manager.save_checkpoint(self.args.checkpoint_dir, tag=f"step_{step}")
            
            logger.info(f"Checkpoint saved at step {step}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        try:
            if self.args.mode == 'basic':
                self.manager.load_checkpoint(checkpoint_path)
            else:  # DeepSpeed modes
                self.manager.load_checkpoint(checkpoint_path)
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def log_performance_metrics(self):
        """Log performance metrics"""
        try:
            if self.args.mode == 'basic':
                stats = self.manager.get_comprehensive_stats()
            else:  # DeepSpeed modes
                stats = self.manager.get_performance_metrics()
            
            logger.info("=== Performance Metrics ===")
            for key, value in stats.items():
                if isinstance(value, dict):
                    logger.info(f"{key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"  {sub_key}: {sub_value}")
                else:
                    logger.info(f"{key}: {value}")
            logger.info("==========================")
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
    
    def run_training(self):
        """Run complete training process"""
        logger.info("Starting distributed training")
        
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
                
                # Log epoch statistics
                epoch_time = time.time() - self.training_stats['start_time']
                avg_loss = self.training_stats['total_loss'] / self.training_stats['total_steps']
                
                logger.info(
                    f"Epoch {epoch + 1} completed. "
                    f"Time: {epoch_time:.2f}s, "
                    f"Avg Loss: {avg_loss:.6f}, "
                    f"Best Loss: {self.training_stats['best_loss']:.6f}"
                )
            
            # Final checkpoint
            self.save_checkpoint("final")
            
            # Final performance report
            self.log_performance_metrics()
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            if hasattr(self.manager, 'cleanup'):
                self.manager.cleanup()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PiKV Distributed Training Example')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'deepspeed', 'moe'],
                       help='Training mode')
    
    # DeepSpeed specific
    parser.add_argument('--zero_stage', type=int, default=3,
                       choices=[1, 2, 3],
                       help='DeepSpeed ZeRO stage')
    parser.add_argument('--offload_optimizer', action='store_true',
                       help='Offload optimizer to CPU')
    parser.add_argument('--offload_param', action='store_true',
                       help='Offload parameters to CPU')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='Path to DeepSpeed config file')
    
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
    parser.add_argument('--metrics_interval', type=int, default=25,
                       help='Performance metrics logging interval')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['deepspeed', 'moe'] and not torch.cuda.is_available():
        logger.error("DeepSpeed training requires CUDA")
        sys.exit(1)
    
    # Create and run trainer
    trainer = PiKVDistributedTrainer(args)
    trainer.run_training()


if __name__ == '__main__':
    main()
