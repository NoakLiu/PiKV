#!/usr/bin/env python3
"""
Unified MoE Example
Demonstrates usage of the consolidated MoE implementation
"""

import torch
import torch.nn as nn
import sys
import os

# Add repository root and core/single for both package and legacy imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'core', 'single'))

from moe import create_moe

def example_base_moe():
    """Basic MoE example"""
    print("Basic MoE Example")
    print("-" * 40)
    
    # Create basic MoE
    model = create_moe('base', hidden_size=512, num_experts=4, top_k=2)
    
    # Test data
    x = torch.randn(2, 64, 512)
    
    # Forward pass
    output, aux_loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss:.4f}")
    print()

def example_flex_moe():
    """Flex-MoE example for multimodal learning"""
    print("Flex-MoE Example: Multimodal Learning")
    print("-" * 40)
    
    # Create Flex-MoE
    model = create_moe('flex', hidden_size=512, num_experts=4, top_k=2)
    
    # Multimodal data
    text_data = torch.randn(2, 64, 512)
    image_data = torch.randn(2, 64, 512)
    
    # Modality info
    modality_info = {
        'image': image_data,
        'text': text_data
    }
    
    # Forward pass
    output, aux_loss = model(text_data, modality_info=modality_info)
    print(f"Text input shape: {text_data.shape}")
    print(f"Image input shape: {image_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss:.4f}")
    print()

def example_time_moe():
    """Time-MoE example for time series prediction"""
    print("Time-MoE Example: Time Series Prediction")
    print("-" * 40)
    
    # Create Time-MoE
    model = create_moe('time', hidden_size=512, num_experts=4, top_k=2)
    
    # Time series data
    x = torch.randn(2, 128, 512)
    
    # Time information
    time_info = {
        'timestamps': torch.arange(128).float(),
        'seasonality': torch.sin(torch.arange(128) * 2 * torch.pi / 24)
    }
    
    # Forward pass
    output, aux_loss = model(x, time_info=time_info)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss:.4f}")
    print()

def example_pikv_moe():
    """PiKV MoE example with LoRA and distillation"""
    print("PiKV MoE Example: LoRA + Distillation")
    print("-" * 40)
    
    # Create PiKV MoE
    model = create_moe('pikv', hidden_size=512, num_experts=4, top_k=2, 
                       rank=8, alpha=1.0, use_distillation=True)
    
    # Test data
    x = torch.randn(2, 64, 512)
    
    # Training mode
    model.train()
    output, aux_loss = model(x)
    print(f"Training mode - Input shape: {x.shape}")
    print(f"Training mode - Output shape: {output.shape}")
    print(f"Training mode - Auxiliary loss: {aux_loss:.4f}")
    
    # Evaluation mode
    model.eval()
    with torch.no_grad():
        output, aux_loss = model(x)
    print(f"Evaluation mode - Output shape: {output.shape}")
    print(f"Evaluation mode - Auxiliary loss: {aux_loss:.4f}")
    print()

def main():
    """Main function"""
    print("Unified MoE Examples")
    print("=" * 60)
    
    # Run examples
    example_base_moe()
    example_flex_moe()
    example_time_moe()
    example_pikv_moe()
    
    print("All examples completed!")
    print("\nUsage Guide:")
    print("1. Base MoE: Standard mixture of experts")
    print("2. Flex-MoE: Multimodal learning with flexible routing")
    print("3. Time-MoE: Time series prediction with temporal awareness")
    print("4. PiKV MoE: Enhanced with LoRA and knowledge distillation")

if __name__ == "__main__":
    main()
