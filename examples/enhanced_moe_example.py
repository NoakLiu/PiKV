#!/usr/bin/env python3
"""
Enhanced MoE Example
Demonstrates all advanced MoE features: normalization, LoRA, EPLB, hierarchical routing
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

def example_enhanced_moe():
    """Enhanced MoE with all features"""
    print("Enhanced MoE Example: All Features Enabled")
    print("-" * 50)
    
    # Create enhanced MoE with normalization and LoRA
    model = create_moe(
        'base', 
        hidden_size=512, 
        num_experts=8, 
        top_k=2,
        use_normalization=True,
        use_lora=True,
        lora_rank=16
    )
    
    # Test data
    x = torch.randn(2, 64, 512)
    
    # Forward pass
    output, aux_loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss:.4f}")
    print(f"Normalization enabled: {hasattr(model, 'input_norm')}")
    print(f"LoRA enabled: {hasattr(model, 'expert_lora')}")
    print()

def example_eplb_moe():
    """EPLB MoE with load balancing"""
    print("EPLB MoE Example: Expert Parallel Load Balancing")
    print("-" * 50)
    
    # Create EPLB MoE
    model = create_moe('eplb', hidden_size=512, num_experts=8, top_k=2)
    
    # Test data
    x = torch.randn(2, 64, 512)
    
    # Forward pass
    output, aux_loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss:.4f}")
    print(f"Router type: {type(model.router).__name__}")
    print(f"Load balancing enabled: {hasattr(model.router, 'load_balancer')}")
    print()

def example_hierarchical_moe():
    """Hierarchical MoE for large-scale systems"""
    print("Hierarchical MoE Example: Large-Scale Expert Systems")
    print("-" * 50)
    
    # Create hierarchical MoE
    model = create_moe('hierarchical', hidden_size=512, num_experts=16, top_k=2)
    
    # Test data
    x = torch.randn(2, 64, 512)
    
    # Forward pass
    output, aux_loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss:.4f}")
    print(f"Router type: {type(model.router).__name__}")
    print(f"Number of groups: {model.router.num_groups}")
    print(f"Experts per group: {model.router.experts_per_group}")
    print()

def example_flex_moe_enhanced():
    """Enhanced Flex-MoE with normalization"""
    print("Enhanced Flex-MoE Example: Multimodal + Normalization")
    print("-" * 50)
    
    # Create enhanced Flex-MoE
    model = create_moe(
        'flex', 
        hidden_size=512, 
        num_experts=16, 
        top_k=4,
        use_normalization=True
    )
    
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
    print(f"Normalization enabled: {hasattr(model, 'input_norm')}")
    print()

def example_time_moe_enhanced():
    """Enhanced Time-MoE with normalization"""
    print("Enhanced Time-MoE Example: Time Series + Normalization")
    print("-" * 50)
    
    # Create enhanced Time-MoE
    model = create_moe(
        'time', 
        hidden_size=512, 
        num_experts=8, 
        top_k=2,
        use_normalization=True
    )
    
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
    print(f"Normalization enabled: {hasattr(model, 'input_norm')}")
    print(f"Temporal encoder: {hasattr(model.router, 'temporal_encoder')}")
    print()

def example_pikv_moe_enhanced():
    """Enhanced PiKV MoE with all features"""
    print("Enhanced PiKV MoE Example: LoRA + Distillation + Normalization")
    print("-" * 50)
    
    # Create enhanced PiKV MoE
    model = create_moe(
        'pikv', 
        hidden_size=512, 
        num_experts=8, 
        top_k=2,
        rank=16, 
        alpha=1.0, 
        use_distillation=True,
        use_normalization=True,
        use_lora=True,
        lora_rank=16
    )
    
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
    
    print(f"Normalization enabled: {hasattr(model, 'input_norm')}")
    print(f"LoRA enabled: {hasattr(model, 'expert_lora')}")
    print(f"Distillation enabled: {hasattr(model, 'teacher_projection')}")
    print()

def example_compression_integration():
    """Integration with compression methods"""
    print("Compression Integration Example")
    print("-" * 50)
    
    try:
        from pikv_compression import create_compressor
        
        # Create different compressors
        lora_compressor = create_compressor('lora', hidden_size=512, rank=16)
        pyramid_compressor = create_compressor('pyramid', hidden_size=512)
        pikv_compressor = create_compressor('pikv', hidden_size=512, 
                                          compression_methods=['lora', 'pyramid', 'svd'])
        
        # Test data
        keys = torch.randn(2, 64, 512)
        values = torch.randn(2, 64, 512)
        importance = torch.rand(2, 64)
        
        # Test LoRA compression
        compressed_keys, compressed_values = lora_compressor(keys, values, importance)
        print(f"LoRA compression - Keys: {compressed_keys.shape}, Values: {compressed_values.shape}")
        
        # Test Pyramid compression
        compressed_keys, compressed_values = pyramid_compressor(keys, values, importance)
        print(f"Pyramid compression - Keys: {compressed_keys.shape}, Values: {compressed_values.shape}")
        
        # Test unified PiKV compression
        compressed_keys, compressed_values = pikv_compressor(keys, values, importance)
        print(f"PiKV compression - Keys: {compressed_keys.shape}, Values: {compressed_values.shape}")
        
        # Get compression stats
        stats = pikv_compressor.get_compression_stats()
        print(f"Compression stats: {stats}")
        
    except ImportError:
        print("Compression module not available")
    print()

def main():
    """Main function"""
    print("Enhanced MoE Examples with All Features")
    print("=" * 70)
    
    # Run examples
    example_enhanced_moe()
    example_eplb_moe()
    example_hierarchical_moe()
    example_flex_moe_enhanced()
    example_time_moe_enhanced()
    example_pikv_moe_enhanced()
    example_compression_integration()
    
    print("All enhanced examples completed!")
    print("\nFeature Summary:")
    print("1. Normalization: Layer normalization for inputs and outputs")
    print("2. LoRA: Low-rank adaptation for efficient fine-tuning")
    print("3. EPLB: Expert parallel load balancing")
    print("4. Hierarchical: Multi-level expert routing")
    print("5. Flex-MoE: Multimodal learning with flexible routing")
    print("6. Time-MoE: Time series optimization with temporal awareness")
    print("7. PiKV MoE: Enhanced with LoRA, distillation, and normalization")
    print("8. Compression: Unified compression strategies")
    print("\nUsage:")
    print("- Enable normalization: use_normalization=True")
    print("- Enable LoRA: use_lora=True, lora_rank=16")
    print("- Select router: router_type='eplb', 'hierarchical', 'flex', 'time'")
    print("- Enable distillation: use_distillation=True (PiKV MoE only)")

if __name__ == "__main__":
    main()
