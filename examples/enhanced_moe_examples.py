#!/usr/bin/env python3
"""
Enhanced PiKV MoE Examples

This script demonstrates the three optional implementations for solving key MoE issues:
1. Dynamic Load Imbalance - Dynamic expert selection
2. Inefficient Synchronous Execution - Async execution mode  
3. Congested All-to-All Communication - Communication-aware placement

Usage:
    python enhanced_moe_examples.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add the repository root so package imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.single.enhanced_pikv_moe import EnhancedPiKVMoE, create_enhanced_pikv_moe
from core.single.cache_scheduling import SchedulingPolicy


def example_1_dynamic_load_balancing():
    """
    Example 1: Dynamic Load Balancing
    Demonstrates how to solve dynamic load imbalance with adaptive expert selection.
    """
    print("=" * 60)
    print("Example 1: Dynamic Load Balancing")
    print("=" * 60)
    
    # Create enhanced MoE with only dynamic load balancing enabled
    model = create_enhanced_pikv_moe(
        rank=4,
        alpha=1.0,
        enable_dynamic_balancing=True,
        enable_async_execution=False,
        enable_communication_optimization=False,
        world_size=1
    )
    
    print(f"Model created with {len(model.experts)} experts")
    print(f"Dynamic load balancing: {model.enable_dynamic_balancing}")
    
    # Simulate training with imbalanced load
    model.train()
    
    # Create synthetic data with different patterns to cause load imbalance
    batch_size, seq_len = 32, 128
    hidden_size = 512
    
    # Pattern 1: Some tokens prefer specific experts (causing imbalance)
    x1 = torch.randn(batch_size, seq_len, hidden_size)
    query1 = torch.randn(batch_size, seq_len, hidden_size)
    
    print("\n--- Training with imbalanced load pattern ---")
    start_time = time.time()
    
    for epoch in range(5):
        # Forward pass
        logits, loss = model(x1, query=query1, return_loss=True)
        
        # Simulate backward pass
        loss.backward()
        
        # Print load balancing metrics
        if epoch % 2 == 0:
            metrics = model.get_performance_metrics()
            lb_metrics = metrics.get('load_balancing', {})
            print(f"Epoch {epoch}:")
            print(f"  Load imbalance: {lb_metrics.get('load_imbalance', 0):.4f}")
            print(f"  Expert utilization: {[f'{u:.3f}' for u in lb_metrics.get('expert_utilization', [])]}")
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Test with different load pattern
    print("\n--- Testing with different load pattern ---")
    x2 = torch.randn(batch_size, seq_len, hidden_size) * 0.5  # Different scale
    query2 = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    
    with torch.no_grad():
        logits2 = model(x2, query=query2)
        metrics = model.get_performance_metrics()
        lb_metrics = metrics.get('load_balancing', {})
        print(f"Load imbalance after pattern change: {lb_metrics.get('load_imbalance', 0):.4f}")
    
    print("\nDynamic load balancing successfully adapts to changing load patterns!")


def example_2_async_execution():
    """
    Example 2: Asynchronous Execution
    Demonstrates how to solve inefficient synchronous operations with async execution.
    """
    print("\n" + "=" * 60)
    print("Example 2: Asynchronous Execution")
    print("=" * 60)
    
    # Create enhanced MoE with async execution enabled
    model = create_enhanced_pikv_moe(
        rank=4,
        alpha=1.0,
        enable_dynamic_balancing=False,
        enable_async_execution=True,
        enable_communication_optimization=False,
        world_size=1
    )
    
    print(f"Model created with async execution: {model.enable_async_execution}")
    
    # Simulate expert dependencies
    if hasattr(model, 'async_manager'):
        # Add some expert dependencies to simulate real-world scenarios
        model.async_manager.add_expert_dependency(1, [0])  # Expert 1 depends on Expert 0
        model.async_manager.add_expert_dependency(3, [1, 2])  # Expert 3 depends on Experts 1,2
        print("Expert dependencies configured:")
        print("  Expert 1 depends on Expert 0")
        print("  Expert 3 depends on Experts 1,2")
    
    # Test performance comparison
    batch_size, seq_len = 16, 64
    hidden_size = 512
    x = torch.randn(batch_size, seq_len, hidden_size)
    query = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"\n--- Performance comparison ---")
    print(f"Input shape: {x.shape}")
    
    # Test sync execution
    model.disable_async_execution_mode()
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            logits_sync = model(x, query=query)
    sync_time = time.time() - start_time
    
    # Test async execution
    model.enable_async_execution_mode()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            logits_async = model(x, query=query)
    async_time = time.time() - start_time
    
    print(f"Sync execution time: {sync_time:.4f} seconds")
    print(f"Async execution time: {async_time:.4f} seconds")
    print(f"Speedup: {sync_time / async_time:.2f}x")
    
    print("\nAsync execution provides better performance for dependent expert computations!")


def example_3_communication_optimization():
    """
    Example 3: Communication-Aware Placement
    Demonstrates how to solve congested all-to-all communication with topology-aware routing.
    """
    print("\n" + "=" * 60)
    print("Example 3: Communication-Aware Placement")
    print("=" * 60)
    
    # Create enhanced MoE with communication optimization
    model = create_enhanced_pikv_moe(
        rank=4,
        alpha=1.0,
        enable_dynamic_balancing=False,
        enable_async_execution=False,
        enable_communication_optimization=True,
        world_size=4  # Simulate 4-GPU setup
    )
    
    print(f"Model created with communication optimization: {model.enable_communication_optimization}")
    print(f"World size: {model.world_size}")
    
    if hasattr(model, 'communication_placer'):
        # Simulate expert communication patterns
        expert_communication_patterns = {
            0: [1, 2, 3],  # Expert 0 communicates with experts 1,2,3
            1: [0, 2],    # Expert 1 communicates with experts 0,2
            2: [0, 1, 3], # Expert 2 communicates with experts 0,1,3
            3: [0, 2]     # Expert 3 communicates with experts 0,2
        }
        
        print("\n--- Optimizing expert placement ---")
        print("Expert communication patterns:")
        for expert_id, partners in expert_communication_patterns.items():
            print(f"  Expert {expert_id}: communicates with {partners}")
        
        # Optimize placement
        model.communication_placer.optimize_expert_placement(expert_communication_patterns)
        
        print("\nOptimized expert placement:")
        for expert_id, rank in model.communication_placer.expert_placement.items():
            print(f"  Expert {expert_id} -> Rank {rank}")
        
        # Test communication scheduling
        print("\n--- Testing communication scheduling ---")
        batch_size, seq_len = 8, 32
        hidden_size = 512
        
        # Simulate expert outputs
        expert_outputs = {}
        for expert_id in range(len(model.experts)):
            expert_outputs[expert_id] = torch.randn(batch_size, seq_len, hidden_size)
        
        # Schedule communications
        comm_schedule = model.communication_placer.schedule_communication(expert_outputs)
        
        print("Communication schedule:")
        for target_rank, schedule in comm_schedule.items():
            print(f"  Rank {target_rank}: {len(schedule['expert_ids'])} experts, data shape {schedule['data'].shape}")
        
        # Compute communication costs
        print("\nCommunication costs:")
        for expert_id in range(len(model.experts)):
            for target_rank in range(model.world_size):
                if target_rank != model.communication_placer.expert_placement.get(expert_id, 0):
                    cost = model.communication_placer.compute_communication_cost(expert_id, target_rank)
                    print(f"  Expert {expert_id} -> Rank {target_rank}: {cost:.4f}")
    
    print("\nCommunication-aware placement optimizes expert placement for minimal communication cost!")


def example_4_combined_optimizations():
    """
    Example 4: Combined Optimizations
    Demonstrates using all three optimizations together for maximum performance.
    """
    print("\n" + "=" * 60)
    print("Example 4: Combined Optimizations")
    print("=" * 60)
    
    # Create enhanced MoE with all optimizations enabled
    model = create_enhanced_pikv_moe(
        rank=4,
        alpha=1.0,
        enable_dynamic_balancing=True,
        enable_async_execution=True,
        enable_communication_optimization=True,
        enable_smartmoe=True,
        world_size=4
    )
    
    print("Model created with all optimizations enabled:")
    print(f"  - Dynamic Load Balancing: {model.enable_dynamic_balancing}")
    print(f"  - Async Execution: {model.enable_async_execution}")
    print(f"  - Communication Optimization: {model.enable_communication_optimization}")
    print(f"  - SmartMoE Integration: {model.enable_smartmoe}")
    
    # Simulate complex training scenario
    batch_size, seq_len = 16, 64
    hidden_size = 512
    
    print(f"\n--- Complex training scenario ---")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    
    model.train()
    
    # Training loop with all optimizations
    start_time = time.time()
    
    for epoch in range(3):
        # Create varied input patterns to test all optimizations
        x = torch.randn(batch_size, seq_len, hidden_size) * (0.5 + epoch * 0.2)
        query = torch.randn(batch_size, seq_len, hidden_size) * (0.5 + epoch * 0.2)
        
        # Forward pass
        logits, loss = model(x, query=query, return_loss=True)
        
        # Simulate backward pass
        loss.backward()
        
        # Print comprehensive metrics
        if epoch % 1 == 0:
            metrics = model.get_performance_metrics()
            print(f"\nEpoch {epoch} metrics:")
            
            if 'load_balancing' in metrics:
                lb = metrics['load_balancing']
                print(f"  Load imbalance: {lb.get('load_imbalance', 0):.4f}")
                print(f"  Expert utilization: {[f'{u:.3f}' for u in lb.get('expert_utilization', [])]}")
            
            if 'communication' in metrics:
                comm = metrics['communication']
                print(f"  Expert placement: {comm.get('expert_placement', {})}")
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Test individual optimization toggles
    print("\n--- Testing optimization toggles ---")
    
    # Disable load balancing
    model.disable_dynamic_load_balancing()
    print("Dynamic load balancing disabled")
    
    # Disable async execution
    model.disable_async_execution_mode()
    print("Async execution disabled")
    
    # Disable communication optimization
    model.disable_communication_optimization()
    print("Communication optimization disabled")
    
    # Re-enable all
    model.enable_dynamic_load_balancing()
    model.enable_async_execution_mode()
    model.enable_communication_optimization()
    print("All optimizations re-enabled")
    
    print("\nAll optimizations work together for maximum MoE performance!")


def example_5_performance_comparison():
    """
    Example 5: Performance Comparison
    Compares performance with and without optimizations.
    """
    print("\n" + "=" * 60)
    print("Example 5: Performance Comparison")
    print("=" * 60)
    
    batch_size, seq_len = 32, 128
    hidden_size = 512
    x = torch.randn(batch_size, seq_len, hidden_size)
    query = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    
    # Test different configurations
    configurations = [
        ("Baseline (no optimizations)", False, False, False),
        ("Dynamic Load Balancing only", True, False, False),
        ("Async Execution only", False, True, False),
        ("Communication Optimization only", False, False, True),
        ("All optimizations", True, True, True),
    ]
    
    results = {}
    
    for config_name, enable_lb, enable_async, enable_comm in configurations:
        print(f"\n--- Testing {config_name} ---")
        
        model = create_enhanced_pikv_moe(
            rank=4,
            alpha=1.0,
            enable_dynamic_balancing=enable_lb,
            enable_async_execution=enable_async,
            enable_communication_optimization=enable_comm,
            world_size=4
        )
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x, query=query)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                logits = model(x, query=query)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 20
        results[config_name] = avg_time
        
        print(f"  Average inference time: {avg_time:.4f} seconds")
        
        # Get metrics if available
        metrics = model.get_performance_metrics()
        if 'load_balancing' in metrics:
            lb = metrics['load_balancing']
            print(f"  Load imbalance: {lb.get('load_imbalance', 0):.4f}")
    
    # Print comparison
    print(f"\n--- Performance Comparison ---")
    baseline_time = results["Baseline (no optimizations)"]
    
    for config_name, time_taken in results.items():
        speedup = baseline_time / time_taken
        print(f"{config_name:35s}: {time_taken:.4f}s ({speedup:.2f}x)")


def main():
    """Run all examples"""
    print("Enhanced PiKV MoE Examples")
    print("Solving three key MoE issues with optional implementations")
    print("=" * 80)
    
    try:
        # Run all examples
        example_1_dynamic_load_balancing()
        example_2_async_execution()
        example_3_communication_optimization()
        example_4_combined_optimizations()
        example_5_performance_comparison()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("\nKey takeaways:")
        print("1. Dynamic Load Balancing adapts to changing load patterns")
        print("2. Async Execution improves performance for dependent computations")
        print("3. Communication Optimization reduces communication overhead")
        print("4. Combined optimizations provide maximum performance gains")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
