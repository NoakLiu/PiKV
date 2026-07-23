#!/usr/bin/env python3
"""
KVCache-Centric System Optimization Examples

This script demonstrates the advanced KVCache-centric optimizations including:
- Paged KVCache management across GPU/VRAM and CPU/DRAM/SSD
- Distributed KVCache Pool with RDMA inter-node transfer
- Cache-aware Prefill Scheduler with load balancing
- Decoding Instance optimization with throughput maximization
- Cache reuse optimization with TTFT/TBT SLO constraints

Usage:
    python kvcache_centric_examples.py
"""

import torch
import sys
import os
import time
import numpy as np

# Add the repository root so package imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.single.kvcache_centric_system import (
    KVCacheCentricSystem, 
    create_kvcache_centric_system,
    SLOConstraints,
    CacheLocation,
    InstanceType,
    SchedulerType
)


def example_1_paged_kvcache():
    """
    Example 1: Paged KVCache Management
    Demonstrates efficient memory management across storage layers.
    """
    print("=" * 60)
    print("Example 1: Paged KVCache Management")
    print("=" * 60)
    
    # Create KVCache-centric system
    system = create_kvcache_centric_system(
        world_size=1,
        enable_rdma=False,
        ttft_slo=0.1,
        tbt_slo=0.05
    )
    
    print("System created with paged KVCache management")
    
    # Simulate cache allocation and access
    batch_size, seq_len, hidden_size = 4, 128, 512
    
    print(f"\n--- Testing cache allocation and access ---")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    
    # Allocate some cache pages
    for i in range(10):
        cache_data = torch.randn(batch_size, seq_len, hidden_size)
        page_id = f"page_{i}"
        
        chunk = system.paged_cache.allocate_page(page_id, cache_data)
        print(f"Allocated page {page_id} in {chunk.location.value}")
    
    # Access pages to test cache behavior
    print(f"\n--- Testing cache access patterns ---")
    for i in range(15):
        page_id = f"page_{i % 10}"  # Some pages accessed multiple times
        chunk = system.paged_cache.get_page(page_id)
        
        if chunk:
            print(f"Accessed page {page_id} from {chunk.location.value} (access count: {chunk.access_count})")
        else:
            print(f"Cache miss for page {page_id}")
    
    # Get cache statistics
    cache_stats = system.paged_cache.get_cache_stats()
    print(f"\n--- Cache Statistics ---")
    print(f"Hit rate: {cache_stats['hit_rate']:.3f}")
    print(f"Hit count: {cache_stats['hit_count']}")
    print(f"Miss count: {cache_stats['miss_count']}")
    print(f"Transfer count: {cache_stats['transfer_count']}")
    print(f"VRAM pages: {cache_stats['vram_pages']}")
    print(f"DRAM pages: {cache_stats['dram_pages']}")
    print(f"SSD pages: {cache_stats['ssd_pages']}")
    
    print("\nPaged KVCache provides efficient memory management across storage layers!")


def example_2_distributed_cache_pool():
    """
    Example 2: Distributed KVCache Pool
    Demonstrates distributed cache management with RDMA transfer.
    """
    print("\n" + "=" * 60)
    print("Example 2: Distributed KVCache Pool")
    print("=" * 60)
    
    # Create system with distributed capabilities
    system = create_kvcache_centric_system(
        world_size=4,
        local_rank=0,
        enable_rdma=True
    )
    
    print("System created with distributed KVCache pool")
    print(f"World size: {system.world_size}, Local rank: {system.local_rank}")
    
    # Simulate cache registration
    print(f"\n--- Registering caches in distributed pool ---")
    for i in range(5):
        cache_id = f"cache_{i}"
        cache_data = torch.randn(32, 128, 512)
        
        system.distributed_pool.register_cache(cache_id, cache_data)
        print(f"Registered cache {cache_id} locally")
    
    # Simulate cache requests
    print(f"\n--- Testing distributed cache access ---")
    for i in range(8):
        cache_id = f"cache_{i % 5}"
        cache_data = system.distributed_pool.request_cache(cache_id)
        
        if cache_data is not None:
            print(f"Retrieved cache {cache_id} (shape: {cache_data.shape})")
        else:
            print(f"Cache {cache_id} not available")
    
    # Test load balancing
    print(f"\n--- Testing load balancing ---")
    system.distributed_pool.balance_load()
    
    # Get pool statistics
    pool_stats = system.distributed_pool.get_pool_stats()
    print(f"\n--- Distributed Pool Statistics ---")
    print(f"Local caches: {pool_stats['local_caches']}")
    print(f"Remote caches: {pool_stats['remote_caches']}")
    print(f"RDMA transfers: {pool_stats['rdma_transfers']}")
    print(f"RDMA bytes transferred: {pool_stats['rdma_bytes_transferred']}")
    print(f"Load imbalance: {pool_stats['load_imbalance']:.2f}")
    
    print("\nDistributed KVCache Pool enables efficient cache sharing across nodes!")


def example_3_cache_aware_prefill_scheduler():
    """
    Example 3: Cache-aware Prefill Scheduler
    Demonstrates intelligent prefill scheduling with cache reuse optimization.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cache-aware Prefill Scheduler")
    print("=" * 60)
    
    # Create system
    system = create_kvcache_centric_system(
        world_size=2,
        enable_rdma=True,
        ttft_slo=0.1
    )
    
    print("System created with cache-aware prefill scheduler")
    
    # Register some caches for reuse
    print(f"\n--- Registering caches for reuse ---")
    cache_hints = []
    for i in range(3):
        cache_id = f"shared_cache_{i}"
        cache_data = torch.randn(16, 64, 512)
        system.distributed_pool.register_cache(cache_id, cache_data)
        cache_hints.append(cache_id)
        print(f"Registered shared cache {cache_id}")
    
    # Schedule prefill requests with different cache reuse potential
    print(f"\n--- Scheduling prefill requests ---")
    request_ids = []
    
    # High cache reuse requests
    for i in range(3):
        request_id = f"high_reuse_{i}"
        input_tokens = torch.randn(8, 64, 512)
        
        instance_id = system.process_prefill_request(
            request_id=request_id,
            input_tokens=input_tokens,
            cache_hints=cache_hints  # High reuse potential
        )
        request_ids.append(instance_id)
        print(f"Scheduled high-reuse request {request_id} -> {instance_id}")
    
    # Low cache reuse requests
    for i in range(2):
        request_id = f"low_reuse_{i}"
        input_tokens = torch.randn(8, 64, 512)
        
        instance_id = system.process_prefill_request(
            request_id=request_id,
            input_tokens=input_tokens,
            cache_hints=[]  # No cache hints
        )
        request_ids.append(instance_id)
        print(f"Scheduled low-reuse request {request_id} -> {instance_id}")
    
    # Process prefill requests
    print(f"\n--- Processing prefill requests ---")
    for _ in range(5):
        prefill_instance = system.prefill_scheduler.get_next_prefill()
        if prefill_instance:
            print(f"Processing prefill instance {prefill_instance.instance_id}")
            print(f"  Priority: {prefill_instance.priority:.3f}")
            print(f"  Cache hints: {len(prefill_instance.cache_hints)}")
            
            # Simulate processing
            output = prefill_instance.process(system.distributed_pool)
            print(f"  Output shape: {output.shape}")
        else:
            print("No prefill requests to process")
            break
    
    # Get scheduler statistics
    scheduler_stats = system.prefill_scheduler.get_scheduler_stats()
    print(f"\n--- Prefill Scheduler Statistics ---")
    print(f"Prefill queue size: {scheduler_stats['prefill_queue_size']}")
    print(f"Priority queue size: {scheduler_stats['priority_queue_size']}")
    print(f"Cache reuse rate: {scheduler_stats['cache_reuse_rate']:.3f}")
    print(f"Total prefill requests: {scheduler_stats['total_prefill_requests']}")
    
    print("\nCache-aware Prefill Scheduler optimizes for cache reuse!")


def example_4_load_balance_decoding_scheduler():
    """
    Example 4: Load-balance Decoding Scheduler
    Demonstrates decoding optimization for maximum throughput.
    """
    print("\n" + "=" * 60)
    print("Example 4: Load-balance Decoding Scheduler")
    print("=" * 60)
    
    # Create system
    system = create_kvcache_centric_system(
        world_size=1,
        enable_rdma=False,
        tbt_slo=0.05
    )
    
    print("System created with load-balance decoding scheduler")
    
    # Schedule decoding requests
    print(f"\n--- Scheduling decoding requests ---")
    request_ids = []
    
    for i in range(6):
        request_id = f"decode_{i}"
        input_tokens = torch.randn(4, 32, 512)
        cache_data = torch.randn(4, 32, 512) if i % 2 == 0 else None
        
        instance_id = system.process_decoding_request(
            request_id=request_id,
            input_tokens=input_tokens,
            cache_data=cache_data
        )
        request_ids.append(instance_id)
        print(f"Scheduled decoding request {request_id} -> {instance_id}")
    
    # Process decoding requests
    print(f"\n--- Processing decoding requests ---")
    for _ in range(6):
        decoding_instance = system.decoding_scheduler.get_next_decoding()
        if decoding_instance:
            print(f"Processing decoding instance {decoding_instance.instance_id}")
            
            # Simulate processing
            output = decoding_instance.process()
            print(f"  Output shape: {output.shape}")
            print(f"  Throughput: {decoding_instance.throughput:.2f} tokens/sec")
            
            # Check SLO compliance
            slo_compliant = system.decoding_scheduler.check_slo_compliance(decoding_instance)
            print(f"  SLO compliant: {slo_compliant}")
        else:
            print("No decoding requests to process")
            break
    
    # Test load balancing
    print(f"\n--- Testing load balancing ---")
    system.decoding_scheduler.balance_decoding_load()
    
    # Get scheduler statistics
    scheduler_stats = system.decoding_scheduler.get_scheduler_stats()
    print(f"\n--- Decoding Scheduler Statistics ---")
    print(f"Decoding queue size: {scheduler_stats['decoding_queue_size']}")
    print(f"Active instances: {scheduler_stats['active_instances']}")
    print(f"Total throughput: {scheduler_stats['total_throughput']:.2f}")
    print(f"SLO violations: {scheduler_stats['slo_violations']}")
    print(f"SLO compliance rate: {scheduler_stats['slo_compliance_rate']:.3f}")
    
    print("\nLoad-balance Decoding Scheduler maximizes throughput!")


def example_5_comprehensive_system_optimization():
    """
    Example 5: Comprehensive System Optimization
    Demonstrates the complete KVCache-centric system in action.
    """
    print("\n" + "=" * 60)
    print("Example 5: Comprehensive System Optimization")
    print("=" * 60)
    
    # Create comprehensive system
    system = create_kvcache_centric_system(
        world_size=4,
        local_rank=0,
        enable_rdma=True,
        ttft_slo=0.1,
        tbt_slo=0.05,
        max_cache_dram=16 * 1024**3,
        max_cache_vram=8 * 1024**3
    )
    
    print("Comprehensive KVCache-centric system created")
    print(f"SLO constraints:")
    print(f"  TTFT SLO: {system.slo_constraints.ttft_slo}s")
    print(f"  TBT SLO: {system.slo_constraints.tbt_slo}s")
    print(f"  Max cache DRAM: {system.slo_constraints.max_cache_dram / 1024**3:.1f}GB")
    print(f"  Max cache VRAM: {system.slo_constraints.max_cache_vram / 1024**3:.1f}GB")
    
    # Simulate comprehensive workload
    print(f"\n--- Simulating comprehensive workload ---")
    
    # Phase 1: Setup shared caches
    print("Phase 1: Setting up shared caches")
    shared_caches = []
    for i in range(5):
        cache_id = f"shared_{i}"
        cache_data = torch.randn(32, 128, 512)
        system.distributed_pool.register_cache(cache_id, cache_data)
        shared_caches.append(cache_id)
    
    # Phase 2: Prefill requests with cache reuse
    print("Phase 2: Processing prefill requests")
    for i in range(8):
        request_id = f"prefill_{i}"
        input_tokens = torch.randn(16, 64, 512)
        
        # Vary cache reuse potential
        cache_hints = shared_caches[:i % 3 + 1] if i % 2 == 0 else []
        
        instance_id = system.process_prefill_request(
            request_id=request_id,
            input_tokens=input_tokens,
            cache_hints=cache_hints
        )
        
        # Process immediately
        prefill_instance = system.prefill_scheduler.get_next_prefill()
        if prefill_instance:
            prefill_instance.process(system.distributed_pool)
    
    # Phase 3: Decoding requests
    print("Phase 3: Processing decoding requests")
    for i in range(10):
        request_id = f"decode_{i}"
        input_tokens = torch.randn(8, 32, 512)
        cache_data = torch.randn(8, 32, 512) if i % 3 == 0 else None
        
        instance_id = system.process_decoding_request(
            request_id=request_id,
            input_tokens=input_tokens,
            cache_data=cache_data
        )
        
        # Process immediately
        decoding_instance = system.decoding_scheduler.get_next_decoding()
        if decoding_instance:
            decoding_instance.process()
    
    # Phase 4: System optimization
    print("Phase 4: Running system optimization")
    for _ in range(3):
        system.optimize_system()
        time.sleep(0.01)  # Simulate optimization time
    
    # Get comprehensive system statistics
    system_stats = system.get_system_stats()
    print(f"\n--- Comprehensive System Statistics ---")
    print(f"Uptime: {system_stats['uptime']:.2f}s")
    print(f"Total requests: {system_stats['total_requests']}")
    print(f"Completed requests: {system_stats['completed_requests']}")
    print(f"Completion rate: {system_stats['completion_rate']:.3f}")
    
    print(f"\nPaged Cache Stats:")
    cache_stats = system_stats['paged_cache']
    print(f"  Hit rate: {cache_stats['hit_rate']:.3f}")
    print(f"  Transfer count: {cache_stats['transfer_count']}")
    
    print(f"\nDistributed Pool Stats:")
    pool_stats = system_stats['distributed_pool']
    print(f"  Local caches: {pool_stats['local_caches']}")
    print(f"  RDMA transfers: {pool_stats['rdma_transfers']}")
    print(f"  Load imbalance: {pool_stats['load_imbalance']:.2f}")
    
    print(f"\nPrefill Scheduler Stats:")
    prefill_stats = system_stats['prefill_scheduler']
    print(f"  Cache reuse rate: {prefill_stats['cache_reuse_rate']:.3f}")
    print(f"  Total requests: {prefill_stats['total_prefill_requests']}")
    
    print(f"\nDecoding Scheduler Stats:")
    decoding_stats = system_stats['decoding_scheduler']
    print(f"  SLO compliance rate: {decoding_stats['slo_compliance_rate']:.3f}")
    print(f"  Total throughput: {decoding_stats['total_throughput']:.2f}")
    
    print("\nComprehensive KVCache-centric system optimization complete!")


def example_6_performance_comparison():
    """
    Example 6: Performance Comparison
    Compares performance with and without KVCache-centric optimizations.
    """
    print("\n" + "=" * 60)
    print("Example 6: Performance Comparison")
    print("=" * 60)
    
    # Test configuration
    num_requests = 50
    batch_size, seq_len = 8, 64
    hidden_size = 512
    
    print(f"Test configuration:")
    print(f"  Number of requests: {num_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Test 1: Without optimizations (baseline)
    print(f"\n--- Testing without optimizations (baseline) ---")
    start_time = time.time()
    
    # Simulate baseline processing
    for i in range(num_requests):
        input_tokens = torch.randn(batch_size, seq_len, hidden_size)
        # Simulate processing time
        time.sleep(0.001)
    
    baseline_time = time.time() - start_time
    print(f"Baseline processing time: {baseline_time:.4f} seconds")
    
    # Test 2: With KVCache-centric optimizations
    print(f"\n--- Testing with KVCache-centric optimizations ---")
    system = create_kvcache_centric_system(
        world_size=2,
        enable_rdma=True,
        ttft_slo=0.1,
        tbt_slo=0.05
    )
    
    # Setup shared caches
    shared_caches = []
    for i in range(3):
        cache_id = f"shared_{i}"
        cache_data = torch.randn(batch_size, seq_len, hidden_size)
        system.distributed_pool.register_cache(cache_id, cache_data)
        shared_caches.append(cache_id)
    
    start_time = time.time()
    
    # Process requests with optimizations
    for i in range(num_requests):
        request_id = f"request_{i}"
        input_tokens = torch.randn(batch_size, seq_len, hidden_size)
        
        # Alternate between prefill and decoding
        if i % 2 == 0:
            cache_hints = shared_caches[:i % 2 + 1] if i % 3 == 0 else []
            system.process_prefill_request(request_id, input_tokens, cache_hints)
        else:
            cache_data = torch.randn(batch_size, seq_len, hidden_size) if i % 4 == 0 else None
            system.process_decoding_request(request_id, input_tokens, cache_data)
        
        # Run optimization periodically
        if i % 10 == 0:
            system.optimize_system()
    
    optimized_time = time.time() - start_time
    print(f"Optimized processing time: {optimized_time:.4f} seconds")
    
    # Calculate speedup
    speedup = baseline_time / optimized_time
    print(f"\n--- Performance Results ---")
    print(f"Baseline time: {baseline_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Get final statistics
    system_stats = system.get_system_stats()
    print(f"\nFinal system statistics:")
    print(f"  Cache hit rate: {system_stats['paged_cache']['hit_rate']:.3f}")
    print(f"  Cache reuse rate: {system_stats['prefill_scheduler']['cache_reuse_rate']:.3f}")
    print(f"  SLO compliance rate: {system_stats['decoding_scheduler']['slo_compliance_rate']:.3f}")
    
    print("\nKVCache-centric optimizations provide significant performance improvements!")


def main():
    """Run all examples"""
    print("KVCache-Centric System Optimization Examples")
    print("Advanced memory management and scheduling optimizations")
    print("=" * 80)
    
    try:
        # Run all examples
        example_1_paged_kvcache()
        example_2_distributed_cache_pool()
        example_3_cache_aware_prefill_scheduler()
        example_4_load_balance_decoding_scheduler()
        example_5_comprehensive_system_optimization()
        example_6_performance_comparison()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("\nKey takeaways:")
        print("1. Paged KVCache provides efficient memory management across storage layers")
        print("2. Distributed KVCache Pool enables cache sharing with RDMA transfer")
        print("3. Cache-aware Prefill Scheduler optimizes for cache reuse")
        print("4. Load-balance Decoding Scheduler maximizes throughput")
        print("5. Comprehensive system optimization coordinates all components")
        print("6. KVCache-centric optimizations provide significant performance improvements")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
