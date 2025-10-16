#!/usr/bin/env python3
"""
PiKV-vLLM Integration Examples

This script demonstrates the integration between PiKV and vLLM, showcasing:
- PiKV-enhanced vLLM engine with compression and scheduling
- Async server with request handling
- Performance monitoring and optimization
- Distributed inference with MoE support

Usage:
    python vllm_integration_examples.py
"""

import asyncio
import torch
import sys
import os
import time
import logging
from typing import List, Dict, Any

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'single'))

try:
    from vllm_integration import (
        PiKVvLLMConfig,
        create_pikv_vllm_engine,
        create_pikv_vllm_server,
        create_pikv_vllm
    )
    # Mock SchedulingPolicy for development
    class SchedulingPolicy:
        LRU = "lru"
except ImportError:
    # Fallback for development
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'single'))
    from vllm_integration import (
        PiKVvLLMConfig,
        create_pikv_vllm_engine,
        create_pikv_vllm_server,
        create_pikv_vllm
    )
    # Mock SchedulingPolicy for development
    class SchedulingPolicy:
        LRU = "lru"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_integration():
    """
    Example 1: Basic PiKV-vLLM Integration
    Demonstrates basic integration with compression and scheduling.
    """
    print("=" * 60)
    print("Example 1: Basic PiKV-vLLM Integration")
    print("=" * 60)
    
    # Create configuration
    config = PiKVvLLMConfig(
        model_name="microsoft/DialoGPT-medium",
        enable_pikv_compression=True,
        enable_pikv_scheduling=True,
        compression_ratio=0.5,
        scheduling_policy=SchedulingPolicy.LRU,
        max_model_len=1024
    )
    
    print("Configuration created:")
    print(f"  Model: {config.model_name}")
    print(f"  Compression: {config.enable_pikv_compression}")
    print(f"  Scheduling: {config.enable_pikv_scheduling}")
    print(f"  Compression ratio: {config.compression_ratio}")
    print(f"  Scheduling policy: {config.scheduling_policy}")
    
    # Create PiKV-vLLM engine
    engine = create_pikv_vllm_engine(config)
    print("\nPiKV-vLLM engine created successfully")
    
    # Test prompts
    prompts = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    
    print(f"\n--- Testing generation with {len(prompts)} prompts ---")
    
    # Generate responses
    start_time = time.time()
    results = asyncio.run(engine.generate(prompts))
    generation_time = time.time() - start_time
    
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {result}")
    
    # Get performance statistics
    stats = engine.get_performance_stats()
    print(f"\n--- Performance Statistics ---")
    print(f"Request count: {stats['request_count']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.3f}")
    print(f"Tokens per second: {stats['tokens_per_second']:.2f}")
    
    print("\nBasic PiKV-vLLM integration working successfully!")


async def example_2_async_server():
    """
    Example 2: Async PiKV-vLLM Server
    Demonstrates async server with request handling and callbacks.
    """
    print("\n" + "=" * 60)
    print("Example 2: Async PiKV-vLLM Server")
    print("=" * 60)
    
    # Create configuration
    config = PiKVvLLMConfig(
        model_name="microsoft/DialoGPT-medium",
        enable_pikv_compression=True,
        enable_pikv_scheduling=True,
        enable_kvcache_centric=True,
        max_model_len=1024
    )
    
    # Create server
    server = create_pikv_vllm_server(config)
    print("PiKV-vLLM server created")
    
    # Start server
    await server.start(num_workers=2)
    print("Server started with 2 workers")
    
    # Request callback
    async def request_callback(request_id: str, results: List[str], error: str = None):
        if error:
            print(f"Request {request_id} failed: {error}")
        else:
            print(f"Request {request_id} completed with {len(results)} results")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: {result[:100]}...")
    
    # Submit multiple requests
    print(f"\n--- Submitting requests ---")
    request_ids = []
    
    for i in range(5):
        prompts = [f"Tell me about topic {i+1}"]
        request_id = await server.submit_request(
            prompts=prompts,
            request_id=f"async_req_{i}",
            callback=request_callback
        )
        request_ids.append(request_id)
        print(f"Submitted request {request_id}")
    
    # Wait for requests to complete
    print(f"\n--- Waiting for requests to complete ---")
    await asyncio.sleep(3)  # Give time for processing
    
    # Get server statistics
    server_stats = server.get_server_stats()
    print(f"\n--- Server Statistics ---")
    print(f"Server running: {server_stats['is_running']}")
    print(f"Queue size: {server_stats['queue_size']}")
    print(f"Active workers: {server_stats['active_workers']}")
    
    engine_stats = server_stats['engine_stats']
    print(f"Engine request count: {engine_stats['request_count']}")
    print(f"Engine total tokens: {engine_stats['total_tokens']}")
    print(f"Engine cache hit rate: {engine_stats['cache_hit_rate']:.3f}")
    
    # Stop server
    await server.stop()
    print("\nServer stopped")
    
    print("\nAsync PiKV-vLLM server working successfully!")


def example_3_performance_optimization():
    """
    Example 3: Performance Optimization
    Demonstrates performance optimization with different configurations.
    """
    print("\n" + "=" * 60)
    print("Example 3: Performance Optimization")
    print("=" * 60)
    
    # Test different configurations
    configurations = [
        {
            'name': 'Baseline (no PiKV)',
            'config': PiKVvLLMConfig(
                model_name="microsoft/DialoGPT-medium",
                enable_pikv_compression=False,
                enable_pikv_scheduling=False,
                enable_kvcache_centric=False
            )
        },
        {
            'name': 'Compression Only',
            'config': PiKVvLLMConfig(
                model_name="microsoft/DialoGPT-medium",
                enable_pikv_compression=True,
                enable_pikv_scheduling=False,
                enable_kvcache_centric=False,
                compression_ratio=0.5
            )
        },
        {
            'name': 'Scheduling Only',
            'config': PiKVvLLMConfig(
                model_name="microsoft/DialoGPT-medium",
                enable_pikv_compression=False,
                enable_pikv_scheduling=True,
                enable_kvcache_centric=False,
                scheduling_policy=SchedulingPolicy.LRU
            )
        },
        {
            'name': 'Full PiKV Optimization',
            'config': PiKVvLLMConfig(
                model_name="microsoft/DialoGPT-medium",
                enable_pikv_compression=True,
                enable_pikv_scheduling=True,
                enable_kvcache_centric=True,
                compression_ratio=0.5,
                scheduling_policy=SchedulingPolicy.LRU
            )
        }
    ]
    
    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning concepts.",
        "How does deep learning work?",
        "What are neural networks?",
        "Describe computer vision applications."
    ]
    
    results = []
    
    for config_info in configurations:
        print(f"\n--- Testing {config_info['name']} ---")
        
        # Create engine
        engine = create_pikv_vllm_engine(config_info['config'])
        
        # Run performance test
        start_time = time.time()
        
        # Generate responses
        generation_results = asyncio.run(engine.generate(test_prompts))
        
        generation_time = time.time() - start_time
        
        # Get statistics
        stats = engine.get_performance_stats()
        
        # Store results
        result = {
            'name': config_info['name'],
            'generation_time': generation_time,
            'tokens_per_second': stats['tokens_per_second'],
            'cache_hit_rate': stats['cache_hit_rate'],
            'total_tokens': stats['total_tokens']
        }
        results.append(result)
        
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Tokens per second: {stats['tokens_per_second']:.2f}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.3f}")
        print(f"Total tokens: {stats['total_tokens']}")
    
    # Compare results
    print(f"\n--- Performance Comparison ---")
    baseline_tps = results[0]['tokens_per_second']
    
    for result in results:
        speedup = result['tokens_per_second'] / baseline_tps if baseline_tps > 0 else 1.0
        print(f"{result['name']}:")
        print(f"  Tokens/sec: {result['tokens_per_second']:.2f}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Cache hit rate: {result['cache_hit_rate']:.3f}")
    
    print("\nPerformance optimization comparison completed!")


def example_4_moe_integration():
    """
    Example 4: MoE Integration
    Demonstrates PiKV MoE integration with vLLM.
    """
    print("\n" + "=" * 60)
    print("Example 4: MoE Integration")
    print("=" * 60)
    
    # Create configuration with MoE enabled
    config = PiKVvLLMConfig(
        model_name="microsoft/DialoGPT-medium",
        enable_pikv_compression=True,
        enable_pikv_scheduling=True,
        enable_pikv_moe=True,
        enable_kvcache_centric=True,
        world_size=2
    )
    
    print("Configuration with MoE enabled:")
    print(f"  MoE: {config.enable_pikv_moe}")
    print(f"  World size: {config.world_size}")
    print(f"  Compression: {config.enable_pikv_compression}")
    print(f"  KVCache-centric: {config.enable_kvcache_centric}")
    
    # Create engine
    engine = create_pikv_vllm_engine(config)
    print("\nPiKV-vLLM engine with MoE created")
    
    # Test with different types of prompts
    prompts = [
        "Explain quantum physics concepts.",
        "Describe machine learning algorithms.",
        "What are the benefits of renewable energy?",
        "How does blockchain technology work?",
        "Explain the theory of relativity."
    ]
    
    print(f"\n--- Testing MoE integration with {len(prompts)} prompts ---")
    
    # Generate responses
    start_time = time.time()
    results = asyncio.run(engine.generate(prompts))
    generation_time = time.time() - start_time
    
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {result[:200]}...")
    
    # Get comprehensive statistics
    stats = engine.get_performance_stats()
    print(f"\n--- MoE Integration Statistics ---")
    print(f"Request count: {stats['request_count']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Tokens per second: {stats['tokens_per_second']:.2f}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.3f}")
    
    # Check for MoE-specific statistics
    if 'kvcache_system_stats' in stats:
        kvcache_stats = stats['kvcache_system_stats']
        print(f"\nKVCache-centric system stats:")
        print(f"  Completion rate: {kvcache_stats['completion_rate']:.3f}")
        print(f"  Cache hit rate: {kvcache_stats['paged_cache']['hit_rate']:.3f}")
        print(f"  Cache reuse rate: {kvcache_stats['prefill_scheduler']['cache_reuse_rate']:.3f}")
    
    print("\nMoE integration working successfully!")


def example_5_quick_setup():
    """
    Example 5: Quick Setup
    Demonstrates the convenience function for quick PiKV-vLLM setup.
    """
    print("\n" + "=" * 60)
    print("Example 5: Quick Setup")
    print("=" * 60)
    
    # Quick setup with default configuration
    print("Creating PiKV-vLLM engine with quick setup...")
    
    engine = create_pikv_vllm(
        model_name="microsoft/DialoGPT-medium",
        enable_compression=True,
        enable_scheduling=True,
        enable_moe=False,
        enable_kvcache_centric=True,
        max_model_len=1024,
        compression_ratio=0.6
    )
    
    print("Engine created with quick setup")
    
    # Test with simple prompts
    prompts = [
        "Hello world!",
        "How are you?",
        "What's the weather like?"
    ]
    
    print(f"\n--- Testing quick setup with {len(prompts)} prompts ---")
    
    # Generate responses
    start_time = time.time()
    results = asyncio.run(engine.generate(prompts))
    generation_time = time.time() - start_time
    
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {result}")
    
    # Get statistics
    stats = engine.get_performance_stats()
    print(f"\n--- Quick Setup Statistics ---")
    print(f"Request count: {stats['request_count']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Tokens per second: {stats['tokens_per_second']:.2f}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.3f}")
    
    print("\nQuick setup working successfully!")


async def example_6_comprehensive_test():
    """
    Example 6: Comprehensive Test
    Demonstrates comprehensive testing of all PiKV-vLLM features.
    """
    print("\n" + "=" * 60)
    print("Example 6: Comprehensive Test")
    print("=" * 60)
    
    # Create comprehensive configuration
    config = PiKVvLLMConfig(
        model_name="microsoft/DialoGPT-medium",
        enable_pikv_compression=True,
        enable_pikv_scheduling=True,
        enable_pikv_moe=True,
        enable_kvcache_centric=True,
        compression_ratio=0.5,
        scheduling_policy=SchedulingPolicy.LRU,
        max_model_len=2048,
        world_size=2
    )
    
    # Create server
    server = create_pikv_vllm_server(config)
    
    # Start server
    await server.start(num_workers=3)
    print("Comprehensive server started with 3 workers")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Short prompts',
            'prompts': ["Hi", "Hello", "Hey"]
        },
        {
            'name': 'Medium prompts',
            'prompts': [
                "Explain machine learning",
                "What is artificial intelligence?",
                "How does deep learning work?"
            ]
        },
        {
            'name': 'Long prompts',
            'prompts': [
                "Write a detailed explanation of quantum computing and its applications in cryptography and optimization problems.",
                "Describe the history and development of artificial intelligence from its early beginnings to modern deep learning systems.",
                "Explain the principles of blockchain technology and its potential impact on various industries including finance, healthcare, and supply chain management."
            ]
        }
    ]
    
    # Process each scenario
    for scenario in test_scenarios:
        print(f"\n--- Testing {scenario['name']} ---")
        
        # Submit requests
        request_ids = []
        for i, prompt in enumerate(scenario['prompts']):
            request_id = await server.submit_request(
                prompts=[prompt],
                request_id=f"comprehensive_{scenario['name'].lower().replace(' ', '_')}_{i}"
            )
            request_ids.append(request_id)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        print(f"Submitted {len(request_ids)} requests for {scenario['name']}")
    
    # Wait for all processing to complete
    await asyncio.sleep(5)
    
    # Get final statistics
    server_stats = server.get_server_stats()
    engine_stats = server_stats['engine_stats']
    
    print(f"\n--- Comprehensive Test Results ---")
    print(f"Total requests processed: {engine_stats['request_count']}")
    print(f"Total tokens generated: {engine_stats['total_tokens']}")
    print(f"Total processing time: {engine_stats['total_time']:.2f}s")
    print(f"Average tokens per second: {engine_stats['tokens_per_second']:.2f}")
    print(f"Cache hit rate: {engine_stats['cache_hit_rate']:.3f}")
    print(f"Average time per request: {engine_stats['avg_time_per_request']:.3f}s")
    
    # Check component statistics
    if 'compression_stats' in engine_stats:
        comp_stats = engine_stats['compression_stats']
        print(f"\nCompression statistics:")
        print(f"  Compression ratio: {comp_stats.get('compression_ratio', 0):.3f}")
        print(f"  Memory saved: {comp_stats.get('memory_saved', 0):.2f}MB")
    
    if 'kvcache_system_stats' in engine_stats:
        kvcache_stats = engine_stats['kvcache_system_stats']
        print(f"\nKVCache-centric system statistics:")
        print(f"  Completion rate: {kvcache_stats['completion_rate']:.3f}")
        print(f"  Cache hit rate: {kvcache_stats['paged_cache']['hit_rate']:.3f}")
        print(f"  Cache reuse rate: {kvcache_stats['prefill_scheduler']['cache_reuse_rate']:.3f}")
        print(f"  SLO compliance: {kvcache_stats['decoding_scheduler']['slo_compliance_rate']:.3f}")
    
    # Stop server
    await server.stop()
    print("\nComprehensive test completed successfully!")


async def main():
    """Run all examples"""
    print("PiKV-vLLM Integration Examples")
    print("Advanced integration between PiKV and vLLM")
    print("=" * 80)
    
    try:
        # Run synchronous examples
        example_1_basic_integration()
        example_3_performance_optimization()
        example_4_moe_integration()
        example_5_quick_setup()
        
        # Run async examples
        await example_2_async_server()
        await example_6_comprehensive_test()
        
        print("\n" + "=" * 80)
        print("All PiKV-vLLM integration examples completed successfully!")
        print("\nKey takeaways:")
        print("1. PiKV seamlessly integrates with vLLM for enhanced performance")
        print("2. Compression and scheduling optimizations provide significant improvements")
        print("3. Async server enables high-throughput request handling")
        print("4. MoE integration supports distributed inference")
        print("5. KVCache-centric system optimizes memory management")
        print("6. Quick setup function simplifies deployment")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
