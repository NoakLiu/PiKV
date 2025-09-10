"""
Enhanced PiKV MoE Configuration

Configuration options for the enhanced MoE implementation that addresses
the three key issues: dynamic load imbalance, synchronous execution, and
communication congestion.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
import torch


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    NONE = "none"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"


class ExecutionMode(Enum):
    """Execution modes"""
    SYNC = "sync"
    ASYNC = "async"
    PIPELINE = "pipeline"
    HYBRID = "hybrid"


class CommunicationStrategy(Enum):
    """Communication strategies"""
    NONE = "none"
    TOPOLOGY_AWARE = "topology_aware"
    BATCHED = "batched"
    HIERARCHICAL = "hierarchical"


class NetworkTopology(Enum):
    """Network topology types"""
    FULLY_CONNECTED = "fully_connected"
    RING = "ring"
    MESH = "mesh"
    TREE = "tree"
    CUSTOM = "custom"


# Enhanced MoE Configuration
enhanced_config = {
    # Basic model configuration
    'hidden_size': 512,
    'num_experts': 8,
    'vocab_size': 10000,
    'num_layers': 6,
    'top_k': 2,
    
    # LoRA configuration
    'lora_rank': 4,
    'lora_alpha': 1.0,
    'lora_dropout': 0.1,
    
    # Cache configuration
    'kv_cache_size': 1024,
    'cache_decrement': 128,
    'use_cache_scheduling': True,
    'cache_scheduling_policy': 'lru',
    
    # Dynamic Load Balancing Configuration
    'load_balancing': {
        'enabled': True,
        'strategy': LoadBalancingStrategy.ADAPTIVE,
        'threshold': 0.1,  # Load imbalance threshold
        'adaptation_rate': 0.01,  # How quickly to adapt
        'history_window': 100,  # Number of recent loads to track
        'overload_penalty': 2.0,  # Penalty for overloaded experts
        'underload_reward': 0.5,  # Reward for underloaded experts
        'performance_weight': 0.3,  # Weight for performance in routing
        'capacity_scaling': True,  # Scale expert capacities dynamically
        'rebalancing_frequency': 10,  # Steps between rebalancing
    },
    
    # Async Execution Configuration
    'async_execution': {
        'enabled': True,
        'mode': ExecutionMode.ASYNC,
        'max_workers': 8,  # Maximum number of async workers
        'pipeline_stages': 4,  # Number of pipeline stages
        'dependency_tracking': True,  # Track expert dependencies
        'communication_overlap': True,  # Overlap computation and communication
        'batch_size_threshold': 16,  # Minimum batch size for async execution
        'timeout': 30.0,  # Timeout for async operations (seconds)
    },
    
    # Communication Optimization Configuration
    'communication_optimization': {
        'enabled': True,
        'strategy': CommunicationStrategy.TOPOLOGY_AWARE,
        'topology': NetworkTopology.FULLY_CONNECTED,
        'bandwidth': 1.0,  # Base bandwidth (normalized)
        'latency': 0.001,  # Base latency (seconds)
        'batching_threshold': 4,  # Minimum experts for batching
        'hierarchical_levels': 2,  # Number of hierarchical levels
        'placement_optimization': True,  # Optimize expert placement
        'communication_scheduling': True,  # Schedule communications
        'cost_aware_routing': True,  # Use cost-aware routing
    },
    
    # Network Topology Configuration
    'network_topology': {
        'type': NetworkTopology.FULLY_CONNECTED,
        'bandwidth_matrix': None,  # Custom bandwidth matrix
        'latency_matrix': None,  # Custom latency matrix
        'mesh_dimensions': (2, 2),  # For mesh topology
        'ring_size': 4,  # For ring topology
        'tree_fanout': 2,  # For tree topology
    },
    
    # Performance Monitoring
    'monitoring': {
        'enabled': True,
        'metrics_interval': 100,  # Steps between metric collection
        'detailed_logging': False,  # Enable detailed logging
        'performance_profiling': False,  # Enable performance profiling
        'memory_tracking': True,  # Track memory usage
        'communication_tracking': True,  # Track communication patterns
    },
    
    # Training Configuration
    'training': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'use_mixed_precision': True,
        'gradient_checkpointing': False,
    },
    
    # Distributed Training Configuration
    'distributed': {
        'world_size': 1,
        'rank': 0,
        'backend': 'nccl',
        'init_method': 'env://',
        'expert_parallel': True,
        'data_parallel': False,
        'pipeline_parallel': False,
        'tensor_parallel': False,
    },
    
    # Advanced Optimizations
    'advanced': {
        'expert_capacity_scaling': True,  # Dynamically scale expert capacities
        'adaptive_top_k': True,  # Adapt top_k based on load
        'expert_pruning': False,  # Prune unused experts
        'cache_compression': True,  # Compress KV caches
        'quantization': False,  # Use quantization
        'sparse_attention': False,  # Use sparse attention patterns
    },
}


def get_enhanced_config(
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
    execution_mode: ExecutionMode = ExecutionMode.ASYNC,
    communication_strategy: CommunicationStrategy = CommunicationStrategy.TOPOLOGY_AWARE,
    network_topology: NetworkTopology = NetworkTopology.FULLY_CONNECTED,
    world_size: int = 1,
    **kwargs
) -> Dict:
    """
    Get enhanced configuration with specified options.
    
    Args:
        load_balancing_strategy: Load balancing strategy to use
        execution_mode: Execution mode to use
        communication_strategy: Communication strategy to use
        network_topology: Network topology to use
        world_size: Number of distributed processes
        **kwargs: Additional configuration overrides
    
    Returns:
        Configuration dictionary
    """
    config = enhanced_config.copy()
    
    # Update load balancing configuration
    config['load_balancing']['strategy'] = load_balancing_strategy
    config['load_balancing']['enabled'] = load_balancing_strategy != LoadBalancingStrategy.NONE
    
    # Update async execution configuration
    config['async_execution']['mode'] = execution_mode
    config['async_execution']['enabled'] = execution_mode != ExecutionMode.SYNC
    
    # Update communication optimization configuration
    config['communication_optimization']['strategy'] = communication_strategy
    config['communication_optimization']['enabled'] = communication_strategy != CommunicationStrategy.NONE
    config['communication_optimization']['topology'] = network_topology
    
    # Update network topology
    config['network_topology']['type'] = network_topology
    
    # Update distributed configuration
    config['distributed']['world_size'] = world_size
    
    # Apply additional overrides
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        else:
            # Handle nested keys like 'load_balancing.threshold'
            keys = key.split('.')
            if len(keys) == 2:
                if keys[0] in config and isinstance(config[keys[0]], dict):
                    config[keys[0]][keys[1]] = value
    
    return config


def create_optimization_presets() -> Dict[str, Dict]:
    """
    Create predefined optimization presets for different scenarios.
    
    Returns:
        Dictionary of preset configurations
    """
    presets = {
        'balanced': get_enhanced_config(
            load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
            execution_mode=ExecutionMode.ASYNC,
            communication_strategy=CommunicationStrategy.TOPOLOGY_AWARE,
            network_topology=NetworkTopology.FULLY_CONNECTED
        ),
        
        'high_performance': get_enhanced_config(
            load_balancing_strategy=LoadBalancingStrategy.PREDICTIVE,
            execution_mode=ExecutionMode.PIPELINE,
            communication_strategy=CommunicationStrategy.HIERARCHICAL,
            network_topology=NetworkTopology.MESH,
            load_balancing={'adaptation_rate': 0.02, 'threshold': 0.05},
            async_execution={'max_workers': 16, 'pipeline_stages': 6},
            communication_optimization={'batching_threshold': 2}
        ),
        
        'memory_efficient': get_enhanced_config(
            load_balancing_strategy=LoadBalancingStrategy.REACTIVE,
            execution_mode=ExecutionMode.SYNC,
            communication_strategy=CommunicationStrategy.BATCHED,
            network_topology=NetworkTopology.RING,
            load_balancing={'adaptation_rate': 0.005, 'threshold': 0.2},
            async_execution={'max_workers': 4},
            advanced={'expert_capacity_scaling': False, 'cache_compression': True}
        ),
        
        'distributed_optimized': get_enhanced_config(
            load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
            execution_mode=ExecutionMode.HYBRID,
            communication_strategy=CommunicationStrategy.HIERARCHICAL,
            network_topology=NetworkTopology.MESH,
            world_size=8,
            load_balancing={'rebalancing_frequency': 5},
            async_execution={'communication_overlap': True},
            communication_optimization={'hierarchical_levels': 3}
        ),
        
        'research_mode': get_enhanced_config(
            load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
            execution_mode=ExecutionMode.ASYNC,
            communication_strategy=CommunicationStrategy.TOPOLOGY_AWARE,
            network_topology=NetworkTopology.FULLY_CONNECTED,
            monitoring={'detailed_logging': True, 'performance_profiling': True},
            advanced={'adaptive_top_k': True, 'expert_pruning': True}
        )
    }
    
    return presets


def validate_config(config: Dict) -> bool:
    """
    Validate configuration for consistency and correctness.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check required fields
        required_fields = ['hidden_size', 'num_experts', 'vocab_size']
        for field in required_fields:
            if field not in config:
                print(f"Missing required field: {field}")
                return False
        
        # Validate load balancing configuration
        if config.get('load_balancing', {}).get('enabled', False):
            lb_config = config['load_balancing']
            if lb_config.get('threshold', 0) < 0 or lb_config.get('threshold', 0) > 1:
                print("Load balancing threshold must be between 0 and 1")
                return False
        
        # Validate async execution configuration
        if config.get('async_execution', {}).get('enabled', False):
            async_config = config['async_execution']
            if async_config.get('max_workers', 1) < 1:
                print("Max workers must be at least 1")
                return False
        
        # Validate communication optimization configuration
        if config.get('communication_optimization', {}).get('enabled', False):
            comm_config = config['communication_optimization']
            if comm_config.get('bandwidth', 0) <= 0:
                print("Bandwidth must be positive")
                return False
        
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Create a custom configuration
    custom_config = get_enhanced_config(
        load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
        execution_mode=ExecutionMode.ASYNC,
        communication_strategy=CommunicationStrategy.TOPOLOGY_AWARE,
        world_size=4,
        load_balancing={'threshold': 0.15, 'adaptation_rate': 0.02},
        async_execution={'max_workers': 8},
        monitoring={'detailed_logging': True}
    )
    
    print("Custom configuration:")
    print(f"Load balancing: {custom_config['load_balancing']['strategy'].value}")
    print(f"Execution mode: {custom_config['async_execution']['mode'].value}")
    print(f"Communication strategy: {custom_config['communication_optimization']['strategy'].value}")
    print(f"World size: {custom_config['distributed']['world_size']}")
    
    # Validate configuration
    if validate_config(custom_config):
        print("Configuration is valid!")
    else:
        print("Configuration validation failed!")
    
    # Show available presets
    presets = create_optimization_presets()
    print(f"\nAvailable presets: {list(presets.keys())}")
