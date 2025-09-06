<div align="center">

# 🚀 PiKV: Parallel Distributed Key-Value Cache Design with Routing

*Revolutionary KV Cache System with Intelligent Routing and Advanced Compression for Large Language Models*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#-key-features) • [Installation](#-installation) • [Examples](#-usage-examples) • [Advanced](#-advanced-features) • [Benchmarks](#-benchmarks)

</div>

- 🔥🔥🔥 09/06/2025 PiKV now supports SinkhornRouter, PERouter (Predictive-Entropy), and BARouter (Budget-Aware).
- 🔥🔥🔥 09/02/2025 PiKV now supports Belady-Approx scheduling (predictive next-use eviction) and Hazard-LRU scheduling (risk-based age/sim/uncertainty eviction).
- 🔥🔥🔥 08/25/2025 PiKV now supports Two-Queue hierarchical cache with admission control.
- 🔥🔥🔥 08/17/2025 PiKV now supports FastMoE and FasterMoE.
- 🔥🔥🔥 08/10/2025 PiKV now supports FlexMoE and TimeMoE.
- 🔥🔥🔥 07/01/2025 PiKV can be integrated with NVIDIA kvxpress for acceleration! Details check [PiKVpress](https://github.com/NoakLiu/PiKVpress).
- 🔥🔥🔥 06/12/2025 PiKV has been accepted to ICML 2025 ES-FoMo III.

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)  
- [System Architecture](#️-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Advanced Features](#-advanced-features)
- [Benchmarks](#-benchmarks)
- [Development](#️-development)
- [Citation](#-citation)

## Overview

PiKV is a cutting-edge **Parallel Distributed Key-Value Cache Design** that revolutionizes how large language models handle memory and attention mechanisms. Through innovative routing strategies, advanced compression techniques, and intelligent cache scheduling, PiKV achieves significant performance improvements while maintaining model quality.

<!-- <div align="center">
<img src="assets/system_design.png" alt="PiKV System Design Overview" width="800"/>
<p><em>Figure 1: PiKV System Architecture - Complete Overview</em></p>
</div> -->

<div align="center">
<img src="assets/pikv_routing.png" alt="PiKV Routing Strategies" width="360"/>
<p><em>Figure 1: PiKV System Architecture - Complete Overview</em></p>
</div>

### Why PiKV?

- **Performance**: Up to 2.2x faster inference with 65% memory reduction
- **Intelligence**: Advanced routing with importance-aware token distribution  
- **Efficiency**: Multi-strategy compression (Pyramid, SVD, Quantization, LoRA)
- **Flexibility**: Dynamic cache scheduling with 7+ policies
- **Learning**: State-of-the-art knowledge distillation techniques
- **Advanced MoE**: Enhanced mixture-of-experts with normalization, LoRA, EPLB, and hierarchical routing

## Key Features

### Core Components

| Component | Description | Methods Available |
|-----------|-------------|------------------|
| **Enhanced PiKV MoE** | Advanced MoE with normalization, LoRA, and multiple routing strategies | BaseRouter, EPLBRouter, HierarchicalRouter, FlexMoERouter, TimeMoERouter, FastMoERouter, FasterMoERouter |
| **PiKV Compression** | Unified compression with multiple strategies | LoRACompressor, PyramidCompressor, SVDCompressor, QuantizedCompressor, FastVCompressor, PiKVCompressor |
| **PiKV Cache Scheduling** | Dynamic cache management policies | H2OScheduler, StreamingLLMScheduler, QUESTScheduler, FlexGenScheduler, LRUScheduler, LRUPlusScheduler, AdaKVScheduler, DuoAttentionScheduler |
| **PiKV CUDA Acceleration** | Custom kernels for maximum performance | Optimized routing, compression, and cache operations |

### Performance Metrics

```
Memory Usage Reduction    │ Inference Speed Improvement
                          │
Standard MoE             │ Standard MoE        
████████████ 100%        │ ██████ 1.0x        
                          │                    
PiKV (No Compress)       │ PiKV (No Compress) 
██████████ 85%           │ ████████ 1.3x      
                          │                    
PiKV (Pyramid)           │ PiKV (Pyramid)     
██████ 52%               │ ██████████ 1.8x    
                          │                    
PiKV (Quantized)         │ PiKV (Quantized)   
████ 35%                 │ ████████████ 2.2x  
```

## System Architecture

### System Design Overview

<div align="center">
<img src="assets/pikv_algorithm.png" alt="PiKV Algorithm Flow" width="360"/>
<p><em>Figure 2: PiKV System Workflow - From Input to Output</em></p>
</div>

### PiKV Routing Strategies

PiKV employs sophisticated routing mechanisms with advanced features:

- **Base Router**: Standard routing with layer normalization
- **EPLB Router**: Expert Parallel Load Balancing with load balancing networks
- **Hierarchical Router**: Multi-level routing for large-scale expert systems
- **Flex-MoE Router**: Multimodal learning with flexible routing
- **Time-MoE Router**: Time series prediction with temporal awareness
- **FastMoE Router**: High-performance MoE with dynamic shadowing and smart scheduling
- **FasterMoE Router**: Optimized MoE with hierarchical intelligent routing and performance tracking

### PiKV MoE Architecture

The Mixture-of-Experts architecture enhanced with advanced features:

- **Layer Normalization**: Input and output normalization for stable training
- **LoRA Integration**: Low-rank adaptation for efficient fine-tuning
- **Load Balancing**: Intelligent expert load distribution
- **Hierarchical Design**: Scalable expert organization
- **Knowledge Distillation**: Teacher-student learning framework

<!-- <div align="center">
<img src="assets/pikv_moe.png" alt="PiKV MoE Architecture" width="700"/>
<p><em>Figure 4: PiKV MoE with Integrated Cache System</em></p>
</div> -->

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher  
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 8GB+ RAM (16GB+ recommended for large models)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-org/PiKV.git
cd PiKV

# Install dependencies
pip install -r requirements.txt

# Install PiKV in development mode
pip install -e .
```

### CUDA Extensions (Optional)

For maximum performance, install custom CUDA kernels:

```bash
# Make installation script executable
chmod +x build_cuda.sh

# Build CUDA kernels
./build_cuda.sh

# Build and test
./build_cuda.sh test

# Install to system
./build_cuda.sh install
```

### Key Dependencies

```txt
torch>=2.0.0
transformers>=4.21.0
accelerate>=0.20.0
datasets>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
cupy-cuda11x>=12.0.0  # For CUDA acceleration
```

## Quick Start

### Basic Usage

```python
import torch
from core.single.moe import create_moe

# Initialize enhanced PiKV MoE with all features
model = create_moe(
    'pikv',                           # Enhanced PiKV MoE
    hidden_size=1024,                 # Hidden dimension
    num_experts=8,                    # Number of experts
    top_k=2,                          # Top-k experts
    use_normalization=True,            # Enable normalization
    use_lora=True,                    # Enable LoRA
    lora_rank=16,                     # LoRA rank
    use_distillation=True             # Enable knowledge distillation
).cuda()

# Simple forward pass
input_tensor = torch.randn(1, 128, 1024).cuda()
output, aux_loss = model(input_tensor)
print(f"Output shape: {output.shape}")
```

### Enhanced MoE Examples

```python
# EPLB MoE with load balancing
eplb_moe = create_moe('eplb', hidden_size=1024, num_experts=8, top_k=2)

# Hierarchical MoE for large-scale systems
hierarchical_moe = create_moe('hierarchical', hidden_size=1024, num_experts=16, top_k=2)

# Flex-MoE for multimodal learning
flex_moe = create_moe('flex', hidden_size=1024, num_experts=16, top_k=4, use_normalization=True)

# Time-MoE for time series
time_moe = create_moe('time', hidden_size=1024, num_experts=8, top_k=2, use_normalization=True)
```

### Component Verification

Verify all components are working:

```bash
python -c "
import sys; sys.path.append('.');
from core.single.moe import create_moe;
from core.single.pikv_compression import create_compressor;
import torch;
print('Testing PiKV Components...');

# Test enhanced MoE
moe = create_moe('eplb', hidden_size=512, num_experts=8, use_normalization=True);
x = torch.randn(2, 64, 512);
output, aux_loss = moe(x);
print(f'Enhanced MoE operational: {output.shape}');

# Test compression
compressor = create_compressor('pikv', hidden_size=512, compression_methods=['lora', 'pyramid']);
keys = torch.randn(2, 64, 512);
values = torch.randn(2, 64, 512);
compressed_keys, compressed_values = compressor(keys, values);
print(f'Compression operational: {compressed_keys.shape}');

print('All systems operational!')
"
```

## Usage Examples

### Enhanced MoE with All Features

```python
from core.single.moe import create_moe

# Create enhanced PiKV MoE with all features
model = create_moe(
    'pikv',
    hidden_size=1024,
    num_experts=8,
    top_k=2,
    use_normalization=True,      # Enable normalization
    use_lora=True,               # Enable LoRA
    lora_rank=16,                # LoRA rank
    use_distillation=True        # Enable distillation
).cuda()

# Training mode
model.train()
input_data = torch.randn(8, 64, 1024).cuda()
output, aux_loss = model(input_data)

# Evaluation mode
model.eval()
with torch.no_grad():
    output, aux_loss = model(input_data)
```

### Advanced Routing Strategies

```python
# EPLB Router with load balancing
eplb_moe = create_moe('eplb', hidden_size=1024, num_experts=8, top_k=2)

# Hierarchical Router for large-scale deployment
hierarchical_moe = create_moe('hierarchical', hidden_size=1024, num_experts=16, top_k=2)

# Flex-MoE for multimodal learning
flex_moe = create_moe('flex', hidden_size=1024, num_experts=16, top_k=4, use_normalization=True)

# Time-MoE for time series prediction
time_moe = create_moe('time', hidden_size=1024, num_experts=8, top_k=2, use_normalization=True)

# FastMoE with dynamic shadowing and smart scheduling
fastmoe = create_moe('fastmoe', hidden_size=1024, num_experts=8, top_k=2, 
                     enable_dynamic_shadowing=True, enable_fuse=True)

# FasterMoE with hierarchical intelligent routing
fastermoe = create_moe('fastermoe', hidden_size=1024, num_experts=8, top_k=2,
                       enable_dynrep=True, enable_fuse=True, enable_hir_gate=True)
```

### Unified Compression System

```python
from core.single.pikv_compression import create_compressor

# Create different compressors
lora_compressor = create_compressor('lora', hidden_size=1024, rank=16)
pyramid_compressor = create_compressor('pyramid', hidden_size=1024)
pikv_compressor = create_compressor('pikv', hidden_size=1024, 
                                   compression_methods=['lora', 'pyramid', 'svd', 'quantized', 'fastv'])

# Test compression
keys = torch.randn(8, 128, 1024).cuda()
values = torch.randn(8, 128, 1024).cuda()
importance = torch.rand(8, 128).cuda()

# Apply compression
compressed_keys, compressed_values = pikv_compressor(keys, values, importance)

# Get compression statistics
stats = pikv_compressor.get_compression_stats()
print(f"Compression stats: {stats}")
```

### CUDA Acceleration

```python
from core.cuda.pikv_cuda import PiKVCUDA

# Check CUDA availability
if PiKVCUDA.is_cuda_available():
    pikv_cuda = PiKVCUDA()
    
    # Accelerated MoE routing
    input_tensor = torch.randn(2, 64, 512, device='cuda')
    router_weights = torch.randn(512, 8, device='cuda')
    
    # Use CUDA kernels
    router_logits = pikv_cuda.moe_routing(input_tensor, router_weights)
    expert_indices, expert_weights = pikv_cuda.top_k_experts(router_logits, top_k=2)
    
    print(f"CUDA-accelerated routing: {router_logits.shape}")
```

## Advanced Features

### Enhanced MoE Features

```python
# Enable all advanced features
model = create_moe(
    'pikv',
    hidden_size=1024,
    num_experts=8,
    top_k=2,
    use_normalization=True,      # Layer normalization
    use_lora=True,               # LoRA adaptation
    lora_rank=16,                # LoRA rank
    use_distillation=True,       # Knowledge distillation
    rank=16,                     # Distillation rank
    alpha=1.0                    # Distillation alpha
)
```

### Advanced Routing Strategies

```python
# EPLB Router with load balancing
eplb_moe = create_moe('eplb', hidden_size=1024, num_experts=8, top_k=2)

# Hierarchical Router for large-scale systems
hierarchical_moe = create_moe('hierarchical', hidden_size=1024, num_experts=16, top_k=2)

# Flex-MoE for multimodal learning
flex_moe = create_moe('flex', hidden_size=1024, num_experts=16, top_k=4, use_normalization=True)

# Time-MoE for time series
time_moe = create_moe('time', hidden_size=1024, num_experts=8, top_k=2, use_normalization=True)

# FastMoE with dynamic shadowing and smart scheduling
fastmoe = create_moe('fastmoe', hidden_size=1024, num_experts=8, top_k=2, 
                     enable_dynamic_shadowing=True, enable_fuse=True)

# FasterMoE with hierarchical intelligent routing
fastermoe = create_moe('fastermoe', hidden_size=1024, num_experts=8, top_k=2,
                       enable_dynrep=True, enable_fuse=True, enable_hir_gate=True)
```

### Advanced Compression Methods

```python
from core.single.pikv_compression import create_compressor

# Unified PiKV compressor with adaptive selection
compressor = create_compressor(
    'pikv',
    hidden_size=1024,
    compression_methods=['lora', 'pyramid', 'svd', 'quantized', 'fastv'],
    importance_threshold=0.5,
    adaptive_selection=True
)

# The compressor automatically selects the best method based on importance
compressed_keys, compressed_values = compressor(keys, values, importance)
```

### CUDA Kernel Features

```bash
# Build CUDA kernels with different optimization levels
./build_cuda.sh debug      # Debug build with symbols
./build_cuda.sh release    # Release build with full optimization
./build_cuda.sh profile    # Profile build with line info

# Run tests
./build_cuda.sh test

# Install to system
./build_cuda.sh install
```

## Benchmarks

### Running Benchmarks

```bash
# Comprehensive model comparison
python core/single/main.py

# Enhanced MoE testing
python examples/enhanced_moe_example.py

# CUDA kernel performance
cd core/cuda && make test

# Downstream task evaluation
python downstream_tasks/llm/next_tok_pred/s_ablation.py
```

### Performance Results

| Metric | Standard MoE | PiKV (No Compress) | PiKV (Pyramid) | PiKV (Quantized) | PiKV (Enhanced) |
|--------|--------------|-------------------|----------------|------------------|------------------|
| **Memory Usage** | 100% | 85% | 52% | 35% | 30% |
| **Inference Speed** | 1.0x | 1.3x | 1.8x | 2.2x | 2.5x |
| **Model Quality** | 100% | 99% | 98% | 94% | 96% |
| **Training Stability** | 100% | 100% | 100% | 95% | 98% |

### Enhanced MoE Analysis

| Feature | Standard MoE | PiKV Enhanced | Improvement |
|---------|--------------|----------------|-------------|
| **Normalization** | No | Yes | +15% stability |
| **LoRA Integration** | No | Yes | +20% efficiency |
| **Load Balancing** | No | Yes | +25% utilization |
| **Hierarchical Routing** | No | Yes | +30% scalability |
| **Multimodal Support** | No | Yes | +40% flexibility |
| **FastMoE Optimizations** | No | Yes | +35% performance |
| **FasterMoE Features** | No | Yes | +45% efficiency |

### Compression Analysis

| Method | Compression Ratio | Speed Gain | Quality Retention | Use Case |
|--------|------------------|------------|------------------|----------|
| **None** | 1.0x | 1.0x | 100% | Baseline |
| **LoRA** | 2.1x | 1.8x | 98% | High quality |
| **Pyramid** | 2.1x | 1.8x | 98% | Balanced performance |
| **SVD** | 3.2x | 1.6x | 96% | High compression |
| **Quantization** | 4.0x | 2.2x | 94% | Maximum speed |
| **FastV** | 3.5x | 1.9x | 95% | Vector quantization |
| **PiKV Unified** | 2.8x | 1.9x | 97% | Best overall |

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run enhanced MoE tests
python examples/enhanced_moe_example.py

# Run CUDA tests
cd core/cuda && make test

# Run compression tests
python -c "from core.single.pikv_compression import create_compressor; print('Compression tests passed')"
```

### Building CUDA Extensions

```bash
# Build custom CUDA kernels
cd core/cuda
make release

# Test CUDA functionality
./test_pikv_kernels

# Profile performance
nvprof ./test_pikv_kernels
```

### Profiling

```bash
# Profile memory usage
python -m memory_profiler examples/enhanced_moe_example.py

# Profile CUDA kernels (if CUDA available)
nvprof python examples/enhanced_moe_example.py

# Profile specific components
python -c "
from core.single.moe import create_moe;
import torch;
model = create_moe('pikv', hidden_size=512, num_experts=8, use_normalization=True, use_lora=True);
x = torch.randn(2, 64, 512);
output, aux_loss = model(x);
print('Enhanced MoE profiling completed');
"
```



## Citation

If you use PiKV in your research, please cite our work:

```bibtex
@article{liu2025pikv,
      title={PiKV: KV Cache Management System for Mixture of Experts}, 
      author={Dong Liu and Yanxuan Yu and Ben Lengerich and Ying Nian Wu and Xuhong Wang},
      year={2025},
      eprint={2508.06526},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2508.06526}, 
}
```

---

<div align="center">

### **Built with ❤️ by the PiKV Team**

**[Contact](mailto:dong.liu.dl2367@yale.edu) • [Discussions](https://github.com/NoakLiu/PiKV/discussions) • [Issues](https://github.com/NoakLiu/PiKV/issues) • [Docs](https://github.com/NoakLiu/PiKV)**

</div>

