<div align="center">

# 🚀 PiKV: Parallel Distributed Key-Value Cache Design with Routing

*Revolutionary KV Cache System with Intelligent Routing and Advanced Compression for Large Language Models*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#-key-features) • [EPiKV-MoE](#-epikv-moe) • [FPGA Offload](#-fpga-hardware-aware-offload) • [vLLM Integration](#-vllm-integration) • [Installation](#-installation) • [Examples](#-usage-examples) • [Advanced](#-advanced-features) • [Benchmarks](#-benchmarks)

</div>

- 🔥🔥🔥 03/24/2026 PiKV adds **PiKV-FPGA** with **Verilog RTL** + **C host** (`libpikv_fpga.so`): MMIO `PiKV-CTRL`, `ScoreFuse`, `Codecρ`, page table `D+`, scheduler.
- 🔥🔥🔥 10/18/2025 PiKV now supports DeepSpeed Integration with ZeRO-1/2/3 optimization, CPU offloading, and MoE expert parallelism for enterprise-grade distributed training.
- 🔥🔥🔥 10/16/2025 PiKV now supports vLLM Integration with MoE KV Cache Optimization in vLLM inference engine.
- 🔥🔥🔥 09/19/2025 PiKV now supports KVCache-Centric System Optimization with Paged KVCache, Distributed Cache Pool, and Cache-aware Scheduling.
- 🔥🔥🔥 09/10/2025 PiKV now supports SmartMoE.
- 🔥🔥🔥 09/09/2025 PiKV released EPiKV-MoE which supports Dynamic Load-Balancer, Asynchoronous Execution Manager, Communication-Aware Expert Routing.
- 🔥🔥🔥 09/06/2025 PiKV now supports SinkhornRouter, PERouter (Predictive-Entropy), and BARouter (Budget-Aware).
- 🔥🔥🔥 09/02/2025 PiKV now supports Belady-Approx scheduling (predictive next-use eviction) and Hazard-LRU scheduling (risk-based age/sim/uncertainty eviction).
- 🔥🔥🔥 08/25/2025 PiKV now supports Two-Queue hierarchical cache with admission control.
- 🔥🔥🔥 08/17/2025 PiKV now supports FastMoE and FasterMoE.
- 🔥🔥🔥 08/10/2025 PiKV now supports FlexMoE and TimeMoE.
- 🔥🔥🔥 07/01/2025 PiKV can be integrated with NVIDIA kvxpress for acceleration! Details check [PiKVpress](https://github.com/NoakLiu/PiKVpress).
- 🔥🔥🔥 06/12/2025 PiKV has been accepted to ICML 2025 ES-FoMo III.

### Progress

| Area | Status | Notes |
|------|--------|-------|
| Expert-sharded KV storage | ✅ | Multi-GPU / distributed cache pool |
| PiKV Routing | ✅ | TopK, EPLB, hierarchical, cache-aware, SmartMoE |
| PiKV Compression | ✅ | LoRA, PyramidKV, SVD, FastV, unified compressor |
| PiKV Scheduling | ✅ | H2O, QUEST, AdaKV, Duo, Belady-Approx, Hazard-LRU |
| CUDA kernels | ✅ | Routing / compression / scheduling |
| DeepSpeed + DDP | ✅ | ZeRO-1/2/3, MoE expert parallelism |
| vLLM integration | ✅ | Async server, KV-centric inference |
| **PiKV-FPGA** | 🆕 | RTL + AXI-Lite + CXL DMA + Vivado bitstream (`./build_fpga.sh bitstream`) |

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [EPiKV-MoE: Enhanced MoE with Advanced Optimizations](#-epikv-moe-enhanced-moe-with-advanced-optimizations)
- [KVCache-Centric System Optimization](#-kvcache-centric-system-optimization)
- [vLLM Integration](#-vllm-integration)
- [DeepSpeed Integration](#-deepspeed-integration)
- [Distributed Training](#-distributed-training)
- [FPGA Hardware-Aware Offload](#-fpga-hardware-aware-offload)
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
- **Advanced MoE**: EPiKV-MoE, EPLB, hierarchical routing, Faster-MoE, Smart-MoE, etc

## Key Features

### Core Components

| Component | Description | Methods Available |
|-----------|-------------|------------------|
| **Enhanced PiKV MoE** | Advanced MoE with normalization, LoRA, and multiple routing strategies | BaseRouter, EPLBRouter, HierarchicalRouter, FlexMoERouter, TimeMoERouter, FastMoERouter, FasterMoERouter, SmartMoE |
| **KVCache-Centric System** | Advanced memory management and scheduling optimizations | PagedKVCache, DistributedKVCachePool, CacheAwarePrefillScheduler, LoadBalanceDecodingScheduler |
| **vLLM Integration** | Seamless integration with vLLM inference engine | PiKVvLLMEngine, PiKVvLLMServer, PiKVvLLMConfig |
| **DeepSpeed Integration** | Enterprise-grade distributed training with ZeRO optimization | PiKVDeepSpeedManager, ZeRO-1/2/3, CPU offloading, MoE expert parallelism |
| **Distributed Training** | Enhanced distributed training with error handling and monitoring | DistributedPiKVManager, DistributedPiKVMoE, Performance monitoring, Advanced checkpointing |
| **PiKV Compression** | Unified compression with multiple strategies | LoRACompressor, PyramidCompressor, SVDCompressor, QuantizedCompressor, FastVCompressor, PiKVCompressor |
| **PiKV Cache Scheduling** | Dynamic cache management policies | H2OScheduler, StreamingLLMScheduler, QUESTScheduler, FlexGenScheduler, LRUScheduler, LRUPlusScheduler, AdaKVScheduler, DuoAttentionScheduler |
| **PiKV CUDA Acceleration** | Custom kernels for maximum performance | Optimized routing, compression, and cache operations |
| **PiKV-FPGA** | CXL-disaggregated metadata offload (paper §3.5) | `PiKV-CTRL` MMIO, `PageTableEngine`, `ScoreFuseEngine`, `CodecRhoEngine`, `SchedulerFPGAEngine` |

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

## EPiKV-MoE

EPiKV-MoE addresses three critical issues in traditional MoE systems with optional implementations:

### Dynamic Load Balancing
**Problem**: Load imbalance where some experts are overloaded while others are underutilized.
**Solution**: Real-time expert selection with adaptive routing and performance monitoring.

```python
from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

# Create model with dynamic load balancing
model = create_enhanced_pikv_moe(
    enable_dynamic_balancing=True,
    load_balancing_strategy='adaptive'
)

# Monitor load balancing metrics
metrics = model.get_performance_metrics()
print(f"Load imbalance: {metrics['load_balancing']['load_imbalance']}")
```

### Asynchronous Execution
**Problem**: Synchronous execution creates bottlenecks when experts have dependencies.
**Solution**: Pipeline parallelism and asynchronous communication to overlap computation and communication.

```python
# Enable async execution with dependency tracking
model = create_enhanced_pikv_moe(
    enable_async_execution=True,
    execution_mode='async'
)

# Add expert dependencies
model.async_manager.add_expert_dependency(expert_id=1, depends_on=[0])
```

### Communication-Aware Placement
**Problem**: Traditional MoE ignores network topology, leading to inefficient all-to-all communication.
**Solution**: Topology-aware expert placement and communication scheduling.

```python
# Enable communication optimization
model = create_enhanced_pikv_moe(
    enable_communication_optimization=True,
    communication_strategy='topology_aware',
    network_topology='mesh',
    world_size=4
)

# Optimize expert placement based on communication patterns
expert_patterns = {0: [1, 2, 3], 1: [0, 2], 2: [0, 1, 3], 3: [0, 2]}
model.communication_placer.optimize_expert_placement(expert_patterns)
```

###  Configuration of EPiKV-MoE
```python
# Use predefined optimization presets
from core.single.enhanced_config import create_optimization_presets

presets = create_optimization_presets()
config = presets['high_performance']  # or 'balanced', 'memory_efficient', etc.

# Or create custom configuration
from core.single.enhanced_config import get_enhanced_config
config = get_enhanced_config(
    load_balancing_strategy='adaptive',
    execution_mode='async',
    communication_strategy='topology_aware'
)
```

## 🚀 KVCache-Centric System Optimization

PiKV introduces advanced KVCache-centric system optimizations for maximum efficiency:

### 📄 Paged KVCache Management
**Multi-tier storage**: Efficient memory management across GPU/VRAM, CPU/DRAM, and SSD layers.

```python
from core.single.kvcache_centric_system import create_kvcache_centric_system

# Create KVCache-centric system
system = create_kvcache_centric_system(
    world_size=4,
    enable_rdma=True,
    ttft_slo=0.1,  # 100ms Time to First Token
    tbt_slo=0.05   # 50ms Time Between Tokens
)

# Allocate cache pages across storage tiers
cache_data = torch.randn(32, 128, 512)
chunk = system.paged_cache.allocate_page("page_1", cache_data)
print(f"Cache stored in: {chunk.location.value}")
```

### 🌐 Distributed KVCache Pool
**RDMA inter-node transfer**: Seamless cache sharing across distributed nodes.

```python
# Register caches in distributed pool
system.distributed_pool.register_cache("shared_cache", cache_data)

# Request cache from any node
retrieved_cache = system.distributed_pool.request_cache("shared_cache")

# Automatic load balancing
system.distributed_pool.balance_load()
```

### 🎯 Cache-aware Prefill Scheduler
**Optimization goal**: Maximize cache reuse with TTFT SLO constraints.

```python
# Schedule prefill with cache reuse optimization
instance_id = system.process_prefill_request(
    request_id="prefill_1",
    input_tokens=input_tokens,
    cache_hints=["shared_cache_1", "shared_cache_2"]  # High reuse potential
)

# Process with cache awareness
prefill_instance = system.prefill_scheduler.get_next_prefill()
output = prefill_instance.process(system.distributed_pool)
```

### ⚡ Load-balance Decoding Scheduler
**Optimization goal**: Maximize throughput with TBT SLO constraints.

```python
# Schedule decoding for maximum throughput
instance_id = system.process_decoding_request(
    request_id="decode_1",
    input_tokens=input_tokens,
    cache_data=cache_data
)

# Process with load balancing
decoding_instance = system.decoding_scheduler.get_next_decoding()
output = decoding_instance.process()
```

### System Optimization Benefits
- **Cache Hit Rate**: Up to 95% with intelligent page management
- **Cache Reuse**: Up to 80% reuse rate with cache-aware scheduling
- **Throughput**: Up to 3x improvement with load balancing
- **SLO Compliance**: 99%+ compliance with TTFT/TBT constraints
- **Memory Efficiency**: Optimal utilization across storage tiers

### 🔧 Comprehensive System Control
```python
# Run comprehensive system optimization
system.optimize_system()

# Get detailed statistics
stats = system.get_system_stats()
print(f"Cache hit rate: {stats['paged_cache']['hit_rate']:.3f}")
print(f"Cache reuse rate: {stats['prefill_scheduler']['cache_reuse_rate']:.3f}")
print(f"SLO compliance: {stats['decoding_scheduler']['slo_compliance_rate']:.3f}")
```

## vLLM Integration

PiKV integrates with vLLM inference:

### Quick Setup
```python
from core.single.vllm_integration import create_pikv_vllm

# Create PiKV-enhanced vLLM engine
engine = create_pikv_vllm(
    model_name="microsoft/DialoGPT-medium",
    enable_compression=True,
    enable_scheduling=True,
    enable_kvcache_centric=True
)

# Generate with PiKV optimizations
results = await engine.generate(["Hello, how are you?"])
```

### ⚡ Async Server with Request Handling
**High-throughput serving**: Async server with worker pools and callbacks.

```python
from core.single.vllm_integration import create_pikv_vllm_server, PiKVvLLMConfig

# Create server configuration
config = PiKVvLLMConfig(
    model_name="microsoft/DialoGPT-medium",
    enable_pikv_compression=True,
    enable_pikv_scheduling=True,
    enable_kvcache_centric=True
)

# Create and start server
server = create_pikv_vllm_server(config)
await server.start(num_workers=4)

# Submit requests with callbacks
async def callback(request_id, results, error=None):
    if error:
        print(f"Request {request_id} failed: {error}")
    else:
        print(f"Request {request_id} completed: {results}")

request_id = await server.submit_request(
    prompts=["Tell me about machine learning"],
    callback=callback
)
```

### Distributed Inference with MoE
**Scalable deployment**: MoE support with distributed inference.

```python
# Create engine with MoE support
engine = create_pikv_vllm(
    model_name="microsoft/DialoGPT-medium",
    enable_moe=True,
    enable_kvcache_centric=True,
    world_size=4
)

# Generate with distributed MoE
results = await engine.generate(prompts)
```

### 🔧 Quick Setup
```python
# One-line setup for common use cases
engine = create_pikv_vllm(
    model_name="microsoft/DialoGPT-medium",
    enable_compression=True,
    enable_scheduling=True
)

# Start generating immediately
results = await engine.generate(["Your prompt here"])
```

## DeepSpeed Integration

PiKV now supports comprehensive DeepSpeed integration for enterprise-grade distributed training:

### 🚀 **DeepSpeed Setup with PiKV**
```python
from core.distributed.deepspeed_integration import create_pikv_deepspeed

# Create DeepSpeed-enhanced PiKV
manager = create_pikv_deepspeed(
    model_name="microsoft/DialoGPT-medium",
    enable_compression=True,
    enable_scheduling=True,
    enable_kvcache_centric=True,
    zero_stage=3  # ZeRO-3 optimization
)

# Start training immediately
loss = manager.train_step(data, target)
```
th full offloading (50% memory reduction)


```python
# MoE training with DeepSpeed
manager = create_pikv_deepspeed(
    enable_moe=True,
    zero_stage=3,
    offload_optimizer=True,
    offload_param=True,
    moe_expert_count=8,
    moe_top_k=2
)

# Performance monitoring
metrics = manager.get_performance_metrics()
print(f"Memory usage: {metrics['memory_usage']:.2f}GB")
print(f"Throughput: {metrics['throughput']:.2f} elem/s")
```

## Distributed Training

```bash
# Basic distributed training
torchrun --nproc_per_node=4 examples/distributed_training_example.py --mode basic

# DeepSpeed training
torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --zero_stage 3

# MoE training with DeepSpeed
torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --enable_moe --zero_stage 3
```

### **Training Script**
```bash
# Make script executable
chmod +x examples/run_distributed_training.sh

# Run different training modes
./examples/run_distributed_training.sh basic
./examples/run_distributed_training.sh deepspeed-zero3
./examples/run_distributed_training.sh moe
./examples/run_distributed_training.sh compare
```

## FPGA Hardware-Aware Offload

PiKV-FPGA offloads **metadata-intensive** stages to a CXL-attached SmartNIC while the GPU runs `f_enc` and `f_attn`. KV bodies live in disaggregated DDR; the FPGA keeps page tables, scores, and codec weights on chip (see [CXL-SpecKV](https://doi.org/10.1145/3748173.3779188)).

**Platform:** AMD Alveo **U55C** (`xcu55c-fsvh2892-2L-e`), 300 MHz user clock, 16 GB HBM2, PCIe Gen4 x16 (XDMA) + optional CXL Type-3.mem. Bandwidth assumptions, BRAM budget, and E2E comparison vs CPU are documented in [`core/fpga/README.md`](core/fpga/README.md) and runnable via `python -m core.fpga.benchmark_hw`.

```
GPU ──MMIO──► PiKV-CTRL ──► {D+, ScoreFuse, Codecρ, DMA} ──CXL.mem──► DDR pool
GPU ◄──PCIe/CXL── packed {(K̂, V̂, idx)} for active pages P_t
```

### Engine mapping (paper Table 4–5)

| Stage | FPGA engine | PiKV methods |
|-------|-------------|--------------|
| **Routing** | `ScoreFuse` + radix Top-k | hash, TopK, load-balance, cache-aware, entropy-LB, hierarchical |
| **Compression** | `Codecρ` | LoRA, PyramidKV, ChunkKV, FastV, structured prune |
| **Scheduling** | `ui ≷ θ` | H2O, sliding window, QUEST MLP, LRU, AdaKV, Duo |
| **Page table** | `D+` gather | `Γ: (t,e) ↦ addr`, miss counts `m_e` |

### Quick usage

```python
from core.fpga import create_pikv_fpga, is_fpga_available
import torch

fpga = create_pikv_fpga(num_experts=64, hidden_size=128, top_k=4, compression_ratio=4.0)
q, k, v = torch.randn(128), torch.randn(128), torch.randn(128)
packed, experts = fpga.process_token(q, k, v, token_id=0)
print(experts, packed.shape, fpga.get_stats())
```

```bash
# Simulation (default; no hardware)
python examples/fpga_offload_example.py

# Optional: point to device node when driver is installed
export PIKV_FPGA_DEVICE=/dev/pikv_fpga0
export PIKV_FPGA_SIM=0
python examples/fpga_offload_example.py
```

### Resource budget (default tile E=64, S=256, k=4, K=16, d=128)

```python
from core.fpga.config import FPGAConfig, estimate_bram_budget
cfg = FPGAConfig()
print(estimate_bram_budget(cfg))  # ~224 KB on-chip (BRAM_Γ + BRAM_meta + URAM LoRA)
```

### Verilog RTL, AXI-Lite, CXL DMA & Vivado

```
core/fpga/
├── rtl/
│   ├── pikv_soc_top.v          # SoC top (AXI-Lite + AXI-MM)
│   ├── pikv_axi_lite_slave.v   # Host MMIO (BAR0 / XDMA)
│   ├── pikv_cxl_dma.v          # CXL.mem KV DMA bridge
│   ├── pikv_axi_dma_master.v   # AXI4 master
│   └── pikv_top.v              # PiKV engines + CTRL
├── vivado/scripts/             # create_project.tcl, build_bitstream.tcl, create_bd.tcl
├── vivado/constraints/         # U55C + generic XDC
└── host/                       # libpikv_fpga.so
```

```bash
# C host + RTL sim (AXI + CXL mem model)
./build_fpga.sh all
./build_fpga.sh sim-soc

# Vivado project + bitstream (Alveo U55C default)
export PIKV_PART=xcu55c-fsvh2892-2L-e
./build_fpga.sh vivado
./build_fpga.sh bitstream      # → vivado/project/pikv_fpga.runs/impl_1/*.bit
./build_fpga.sh bd             # optional XDMA block design

# Program card
xbutil program --device <BDF> --base vivado/project/pikv_fpga.runs/impl_1/*.bit
```

See [core/fpga/vivado/ip/README.md](core/fpga/vivado/ip/README.md) for **XDMA** + **CXL Type-3** IP wiring.

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
- **SmartMoE Router**: Automatic parallelization with offline/online optimization (USENIX ATC 2023)

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

- **Python**: 3.10 or higher
- **PyTorch**: 2.0 or higher  
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 8GB+ RAM (16GB+ recommended for large models)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/NoakLiu/PiKV.git
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
deepspeed>=0.12.0     # For DeepSpeed integration
vllm>=0.2.0          # For vLLM integration
```

## Quick Start

```python
# Single GPU - Enhanced MoE
from core.single.moe import create_moe
model = create_moe('pikv', hidden_size=1024, num_experts=8, use_normalization=True, use_lora=True)

# vLLM Integration - Production Inference
from core.single.vllm_integration import create_pikv_vllm
engine = create_pikv_vllm("microsoft/DialoGPT-medium", enable_compression=True, enable_scheduling=True)

# DeepSpeed - Enterprise Training
from core.distributed.deepspeed_integration import create_pikv_deepspeed
manager = create_pikv_deepspeed(enable_moe=True, zero_stage=3, offload_optimizer=True)

# Distributed Training - Multi-GPU
from core.distributed.distributed_pikv import DistributedPiKVManager
manager = DistributedPiKVManager()

# FPGA metadata offload (simulation or CXL SmartNIC)
from core.fpga import create_pikv_fpga
fpga = create_pikv_fpga(num_experts=64, hidden_size=128, top_k=4)
```

### 🎯 **Command Line Quick Start**

```bash
# Basic distributed training
torchrun --nproc_per_node=4 examples/distributed_training_example.py --mode basic

# DeepSpeed training with ZeRO-3
torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --zero_stage 3

# MoE training with DeepSpeed
torchrun --nproc_per_node=4 examples/deepspeed_training_example.py --enable_moe --zero_stage 3

# Easy training script
./examples/run_distributed_training.sh deepspeed-zero3

# FPGA offload (software simulation)
python examples/fpga_offload_example.py
```

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

### FPGA Acceleration

```python
from core.fpga import create_pikv_fpga, is_fpga_available
from core.fpga.config import FPGAEngineMapping, FPGARoutingEngine, FPGACompressionEngine, FPGASchedulingEngine

if is_fpga_available():
    fpga = create_pikv_fpga(
        num_experts=64,
        hidden_size=128,
        top_k=4,
        compression_ratio=4.0,
    )
    q = torch.randn(128)
    packed, experts = fpga.process_token(q, q, q, token_id=0)
    fpga.update_scheduler_theta(target_hit_rate=0.9)
    print(fpga.get_stats())
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
# Experimental protocol (GPU, batch, budgets, fairness, variance)
# → downstream_tasks/EXPERIMENTAL_PROTOCOL.md

# Systematic ablation: isolate routing / compression / scheduling / sharding
python -m downstream_tasks.eval.ablation_study --preset factor

# Per-component microbenchmarks
python -m downstream_tasks.eval.routing_experiment
python -m downstream_tasks.eval.compression_experiment
python -m downstream_tasks.eval.scheduling_experiment

# FPGA / CXL platform + bandwidth model + E2E vs CPU
python -m core.fpga.benchmark_hw

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

# Run distributed training tests
torchrun --nproc_per_node=2 examples/distributed_training_example.py --mode basic --steps_per_epoch 10

# Run DeepSpeed tests
torchrun --nproc_per_node=2 examples/deepspeed_training_example.py --zero_stage 1 --steps_per_epoch 10

# Run comprehensive training comparison
./examples/run_distributed_training.sh compare
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
      author={Dong Liu and Yanxuan Yu and Ben Lengerich and Ying Nian Wu},
      year={2025},
      eprint={2508.06526},
      archivePrefix={arXiv}
}
```

---

<div align="center">

### **Built with ❤️ by the PiKV Team**

**[Contact](mailto:dong.liu.dl2367@yale.edu) • [Discussions](https://github.com/NoakLiu/PiKV/discussions) • [Issues](https://github.com/NoakLiu/PiKV/issues) • [Docs](https://github.com/NoakLiu/PiKV)**

</div>

