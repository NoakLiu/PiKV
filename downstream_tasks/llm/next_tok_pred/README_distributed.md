# PiKV 分布式训练指南

本文档介绍如何使用多GPU进行PiKV模型的分布式训练。

## 概述

根据[PyTorch分布式训练教程](https://docs.pytorch.org/tutorials/distributed/home.html)，我们实现了基于DistributedDataParallel (DDP)的分布式训练系统，可以在多个GPU上并行训练PiKV模型。

## 特性

- **真正的多GPU分布式训练**：使用PyTorch DDP在多个GPU上并行训练
- **调用现有的DistributedPiKVMoE**：使用项目中已定义的分布式PiKV实现
- **支持torchrun和mp.spawn**：提供两种启动方式
- **自动数据分片**：使用DistributedSampler自动分配数据到不同GPU
- **同步训练**：所有GPU同步进行梯度更新
- **检查点保存**：只在rank 0进程保存模型检查点

## 文件结构

```
downstream_tasks/llm/next_tok_pred/
├── train_distributed.py      # 分布式训练主脚本
├── (moved) ../../../scripts/run_ntp_distributed.sh
└── README_distributed.md     # 本文档
```

## 使用方法

### 方法1：使用torchrun（推荐）

这是PyTorch官方推荐的分布式训练启动方式：

```bash
# 从仓库根目录
./scripts/run_ntp_distributed.sh 10 5 pikv

# 或者直接使用torchrun
cd downstream_tasks/llm/next_tok_pred
torchrun --nproc_per_node=4 train_distributed.py --epochs 10 --save_every 5 --model_type pikv
```

### 方法2：使用mp.spawn

```bash
python train_distributed.py --epochs 10 --save_every 5 --model_type pikv --use_spawn
```

## 参数说明

- `--epochs`: 训练轮数（默认：10）
- `--save_every`: 每N个epoch保存一次检查点（默认：5）
- `--model_type`: 模型类型，可选 `pikv` 或 `standard`（默认：pikv）
- `--use_spawn`: 使用mp.spawn而不是torchrun（不推荐）

## 模型架构

### PiKVLanguageModel

我们创建了一个语言模型包装器，它包含：

1. **Embedding层**：将token IDs转换为embeddings
2. **DistributedPiKVMoE**：使用项目中定义的分布式PiKV MoE实现
3. **Vocabulary投影层**：将隐藏状态投影到词汇表大小

### DistributedPiKVMoE特性

- **分布式专家**：每个专家在多个GPU上分布
- **KV缓存压缩**：使用压缩技术减少内存使用
- **自适应路由**：智能选择最相关的专家
- **负载均衡**：确保专家使用均衡

## 系统要求

- **多GPU环境**：至少2个GPU（推荐4个或更多）
- **CUDA支持**：需要CUDA环境
- **PyTorch >= 1.9.0**：支持DDP的PyTorch版本
- **NCCL后端**：用于GPU间通信

## 训练流程

1. **初始化分布式环境**：设置进程组和GPU设备
2. **数据加载**：使用DistributedSampler分配数据
3. **模型包装**：用DDP包装模型
4. **同步训练**：所有GPU同步进行前向和反向传播
5. **检查点保存**：只在rank 0保存模型状态

## 性能优化

### 数据加载优化
- 使用`pin_memory=True`加速GPU传输
- 使用DistributedSampler确保数据均匀分布
- 每个epoch调用`set_epoch()`确保数据随机性

### 内存优化
- 使用梯度累积减少内存使用
- KV缓存压缩减少存储需求
- 适当的批次大小平衡内存和性能

### 通信优化
- 使用NCCL后端优化GPU间通信
- 同步批归一化（如果使用）
- 减少不必要的同步操作

## 监控和调试

### 日志输出
每个GPU进程会输出：
```
[GPU0] Epoch 1 | Batchsize: 4 | Steps: 10
[GPU0] Epoch 1 | Average Loss: 10.5234
```

### 检查点管理
检查点保存在`checkpoints/`目录：
```
checkpoints/
├── distributed_pikv_epoch_0.pt
├── distributed_pikv_epoch_5.pt
└── distributed_pikv_epoch_9.pt
```

### 常见问题

1. **NCCL错误**：确保所有GPU可见且CUDA版本兼容
2. **内存不足**：减少批次大小或使用梯度累积
3. **进程挂起**：检查防火墙设置和端口可用性

## 扩展到多节点

要扩展到多个节点，修改torchrun参数：

```bash
# 节点0（主节点）
torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=29500 --nproc_per_node=4 train_distributed.py

# 节点1
torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 --master_port=29500 --nproc_per_node=4 train_distributed.py
```

## 参考资料

- [PyTorch分布式训练教程](https://docs.pytorch.org/tutorials/distributed/home.html)
- [DDP最佳实践](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [多GPU训练指南](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html)

## 示例输出

```bash
$ ./scripts/run_ntp_distributed.sh 5 2 pikv
Detected 4 GPUs
Starting distributed training with:
  - GPUs: 4
  - Epochs: 5
  - Save every: 2 epochs
  - Model type: pikv
Creating training data...
Launching distributed training...
Running under torchrun...
[Rank 0] Starting training on GPU 0
[Rank 1] Starting training on GPU 1
[Rank 2] Starting training on GPU 2
[Rank 3] Starting training on GPU 3
[GPU0] Epoch 0 | Batchsize: 4 | Steps: 3
[GPU0] Epoch 0 | Average Loss: 10.8234
[GPU0] Checkpoint saved: checkpoints/distributed_pikv_epoch_0.pt
...
Distributed training completed!
Checkpoints saved in:
-rw-r--r-- 1 user user 1.2M distributed_pikv_epoch_0.pt
-rw-r--r-- 1 user user 1.2M distributed_pikv_epoch_2.pt
-rw-r--r-- 1 user user 1.2M distributed_pikv_epoch_4.pt
``` 