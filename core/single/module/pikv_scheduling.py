"""
PiKV Scheduling Module

实现多种缓存调度策略，包括：
- H2OScheduler: Heavy Hitters Oracle调度
- StreamingLLMScheduler: 流式LLM调度
- QUESTScheduler: 质量感知调度
- FlexGenScheduler: 灵活生成调度
- LRUScheduler: 最近最少使用调度
- LRUPlusScheduler: 增强LRU调度
- CacheSchedulingManager: 统一调度管理器
- AdaKV: 自适应KV缓存调度
- DuoAttention: 双注意力调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum

class SchedulingPolicy(Enum):
    """缓存调度策略枚举"""
    NONE = "none"
    H2O = "h2o"
    STREAMING_LLM = "streaming_llm"
    QUEST = "quest"
    FLEXGEN = "flexgen"
    LRU = "lru"
    LRU_PLUS = "lru_plus"
    ADAKV = "adakv"
    DUOATTENTION = "duoattention"
    BELADY_APPROX = "belady_approx"
    HAZARD_LRU = "hazard_lru"
    TWO_QUEUE = "two_queue"

class BaseScheduler(nn.Module):
    """基础缓存调度器"""
    def __init__(self, cache_size: int, hidden_size: int):
        super(BaseScheduler, self).__init__()
        self.cache_size = cache_size
        self.hidden_size = hidden_size
        
        # 缓存统计信息
        self.register_buffer('hit_count', torch.tensor(0))
        self.register_buffer('miss_count', torch.tensor(0))
        self.register_buffer('eviction_count', torch.tensor(0))
        self.register_buffer('total_operations', torch.tensor(0))
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count.float() / total.float() if total > 0 else 0.0
    
    def get_eviction_rate(self) -> float:
        """获取淘汰率"""
        return self.eviction_count.float() / self.total_operations.float() if self.total_operations > 0 else 0.0
    
    def reset_stats(self):
        """重置统计信息"""
        self.hit_count.zero_()
        self.miss_count.zero_()
        self.eviction_count.zero_()
        self.total_operations.zero_()
    
    def update_stats(self, hit: bool = False, miss: bool = False, eviction: bool = False):
        """更新统计信息"""
        if hit:
            self.hit_count += 1
        if miss:
            self.miss_count += 1
        if eviction:
            self.eviction_count += 1
        self.total_operations += 1
    
    def should_evict(self, cache_usage: torch.Tensor, new_importance: torch.Tensor) -> torch.Tensor:
        """判断是否需要淘汰缓存项"""
        raise NotImplementedError
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """选择淘汰候选项"""
        raise NotImplementedError
    
    def get_scheduler_stats(self) -> Dict[str, float]:
        """获取调度器统计信息"""
        return {
            "hit_rate": float(self.get_hit_rate()),
            "eviction_rate": float(self.get_eviction_rate()),
            "total_hits": self.hit_count.item(),
            "total_misses": self.miss_count.item(),
            "total_evictions": self.eviction_count.item(),
            "total_operations": self.total_operations.item()
        }

class H2OScheduler(BaseScheduler):
    """
    H2O (Heavy Hitters Oracle) 调度器
    基于注意力权重的重要性进行缓存管理
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 heavy_ratio: float = 0.1, recent_ratio: float = 0.1,
                 attention_decay: float = 0.95):
        super(H2OScheduler, self).__init__(cache_size, hidden_size)
        self.heavy_ratio = heavy_ratio  # 重要token比例
        self.recent_ratio = recent_ratio  # 最近token比例
        self.attention_decay = attention_decay  # 注意力衰减因子
        
        # 注意力权重累积器
        self.register_buffer('attention_accumulator', torch.zeros(cache_size))
        self.register_buffer('access_timestamps', torch.zeros(cache_size))
        self.register_buffer('current_time', torch.tensor(0))
        
        # 动态阈值
        self.register_buffer('heavy_threshold', torch.tensor(0.1))
        self.register_buffer('threshold_history', torch.zeros(100))
        self.register_buffer('threshold_idx', torch.tensor(0))
    
    def update_attention_scores(self, indices: torch.Tensor, attention_weights: torch.Tensor):
        """更新注意力分数"""
        # 应用衰减到现有分数
        self.attention_accumulator *= self.attention_decay
        
        # 添加新的注意力权重
        self.attention_accumulator[indices] += attention_weights
        self.access_timestamps[indices] = self.current_time
        self.current_time += 1
        
        # 动态调整阈值
        self._update_threshold()
    
    def _update_threshold(self):
        """动态更新重要性阈值"""
        # 计算当前注意力分数的统计信息
        non_zero_scores = self.attention_accumulator[self.attention_accumulator > 0]
        if len(non_zero_scores) > 0:
            # 使用百分位数作为阈值
            threshold = torch.quantile(non_zero_scores, 1.0 - self.heavy_ratio)
            
            # 记录阈值历史
            idx = self.threshold_idx.item() % 100
            self.threshold_history[idx] = threshold
            self.threshold_idx += 1
            
            # 平滑更新阈值
            self.heavy_threshold = 0.9 * self.heavy_threshold + 0.1 * threshold
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """H2O淘汰策略：保留重要token和最近token"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 计算重要token数量
        num_heavy = max(1, int(cache_len * self.heavy_ratio))
        num_recent = max(1, int(cache_len * self.recent_ratio))
        
        # 获取重要token（基于累积注意力权重）
        valid_attention = self.attention_accumulator[:cache_len]
        _, heavy_indices = torch.topk(valid_attention, min(num_heavy, cache_len))
        
        # 获取最近token（基于时间戳）
        valid_timestamps = self.access_timestamps[:cache_len]
        _, recent_indices = torch.topk(valid_timestamps, min(num_recent, cache_len))
        
        # 合并保留的索引
        keep_indices = torch.cat([heavy_indices, recent_indices]).unique()
        
        # 创建淘汰掩码
        evict_mask = torch.ones(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[keep_indices] = False
        
        # 更新统计信息
        self.update_stats(eviction=evict_mask.sum().item() > 0)
        
        return evict_mask
    
    def get_scheduler_stats(self) -> Dict[str, float]:
        """获取H2O调度器统计信息"""
        stats = super().get_scheduler_stats()
        stats.update({
            "heavy_threshold": self.heavy_threshold.item(),
            "attention_decay": self.attention_decay,
            "heavy_ratio": self.heavy_ratio,
            "recent_ratio": self.recent_ratio,
            "current_time": self.current_time.item()
        })
        return stats

class StreamingLLMScheduler(BaseScheduler):
    """
    StreamingLLM 调度器
    保留初始token和最近的滑动窗口
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 start_size: int = 4, recent_ratio: float = 0.8,
                 adaptive_window: bool = True):
        super(StreamingLLMScheduler, self).__init__(cache_size, hidden_size)
        self.start_size = start_size  # 保留的初始token数量
        self.recent_ratio = recent_ratio  # 最近窗口比例
        self.adaptive_window = adaptive_window  # 是否自适应调整窗口
        
        # 计算滑动窗口大小
        self.recent_size = max(1, int((cache_size - start_size) * recent_ratio))
        
        self.register_buffer('position_counter', torch.tensor(0))
        self.register_buffer('window_utilization', torch.zeros(100))
        self.register_buffer('window_idx', torch.tensor(0))
        
        # 自适应窗口参数
        if adaptive_window:
            self.register_buffer('optimal_window_size', torch.tensor(self.recent_size))
            self.register_buffer('performance_history', torch.zeros(50))
            self.register_buffer('perf_idx', torch.tensor(0))
    
    def _adapt_window_size(self, performance_metric: float):
        """自适应调整窗口大小"""
        if not self.adaptive_window:
            return
        
        # 记录性能历史
        idx = self.perf_idx.item() % 50
        self.performance_history[idx] = performance_metric
        self.perf_idx += 1
        
        # 计算最近性能趋势
        if self.perf_idx > 10:
            recent_perf = self.performance_history[max(0, idx-9):idx+1].mean()
            
            # 如果性能下降，调整窗口大小
            if recent_perf < 0.8:  # 阈值可调
                # 增加窗口大小
                new_size = min(self.cache_size - self.start_size, 
                              int(self.optimal_window_size * 1.1))
            else:
                # 可能减少窗口大小以提高效率
                new_size = max(1, int(self.optimal_window_size * 0.95))
            
            # 平滑更新
            self.optimal_window_size = int(0.9 * int(self.optimal_window_size) + 0.1 * int(new_size))
            self.recent_size = int(self.optimal_window_size)
    
    def update_window_utilization(self, utilization: float):
        """更新窗口利用率"""
        idx = self.window_idx.item() % 100
        self.window_utilization[idx] = utilization
        self.window_idx += 1
        
        # 自适应调整窗口大小
        self._adapt_window_size(utilization)
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """StreamingLLM淘汰策略：保留开始token和最近窗口"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 创建淘汰掩码
        evict_mask = torch.ones(cache_len, dtype=torch.bool, device=keys.device)
        
        # 保留开始的token
        evict_mask[:self.start_size] = False
        
        # 保留最近的token
        recent_start = max(self.start_size, cache_len - self.recent_size)
        evict_mask[recent_start:] = False
        
        # 更新统计信息
        self.update_stats(eviction=evict_mask.sum().item() > 0)
        self.position_counter += 1
        
        return evict_mask
    
    def get_scheduler_stats(self) -> Dict[str, float]:
        """获取StreamingLLM调度器统计信息"""
        stats = super().get_scheduler_stats()
        
        # 计算平均窗口利用率
        valid_entries = min(int(self.window_idx.item()), 100)
        avg_utilization = 0.0
        if valid_entries > 0:
            avg_utilization = self.window_utilization[:valid_entries].mean().item()
        
        stats.update({
            "start_size": self.start_size,
            "recent_size": self.recent_size,
            "recent_ratio": self.recent_ratio,
            "position_counter": int(self.position_counter),
            "avg_window_utilization": avg_utilization,
            "adaptive_window": self.adaptive_window
        })
        
        if self.adaptive_window:
            stats["optimal_window_size"] = int(self.optimal_window_size)
        
        return stats

class QUESTScheduler(BaseScheduler):
    """
    QUEST (Quality-aware Eviction with Streaming) 调度器
    基于质量感知的流式淘汰策略
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 quality_threshold: float = 0.1, decay_factor: float = 0.95,
                 importance_weight: float = 0.3, frequency_weight: float = 0.3):
        super(QUESTScheduler, self).__init__(cache_size, hidden_size)
        self.quality_threshold = quality_threshold
        self.decay_factor = decay_factor
        self.importance_weight = importance_weight
        self.frequency_weight = frequency_weight
        self.time_weight = 1.0 - importance_weight - frequency_weight
        
        # 质量分数和使用频率
        self.register_buffer('quality_scores', torch.zeros(cache_size))
        self.register_buffer('usage_frequency', torch.zeros(cache_size))
        self.register_buffer('importance_scores', torch.zeros(cache_size))
        self.register_buffer('last_access_time', torch.zeros(cache_size))
        self.register_buffer('global_time', torch.tensor(0))
        
        # 质量预测网络
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def update_quality_scores(self, indices: torch.Tensor, 
                            keys: torch.Tensor, values: torch.Tensor,
                            importance: Optional[torch.Tensor] = None):
        """更新质量分数"""
        # 预测质量分数
        combined_features = torch.cat([keys, values], dim=-1)  # [num_items, hidden_size*2]
        
        # 如果特征维度不匹配，使用平均池化
        if combined_features.size(-1) != self.quality_predictor[0].in_features:
            # 使用keys作为特征
            predicted_quality = self.quality_predictor(keys)
        else:
            # 先降维到正确大小
            feature_proj = nn.Linear(combined_features.size(-1), 
                                   self.quality_predictor[0].in_features, 
                                   device=combined_features.device)
            projected_features = feature_proj(combined_features)
            predicted_quality = self.quality_predictor(projected_features)
        
        # 更新质量分数
        self.quality_scores[indices] = predicted_quality.squeeze(-1)
        
        # 更新重要性分数
        if importance is not None:
            if importance.dim() > 1:
                importance = importance.mean(dim=-1)
            self.importance_scores[indices] = importance
        
        # 更新访问信息
        self.usage_frequency[indices] += 1
        self.last_access_time[indices] = self.global_time
        self.global_time += 1
    
    def compute_utility_scores(self, cache_len: int) -> torch.Tensor:
        """计算效用分数"""
        # 时间衰减因子
        time_decay = torch.pow(self.decay_factor, 
                              self.global_time - self.last_access_time[:cache_len])
        
        # 归一化各个组件
        quality_norm = F.normalize(self.quality_scores[:cache_len], dim=0)
        frequency_norm = F.normalize(self.usage_frequency[:cache_len], dim=0)
        importance_norm = F.normalize(self.importance_scores[:cache_len], dim=0)
        
        # 综合效用分数
        utility = (self.importance_weight * importance_norm + 
                  self.frequency_weight * frequency_norm + 
                  self.time_weight * quality_norm) * time_decay
        
        return utility
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """QUEST淘汰策略：基于效用分数"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 计算效用分数
        utility_scores = self.compute_utility_scores(cache_len)
        
        # 淘汰效用分数低于阈值的项
        evict_mask = utility_scores < self.quality_threshold
        
        # 如果需要淘汰的项太少，选择效用最低的项
        num_to_evict = cache_len - self.cache_size
        if evict_mask.sum() < num_to_evict:
            _, lowest_indices = torch.topk(utility_scores, num_to_evict, largest=False)
            evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
            evict_mask[lowest_indices] = True
        elif evict_mask.sum() > num_to_evict:
            # 如果要淘汰的太多，选择效用最低的前num_to_evict项
            low_utility_indices = torch.where(evict_mask)[0]
            low_utility_scores = utility_scores[low_utility_indices]
            _, relative_indices = torch.topk(low_utility_scores, num_to_evict, largest=False)
            keep_indices = low_utility_indices[relative_indices]
            
            evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
            evict_mask[keep_indices] = True
        
        # 更新统计信息
        self.update_stats(eviction=evict_mask.sum().item() > 0)
        
        return evict_mask

class FlexGenScheduler(BaseScheduler):
    """
    FlexGen调度器
    针对GPU/CPU层次存储的优化调度
    """
    def __init__(self, cache_size: int, hidden_size: int,
                 gpu_ratio: float = 0.3, cpu_ratio: float = 0.5,
                 offload_threshold: float = 0.8):
        super(FlexGenScheduler, self).__init__(cache_size, hidden_size)
        self.gpu_ratio = gpu_ratio  # GPU缓存比例
        self.cpu_ratio = cpu_ratio  # CPU缓存比例
        self.offload_threshold = offload_threshold  # 卸载阈值
        
        # 层次存储大小
        self.gpu_cache_size = int(cache_size * gpu_ratio)
        self.cpu_cache_size = int(cache_size * cpu_ratio)
        self.storage_size = cache_size - self.gpu_cache_size - self.cpu_cache_size
        
        # 访问成本追踪
        self.register_buffer('access_costs', torch.zeros(cache_size))
        self.register_buffer('storage_levels', torch.zeros(cache_size))  # 0: GPU, 1: CPU, 2: Storage
        self.register_buffer('migration_count', torch.tensor(0))
    
    def update_access_patterns(self, indices: torch.Tensor, access_costs: torch.Tensor):
        """更新访问模式和成本"""
        self.access_costs[indices] = access_costs
    
    def _compute_migration_benefit(self, cache_len: int) -> torch.Tensor:
        """计算迁移收益"""
        # 访问频率（最近访问的权重更高）
        access_freq = torch.zeros(cache_len, device=self.access_costs.device)
        
        # 根据访问成本计算收益
        migration_benefit = 1.0 / (self.access_costs[:cache_len] + 1e-8)
        
        return migration_benefit
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FlexGen淘汰策略：基于存储层次和访问成本"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 计算迁移收益
        migration_benefit = self._compute_migration_benefit(cache_len)
        
        # 基于收益选择淘汰候选
        num_to_evict = cache_len - self.cache_size
        _, evict_indices = torch.topk(migration_benefit, num_to_evict, largest=False)
        
        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[evict_indices] = True
        
        # 更新统计信息
        self.update_stats(eviction=evict_mask.sum().item() > 0)
        self.migration_count += evict_mask.sum()
        
        return evict_mask
    
    def get_scheduler_stats(self) -> Dict[str, float]:
        """获取FlexGen调度器统计信息"""
        stats = super().get_scheduler_stats()
        stats.update({
            "gpu_ratio": self.gpu_ratio,
            "cpu_ratio": self.cpu_ratio,
            "gpu_cache_size": self.gpu_cache_size,
            "cpu_cache_size": self.cpu_cache_size,
            "storage_size": self.storage_size,
            "migration_count": self.migration_count.item(),
            "offload_threshold": self.offload_threshold
        })
        return stats

class LRUScheduler(BaseScheduler):
    """
    LRU (Least Recently Used) 调度器
    传统的最近最少使用调度策略
    """
    def __init__(self, cache_size: int, hidden_size: int):
        super(LRUScheduler, self).__init__(cache_size, hidden_size)
        
        self.register_buffer('access_times', torch.zeros(cache_size))
        self.register_buffer('global_time', torch.tensor(0))
    
    def update_access_time(self, indices: torch.Tensor):
        """更新访问时间"""
        self.access_times[indices] = self.global_time
        self.global_time += 1
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LRU淘汰策略：淘汰最久未使用的项"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 获取最久未使用的项
        num_to_evict = cache_len - self.cache_size
        _, evict_indices = torch.topk(self.access_times[:cache_len], 
                                     num_to_evict, largest=False)
        
        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[evict_indices] = True
        
        # 更新统计信息
        self.update_stats(eviction=evict_mask.sum().item() > 0)
        
        return evict_mask

class LRUPlusScheduler(BaseScheduler):
    """
    LRU+ 调度器
    增强的LRU，结合频率和重要性
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 frequency_weight: float = 0.3, importance_weight: float = 0.4,
                 time_weight: float = 0.3):
        super(LRUPlusScheduler, self).__init__(cache_size, hidden_size)
        self.frequency_weight = frequency_weight
        self.importance_weight = importance_weight
        self.time_weight = time_weight
        
        # 确保权重和为1
        total_weight = frequency_weight + importance_weight + time_weight
        self.frequency_weight /= total_weight
        self.importance_weight /= total_weight
        self.time_weight /= total_weight
        
        self.register_buffer('access_times', torch.zeros(cache_size))
        self.register_buffer('access_frequency', torch.zeros(cache_size))
        self.register_buffer('importance_scores', torch.zeros(cache_size))
        self.register_buffer('global_time', torch.tensor(0))
    
    def update_access_info(self, indices: torch.Tensor, importance: torch.Tensor):
        """更新访问信息"""
        self.access_times[indices] = self.global_time
        self.access_frequency[indices] += 1
        
        if importance.dim() > 1:
            importance = importance.mean(dim=-1)
        self.importance_scores[indices] = importance
        
        self.global_time += 1
    
    def compute_priority_scores(self, cache_len: int) -> torch.Tensor:
        """计算优先级分数"""
        # 归一化各个组件
        valid_times = self.access_times[:cache_len]
        valid_freq = self.access_frequency[:cache_len]
        valid_imp = self.importance_scores[:cache_len]
        
        # 时间组件：越新越好
        time_scores = valid_times / (self.global_time + 1e-8)
        
        # 频率组件：归一化
        freq_scores = valid_freq / (valid_freq.max() + 1e-8)
        
        # 重要性组件：归一化
        imp_scores = valid_imp / (valid_imp.max() + 1e-8)
        
        # 综合优先级分数
        priority_scores = (self.time_weight * time_scores + 
                          self.frequency_weight * freq_scores + 
                          self.importance_weight * imp_scores)
        
        return priority_scores
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LRU+淘汰策略：基于综合优先级分数"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 计算优先级分数
        priority_scores = self.compute_priority_scores(cache_len)
        
        # 淘汰优先级最低的项
        num_to_evict = cache_len - self.cache_size
        _, evict_indices = torch.topk(priority_scores, num_to_evict, largest=False)
        
        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[evict_indices] = True
        
        # 更新统计信息
        self.update_stats(eviction=evict_mask.sum().item() > 0)
        
        return evict_mask


class AdaKVScheduler(BaseScheduler):
    """
    AdaKVScheduler：选择性保留注意力重要的 token（基于线性打分）
    """
    def __init__(self, cache_size: int, hidden_size: int,
                 compression_ratio: float = 0.5, safeguard_ratio: float = 0.2):
        super().__init__(cache_size, hidden_size)
        self.compression_ratio = compression_ratio
        self.safeguard_ratio = safeguard_ratio
        self.register_buffer('importance_scores', torch.zeros(cache_size))

        # 简单的线性重要性评分器
        self.linear_proj = nn.Linear(hidden_size, 1)

    def update_importance(self, indices: torch.Tensor, keys: torch.Tensor):
        """更新重要性分数"""
        scores = self.linear_proj(keys).squeeze(-1)
        self.importance_scores[indices] = scores

    def select_eviction_candidates(self, keys, values, metadata) -> torch.Tensor:
        cache_len = keys.size(0)
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)

        scores = self.importance_scores[:cache_len]
        num_keep = int(cache_len * (1 - self.compression_ratio))
        num_safeguard = int(num_keep * self.safeguard_ratio)

        # 保护最重要的一小部分 token 不被淘汰
        _, safe_indices = torch.topk(scores, num_safeguard)

        # 再根据剩余分数选出其余保留项
        scores_clone = scores.clone()
        scores_clone[safe_indices] = float('-inf')  # 避免重复选中
        _, keep_indices = torch.topk(scores_clone, num_keep - num_safeguard)

        all_keep = torch.cat([safe_indices, keep_indices]).unique()
        evict_mask = torch.ones(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[all_keep] = False

        self.update_stats(eviction=bool(evict_mask.any().item()))
        return evict_mask

    def get_scheduler_stats(self) -> Dict[str, float]:
        stats = super().get_scheduler_stats()
        stats.update({
            "compression_ratio": self.compression_ratio,
            "safeguard_ratio": self.safeguard_ratio
        })
        return stats

class DuoAttentionScheduler(BaseScheduler):
    """
    DuoAttentionScheduler：基于序列位置进行滑动窗口缓存
    - Sink tokens（开头）保留
    - Recent tokens（结尾）保留
    - 中间 token 淘汰
    """
    def __init__(self, cache_size: int, hidden_size: int,
                 sink_size: int = 64, recent_size: int = 512):
        super().__init__(cache_size, hidden_size)
        self.sink_size = sink_size
        self.recent_size = recent_size

    def select_eviction_candidates(self, keys, values, metadata) -> torch.Tensor:
        cache_len = keys.size(0)
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)

        evict_mask = torch.ones(cache_len, dtype=torch.bool, device=keys.device)

        # 保留 sink tokens
        evict_mask[:self.sink_size] = False

        # 保留 recent tokens
        evict_mask[-self.recent_size:] = False

        self.update_stats(eviction=bool(evict_mask.any().item()))
        return evict_mask

    def get_scheduler_stats(self) -> Dict[str, float]:
        stats = super().get_scheduler_stats()
        stats.update({
            "sink_size": self.sink_size,
            "recent_size": self.recent_size
        })
        return stats

class CacheSchedulingManager(nn.Module):
    """
    统一的缓存调度管理器
    管理不同的调度策略并提供统一接口
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 policy: SchedulingPolicy = SchedulingPolicy.NONE):
        super(CacheSchedulingManager, self).__init__()
        self.cache_size = cache_size
        self.hidden_size = hidden_size
        self.current_policy = policy
        
        # 缓存存储
        self.register_buffer('cache_keys', torch.zeros(cache_size, hidden_size))
        self.register_buffer('cache_values', torch.zeros(cache_size, hidden_size))
        self.register_buffer('cache_valid', torch.zeros(cache_size, dtype=torch.bool))
        self.register_buffer('cache_size_current', torch.tensor(0))
        
        # 创建调度器
        self.scheduler = self._create_scheduler(policy)
        
        # 管理统计信息
        self.register_buffer('total_updates', torch.tensor(0))
        self.register_buffer('total_evictions', torch.tensor(0))
        self.register_buffer('policy_switches', torch.tensor(0))
    
    def _create_scheduler(self, policy: SchedulingPolicy) -> Optional[BaseScheduler]:
        """创建指定策略的调度器"""
        if policy == SchedulingPolicy.NONE:
            return None
        elif policy == SchedulingPolicy.H2O:
            return H2OScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.STREAMING_LLM:
            return StreamingLLMScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.QUEST:
            return QUESTScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.FLEXGEN:
            return FlexGenScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.LRU:
            return LRUScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.LRU_PLUS:
            return LRUPlusScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.ADAKV:
            return AdaKVScheduler(self.cache_size, self.hidden_size,
                                compression_ratio=0.5, safeguard_ratio=0.2)

        elif policy == SchedulingPolicy.DUOATTENTION:
            return DuoAttentionScheduler(self.cache_size, self.hidden_size,
                                        sink_size=64, recent_size=512)
        elif policy == SchedulingPolicy.BELADY_APPROX:
            from ..schedulers.belady_approx import BeladyApproxScheduler
            return BeladyApproxScheduler(self.cache_size, self.hidden_size,
                                         alpha=0.6, pos_decay=0.97)
        elif policy == SchedulingPolicy.HAZARD_LRU:
            from ..schedulers.hazard_lru import HazardLRUScheduler
            return HazardLRUScheduler(self.cache_size, self.hidden_size,
                                      alpha=0.5, beta=0.3, gamma=0.2)
        elif policy == SchedulingPolicy.TWO_QUEUE:
            from ..schedulers.two_queue import TwoQueueScheduler
            return TwoQueueScheduler(self.cache_size, self.hidden_size,
                                     admission_tau=0.5, demote_cooldown=16, cpu_ratio=0.5)
        else:
            return None
    
    def update_cache(self, keys: torch.Tensor, values: torch.Tensor, 
                    metadata: Optional[Dict[str, torch.Tensor]] = None):
        """更新缓存"""
        batch_size, seq_len, hidden_size = keys.shape
        
        if metadata is None:
            metadata = {}
        
        # remove invalid pre-loop references to idx/key; handled inside per-item loop

        
        # 确保输入在正确设备上
        keys = keys.to(self.cache_keys.device)
        values = values.to(self.cache_values.device)
        
        # 逐个处理批次中的每个项
        for b in range(batch_size):
            for s in range(seq_len):
                key = keys[b, s]
                value = values[b, s]
                
                # 检查是否需要淘汰
                if self.cache_size_current >= self.cache_size:
                    if self.scheduler is not None:
                        # 使用调度器选择淘汰候选
                        current_keys = self.cache_keys[:self.cache_size_current]
                        current_values = self.cache_values[:self.cache_size_current]
                        
                        evict_mask = self.scheduler.select_eviction_candidates(
                            current_keys.unsqueeze(0),
                            current_values.unsqueeze(0),
                            metadata
                        ).squeeze(0)
                        
                        if evict_mask.any():
                            # 执行淘汰
                            keep_mask = ~evict_mask
                            num_keep = keep_mask.sum().item()
                            
                            if num_keep > 0:
                                self.cache_keys[:num_keep] = self.cache_keys[:self.cache_size_current][keep_mask]
                                self.cache_values[:num_keep] = self.cache_values[:self.cache_size_current][keep_mask]
                                self.cache_valid[:num_keep] = True
                                self.cache_valid[num_keep:] = False
                                self.cache_size_current = num_keep
                            else:
                                # 全部淘汰
                                self.cache_valid.zero_()
                                self.cache_size_current = 0
                            
                            self.total_evictions += int(evict_mask.sum().item())
                    else:
                        # 简单FIFO淘汰
                        if int(self.cache_size_current.item()) > 0:
                            self.cache_keys[:-1] = self.cache_keys[1:self.cache_size_current]
                            self.cache_values[:-1] = self.cache_values[1:self.cache_size_current]
                            self.cache_size_current = int(self.cache_size_current.item()) - 1
                
                # 添加新项
                if int(self.cache_size_current.item()) < int(self.cache_size):
                    idx = int(self.cache_size_current.item())
                    self.cache_keys[idx] = key
                    self.cache_values[idx] = value
                    self.cache_valid[idx] = True
                    self.cache_size_current += 1
                    
                    # 更新调度器特定信息
                    if self.scheduler is not None:
                        if hasattr(self.scheduler, 'update_access_time'):
                            self.scheduler.update_access_time(torch.tensor([idx], device=self.cache_keys.device))
                        elif hasattr(self.scheduler, 'update_access_info'):
                            importance = metadata.get('importance', torch.tensor([0.5], device=self.cache_keys.device))
                            if importance.dim() == 0:
                                importance = importance.unsqueeze(0)
                            self.scheduler.update_access_info(torch.tensor([idx], device=self.cache_keys.device), importance)
        
        self.total_updates += batch_size * seq_len
    
    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        stats = {
            "cache_utilization": float(self.cache_size_current) / float(self.cache_size),
            "policy": self.current_policy.value,
            "total_updates": self.total_updates.item(),
            "total_evictions": self.total_evictions.item(),
            "policy_switches": self.policy_switches.item(),
            "cache_size": int(self.cache_size),
            "current_size": int(self.cache_size_current.item())
        }
        
        # 添加调度器特定统计信息
        if self.scheduler is not None:
            scheduler_stats = self.scheduler.get_scheduler_stats()
            stats.update({f"scheduler_{k}": v for k, v in scheduler_stats.items()})
        
        return stats
    
    def reset_cache(self):
        """重置缓存"""
        self.cache_keys.zero_()
        self.cache_values.zero_()
        self.cache_valid.zero_()
        self.cache_size_current = torch.tensor(0, device=self.cache_size_current.device)
        
        if self.scheduler is not None:
            self.scheduler.reset_stats()
    
    def change_policy(self, new_policy: SchedulingPolicy):
        """更改调度策略"""
        if new_policy != self.current_policy:
            self.current_policy = new_policy
            self.scheduler = self._create_scheduler(new_policy)
            self.policy_switches += 1
            
            # 重置缓存以避免状态不一致
            self.reset_cache()
    
    def get_cached_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前缓存的数据"""
        if self.cache_size_current > 0:
            valid_keys = self.cache_keys[:self.cache_size_current]
            valid_values = self.cache_values[:self.cache_size_current]
            return valid_keys, valid_values
        else:
            return torch.empty(0, self.hidden_size, device=self.cache_keys.device), \
                   torch.empty(0, self.hidden_size, device=self.cache_values.device) 