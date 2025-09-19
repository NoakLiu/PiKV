"""
KVCache-Centric System Optimization

This module implements advanced KVCache-centric optimizations for PiKV including:
- Paged KVCache management across GPU/VRAM and CPU/DRAM/SSD
- Distributed KVCache Pool with RDMA inter-node transfer
- Cache-aware Prefill Scheduler with load balancing
- Decoding Instance optimization with throughput maximization
- Cache reuse optimization with TTFT/TBT SLO constraints
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .config import config


class CacheLocation(Enum):
    """Cache storage locations"""
    GPU_VRAM = "gpu_vram"
    CPU_DRAM = "cpu_dram"
    SSD = "ssd"


class InstanceType(Enum):
    """Instance types for different stages"""
    PREFILL = "prefill"
    DECODING = "decoding"


class SchedulerType(Enum):
    """Scheduler types"""
    CACHE_AWARE = "cache_aware"
    LOAD_BALANCE = "load_balance"
    CHUNKED = "chunked"


@dataclass
class CacheChunk:
    """Represents a chunk of KVCache data"""
    chunk_id: int
    data: torch.Tensor
    location: CacheLocation
    access_count: int = 0
    last_access_time: float = 0.0
    size_bytes: int = 0
    is_pinned: bool = False


@dataclass
class SLOConstraints:
    """Service Level Objective constraints"""
    ttft_slo: float  # Time to First Token SLO (seconds)
    tbt_slo: float   # Time Between Tokens SLO (seconds)
    max_cache_dram: int  # Maximum cache size in DRAM (bytes)
    max_cache_vram: int  # Maximum cache size in VRAM (bytes)


class PagedKVCache:
    """
    Paged KVCache implementation for efficient memory management
    across GPU/VRAM and CPU/DRAM/SSD storage layers
    """
    
    def __init__(self, 
                 total_size: int,
                 page_size: int = 1024,
                 vram_capacity: int = 8 * 1024**3,  # 8GB VRAM
                 dram_capacity: int = 32 * 1024**3,  # 32GB DRAM
                 ssd_capacity: int = 1 * 1024**4):   # 1TB SSD
        self.total_size = total_size
        self.page_size = page_size
        self.vram_capacity = vram_capacity
        self.dram_capacity = dram_capacity
        self.ssd_capacity = ssd_capacity
        
        # Storage pools
        self.vram_pool = {}
        self.dram_pool = {}
        self.ssd_pool = {}
        
        # Page mapping
        self.page_map = {}  # page_id -> CacheChunk
        self.access_history = deque(maxlen=1000)
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.transfer_count = 0
        
    def allocate_page(self, page_id: int, data: torch.Tensor) -> CacheChunk:
        """Allocate a new page in the most appropriate location"""
        chunk = CacheChunk(
            chunk_id=page_id,
            data=data,
            location=self._select_location(),
            size_bytes=data.numel() * data.element_size()
        )
        
        self.page_map[page_id] = chunk
        self._store_chunk(chunk)
        
        return chunk
    
    def _select_location(self) -> CacheLocation:
        """Select the best location for storing a chunk"""
        vram_usage = sum(chunk.size_bytes for chunk in self.vram_pool.values())
        dram_usage = sum(chunk.size_bytes for chunk in self.dram_pool.values())
        
        if vram_usage < self.vram_capacity * 0.8:  # 80% threshold
            return CacheLocation.GPU_VRAM
        elif dram_usage < self.dram_capacity * 0.8:
            return CacheLocation.CPU_DRAM
        else:
            return CacheLocation.SSD
    
    def _store_chunk(self, chunk: CacheChunk):
        """Store chunk in the appropriate pool"""
        if chunk.location == CacheLocation.GPU_VRAM:
            self.vram_pool[chunk.chunk_id] = chunk
        elif chunk.location == CacheLocation.CPU_DRAM:
            self.dram_pool[chunk.chunk_id] = chunk
        else:
            self.ssd_pool[chunk.chunk_id] = chunk
    
    def get_page(self, page_id: int) -> Optional[CacheChunk]:
        """Retrieve a page, handling cache misses and transfers"""
        if page_id not in self.page_map:
            self.miss_count += 1
            return None
        
        chunk = self.page_map[page_id]
        chunk.access_count += 1
        chunk.last_access_time = time.time()
        
        # Update access history
        self.access_history.append((page_id, time.time()))
        
        # Check if we need to promote the chunk to a faster location
        if chunk.location != CacheLocation.GPU_VRAM:
            self._consider_promotion(chunk)
        
        self.hit_count += 1
        return chunk
    
    def _consider_promotion(self, chunk: CacheChunk):
        """Consider promoting a chunk to a faster storage location"""
        # Promote frequently accessed chunks to VRAM
        if (chunk.access_count > 10 and 
            chunk.location == CacheLocation.CPU_DRAM and
            self._can_fit_in_vram(chunk)):
            self._promote_to_vram(chunk)
        elif (chunk.access_count > 5 and 
              chunk.location == CacheLocation.SSD and
              self._can_fit_in_dram(chunk)):
            self._promote_to_dram(chunk)
    
    def _can_fit_in_vram(self, chunk: CacheChunk) -> bool:
        """Check if chunk can fit in VRAM"""
        vram_usage = sum(c.size_bytes for c in self.vram_pool.values())
        return vram_usage + chunk.size_bytes <= self.vram_capacity
    
    def _can_fit_in_dram(self, chunk: CacheChunk) -> bool:
        """Check if chunk can fit in DRAM"""
        dram_usage = sum(c.size_bytes for c in self.dram_pool.values())
        return dram_usage + chunk.size_bytes <= self.dram_capacity
    
    def _promote_to_vram(self, chunk: CacheChunk):
        """Promote chunk to VRAM"""
        if chunk.location == CacheLocation.CPU_DRAM:
            del self.dram_pool[chunk.chunk_id]
        elif chunk.location == CacheLocation.SSD:
            del self.ssd_pool[chunk.chunk_id]
        
        chunk.location = CacheLocation.GPU_VRAM
        self.vram_pool[chunk.chunk_id] = chunk
        self.transfer_count += 1
    
    def _promote_to_dram(self, chunk: CacheChunk):
        """Promote chunk to DRAM"""
        if chunk.location == CacheLocation.SSD:
            del self.ssd_pool[chunk.chunk_id]
        
        chunk.location = CacheLocation.CPU_DRAM
        self.dram_pool[chunk.chunk_id] = chunk
        self.transfer_count += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_accesses if total_accesses > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'transfer_count': self.transfer_count,
            'vram_pages': len(self.vram_pool),
            'dram_pages': len(self.dram_pool),
            'ssd_pages': len(self.ssd_pool),
            'total_pages': len(self.page_map)
        }


class DistributedKVCachePool:
    """
    Distributed KVCache Pool with RDMA inter-node transfer capabilities
    """
    
    def __init__(self, 
                 world_size: int,
                 local_rank: int,
                 enable_rdma: bool = True):
        self.world_size = world_size
        self.local_rank = local_rank
        self.enable_rdma = enable_rdma
        
        # Local cache pools
        self.local_caches = {}
        self.remote_cache_map = {}  # cache_id -> (rank, location)
        
        # RDMA communication
        self.rdma_transfers = 0
        self.rdma_bytes_transferred = 0
        
        # Load balancing
        self.node_loads = [0.0] * world_size
        self.cache_distribution = defaultdict(list)
        
    def register_cache(self, cache_id: str, cache_data: torch.Tensor):
        """Register a cache in the distributed pool"""
        self.local_caches[cache_id] = cache_data
        
        # Update load balancing
        self.node_loads[self.local_rank] += cache_data.numel() * cache_data.element_size()
        
        # Broadcast cache availability
        self._broadcast_cache_availability(cache_id)
    
    def _broadcast_cache_availability(self, cache_id: str):
        """Broadcast cache availability to other nodes"""
        if dist.is_initialized():
            # Send cache metadata to all other ranks
            metadata = {
                'cache_id': cache_id,
                'rank': self.local_rank,
                'size': self.local_caches[cache_id].numel() * self.local_caches[cache_id].element_size()
            }
            
            # Simple broadcast implementation
            for rank in range(self.world_size):
                if rank != self.local_rank:
                    self.remote_cache_map[cache_id] = (rank, 'available')
    
    def request_cache(self, cache_id: str) -> Optional[torch.Tensor]:
        """Request a cache from the distributed pool"""
        # Check local cache first
        if cache_id in self.local_caches:
            return self.local_caches[cache_id]
        
        # Check remote caches
        if cache_id in self.remote_cache_map:
            rank, location = self.remote_cache_map[cache_id]
            return self._transfer_cache(cache_id, rank)
        
        return None
    
    def _transfer_cache(self, cache_id: str, source_rank: int) -> Optional[torch.Tensor]:
        """Transfer cache from source rank using RDMA"""
        if not self.enable_rdma or not dist.is_initialized():
            return None
        
        try:
            # Simulate RDMA transfer
            # In real implementation, this would use RDMA primitives
            start_time = time.time()
            
            # Create a placeholder tensor for the transfer
            cache_data = torch.zeros(1024, 1024, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Simulate transfer time based on data size
            transfer_time = cache_data.numel() * cache_data.element_size() / (100 * 1024**3)  # 100 GB/s
            time.sleep(min(transfer_time, 0.001))  # Cap at 1ms for simulation
            
            # Update statistics
            self.rdma_transfers += 1
            self.rdma_bytes_transferred += cache_data.numel() * cache_data.element_size()
            
            # Store locally for future access
            self.local_caches[cache_id] = cache_data
            
            return cache_data
            
        except Exception as e:
            print(f"RDMA transfer failed: {e}")
            return None
    
    def balance_load(self):
        """Balance cache load across nodes"""
        if not dist.is_initialized():
            return
        
        # Calculate average load
        total_load = sum(self.node_loads)
        avg_load = total_load / self.world_size
        
        # Identify overloaded and underloaded nodes
        overloaded_nodes = []
        underloaded_nodes = []
        
        for rank, load in enumerate(self.node_loads):
            if load > avg_load * 1.2:  # 20% threshold
                overloaded_nodes.append(rank)
            elif load < avg_load * 0.8:
                underloaded_nodes.append(rank)
        
        # Redistribute caches
        self._redistribute_caches(overloaded_nodes, underloaded_nodes)
    
    def _redistribute_caches(self, overloaded_nodes: List[int], underloaded_nodes: List[int]):
        """Redistribute caches from overloaded to underloaded nodes"""
        # Simple redistribution strategy
        for overloaded_rank in overloaded_nodes:
            for underloaded_rank in underloaded_nodes:
                if len(self.cache_distribution[overloaded_rank]) > 0:
                    # Move one cache from overloaded to underloaded
                    cache_id = self.cache_distribution[overloaded_rank].pop(0)
                    self.cache_distribution[underloaded_rank].append(cache_id)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get distributed pool statistics"""
        return {
            'local_caches': len(self.local_caches),
            'remote_caches': len(self.remote_cache_map),
            'rdma_transfers': self.rdma_transfers,
            'rdma_bytes_transferred': self.rdma_bytes_transferred,
            'node_loads': self.node_loads,
            'load_imbalance': max(self.node_loads) - min(self.node_loads) if self.node_loads else 0
        }


class CacheAwarePrefillScheduler:
    """
    Cache-aware Prefill Scheduler with load balancing
    """
    
    def __init__(self, 
                 distributed_pool: DistributedKVCachePool,
                 slo_constraints: SLOConstraints):
        self.distributed_pool = distributed_pool
        self.slo_constraints = slo_constraints
        
        # Prefill instances
        self.prefill_instances = {}
        self.instance_loads = {}
        
        # Scheduling queues
        self.prefill_queue = deque()
        self.priority_queue = deque()
        
        # Cache reuse tracking
        self.cache_reuse_count = 0
        self.total_prefill_requests = 0
        
    def schedule_prefill(self, 
                        request_id: str,
                        input_tokens: torch.Tensor,
                        cache_hints: List[str] = None) -> str:
        """Schedule a prefill request with cache awareness"""
        self.total_prefill_requests += 1
        
        # Check for cache reuse opportunities
        cache_reuse_score = self._calculate_cache_reuse_score(cache_hints)
        
        # Create prefill instance
        instance_id = f"prefill_{request_id}_{int(time.time())}"
        prefill_instance = PrefillInstance(
            instance_id=instance_id,
            input_tokens=input_tokens,
            cache_hints=cache_hints or [],
            priority=cache_reuse_score,
            slo_deadline=time.time() + self.slo_constraints.ttft_slo
        )
        
        # Add to appropriate queue
        if cache_reuse_score > 0.5:  # High cache reuse potential
            self.priority_queue.append(prefill_instance)
            self.cache_reuse_count += 1
        else:
            self.prefill_queue.append(prefill_instance)
        
        return instance_id
    
    def _calculate_cache_reuse_score(self, cache_hints: List[str]) -> float:
        """Calculate cache reuse score based on available caches"""
        if not cache_hints:
            return 0.0
        
        reuse_score = 0.0
        for cache_id in cache_hints:
            if cache_id in self.distributed_pool.local_caches:
                reuse_score += 1.0
            elif cache_id in self.distributed_pool.remote_cache_map:
                reuse_score += 0.5  # Remote cache has lower score
        
        return reuse_score / len(cache_hints) if cache_hints else 0.0
    
    def get_next_prefill(self) -> Optional['PrefillInstance']:
        """Get the next prefill instance to process"""
        # Prioritize high cache reuse instances
        if self.priority_queue:
            return self.priority_queue.popleft()
        elif self.prefill_queue:
            return self.prefill_queue.popleft()
        
        return None
    
    def balance_prefill_load(self):
        """Balance prefill load across instances"""
        if not self.prefill_instances:
            return
        
        # Calculate average load
        avg_load = sum(self.instance_loads.values()) / len(self.instance_loads)
        
        # Redistribute requests from overloaded instances
        overloaded_instances = [
            inst_id for inst_id, load in self.instance_loads.items()
            if load > avg_load * 1.2
        ]
        
        underloaded_instances = [
            inst_id for inst_id, load in self.instance_loads.items()
            if load < avg_load * 0.8
        ]
        
        # Move requests from overloaded to underloaded instances
        for overloaded_id in overloaded_instances:
            for underloaded_id in underloaded_instances:
                if (self.instance_loads[overloaded_id] > 
                    self.instance_loads[underloaded_id] + 1):
                    # Move one request
                    self.instance_loads[overloaded_id] -= 1
                    self.instance_loads[underloaded_id] += 1
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'prefill_queue_size': len(self.prefill_queue),
            'priority_queue_size': len(self.priority_queue),
            'cache_reuse_rate': self.cache_reuse_count / self.total_prefill_requests if self.total_prefill_requests > 0 else 0,
            'total_prefill_requests': self.total_prefill_requests,
            'active_instances': len(self.prefill_instances),
            'instance_loads': dict(self.instance_loads)
        }


class PrefillInstance:
    """Prefill instance for processing prefill requests"""
    
    def __init__(self, 
                 instance_id: str,
                 input_tokens: torch.Tensor,
                 cache_hints: List[str],
                 priority: float,
                 slo_deadline: float):
        self.instance_id = instance_id
        self.input_tokens = input_tokens
        self.cache_hints = cache_hints
        self.priority = priority
        self.slo_deadline = slo_deadline
        self.start_time = time.time()
        self.status = "pending"
        self.output_tokens = None
    
    def process(self, distributed_pool: DistributedKVCachePool) -> torch.Tensor:
        """Process the prefill request"""
        self.status = "processing"
        
        # Check cache availability
        available_caches = []
        for cache_id in self.cache_hints:
            cache_data = distributed_pool.request_cache(cache_id)
            if cache_data is not None:
                available_caches.append(cache_data)
        
        # Simulate prefill processing
        # In real implementation, this would use the actual model
        batch_size, seq_len = self.input_tokens.shape
        output_tokens = torch.randn(batch_size, seq_len, 512)  # Simulated output
        
        self.status = "completed"
        self.output_tokens = output_tokens
        
        return output_tokens


class DecodingInstance:
    """Decoding instance for processing decoding requests"""
    
    def __init__(self, 
                 instance_id: str,
                 input_tokens: torch.Tensor,
                 cache_data: torch.Tensor = None):
        self.instance_id = instance_id
        self.input_tokens = input_tokens
        self.cache_data = cache_data
        self.start_time = time.time()
        self.status = "pending"
        self.output_tokens = None
        self.throughput = 0.0
    
    def process(self) -> torch.Tensor:
        """Process the decoding request"""
        self.status = "processing"
        
        # Simulate decoding processing
        batch_size, seq_len = self.input_tokens.shape
        output_tokens = torch.randn(batch_size, seq_len, 512)  # Simulated output
        
        # Calculate throughput
        processing_time = time.time() - self.start_time
        self.throughput = batch_size * seq_len / processing_time if processing_time > 0 else 0
        
        self.status = "completed"
        self.output_tokens = output_tokens
        
        return output_tokens


class LoadBalanceDecodingScheduler:
    """
    Load-balancing Decoding Scheduler for maximum throughput
    """
    
    def __init__(self, 
                 slo_constraints: SLOConstraints,
                 max_instances: int = 8):
        self.slo_constraints = slo_constraints
        self.max_instances = max_instances
        
        # Decoding instances
        self.decoding_instances = {}
        self.instance_throughputs = {}
        
        # Scheduling queues
        self.decoding_queue = deque()
        self.completed_instances = deque()
        
        # Throughput tracking
        self.total_throughput = 0.0
        self.slo_violations = 0
    
    def schedule_decoding(self, 
                         request_id: str,
                         input_tokens: torch.Tensor,
                         cache_data: torch.Tensor = None) -> str:
        """Schedule a decoding request"""
        instance_id = f"decoding_{request_id}_{int(time.time())}"
        
        decoding_instance = DecodingInstance(
            instance_id=instance_id,
            input_tokens=input_tokens,
            cache_data=cache_data
        )
        
        self.decoding_queue.append(decoding_instance)
        return instance_id
    
    def get_next_decoding(self) -> Optional[DecodingInstance]:
        """Get the next decoding instance to process"""
        if self.decoding_queue:
            return self.decoding_queue.popleft()
        return None
    
    def balance_decoding_load(self):
        """Balance decoding load across instances"""
        if len(self.decoding_instances) < self.max_instances:
            return
        
        # Calculate average throughput
        avg_throughput = sum(self.instance_throughputs.values()) / len(self.instance_throughputs)
        
        # Identify low-throughput instances
        low_throughput_instances = [
            inst_id for inst_id, throughput in self.instance_throughputs.items()
            if throughput < avg_throughput * 0.8
        ]
        
        # Redistribute load
        for low_throughput_id in low_throughput_instances:
            # Move some requests to higher-throughput instances
            high_throughput_instances = [
                inst_id for inst_id, throughput in self.instance_throughputs.items()
                if throughput > avg_throughput * 1.2
            ]
            
            if high_throughput_instances:
                # Redistribute load
                self._redistribute_decoding_load(low_throughput_id, high_throughput_instances[0])
    
    def _redistribute_decoding_load(self, from_instance: str, to_instance: str):
        """Redistribute decoding load between instances"""
        # Simple load redistribution
        if (from_instance in self.instance_throughputs and 
            to_instance in self.instance_throughputs):
            
            # Move some load from low-throughput to high-throughput instance
            load_to_move = min(
                self.instance_throughputs[from_instance] * 0.1,
                self.instance_throughputs[to_instance] * 0.1
            )
            
            self.instance_throughputs[from_instance] -= load_to_move
            self.instance_throughputs[to_instance] += load_to_move
    
    def check_slo_compliance(self, instance: DecodingInstance) -> bool:
        """Check if instance meets SLO constraints"""
        processing_time = time.time() - instance.start_time
        
        # Check TBT SLO
        if processing_time > self.slo_constraints.tbt_slo:
            self.slo_violations += 1
            return False
        
        return True
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'decoding_queue_size': len(self.decoding_queue),
            'active_instances': len(self.decoding_instances),
            'total_throughput': self.total_throughput,
            'slo_violations': self.slo_violations,
            'slo_compliance_rate': 1.0 - (self.slo_violations / max(len(self.decoding_instances), 1)),
            'instance_throughputs': dict(self.instance_throughputs)
        }


class KVCacheCentricSystem:
    """
    Main KVCache-Centric System orchestrating all components
    """
    
    def __init__(self, 
                 world_size: int = 1,
                 local_rank: int = 0,
                 enable_rdma: bool = True,
                 slo_constraints: Optional[SLOConstraints] = None):
        
        self.world_size = world_size
        self.local_rank = local_rank
        self.enable_rdma = enable_rdma
        
        # Default SLO constraints
        if slo_constraints is None:
            slo_constraints = SLOConstraints(
                ttft_slo=0.1,  # 100ms TTFT
                tbt_slo=0.05,  # 50ms TBT
                max_cache_dram=16 * 1024**3,  # 16GB DRAM
                max_cache_vram=8 * 1024**3    # 8GB VRAM
            )
        
        self.slo_constraints = slo_constraints
        
        # Initialize components
        self.paged_cache = PagedKVCache(
            total_size=1024 * 1024,  # 1M pages
            vram_capacity=slo_constraints.max_cache_vram,
            dram_capacity=slo_constraints.max_cache_dram
        )
        
        self.distributed_pool = DistributedKVCachePool(
            world_size=world_size,
            local_rank=local_rank,
            enable_rdma=enable_rdma
        )
        
        self.prefill_scheduler = CacheAwarePrefillScheduler(
            distributed_pool=self.distributed_pool,
            slo_constraints=slo_constraints
        )
        
        self.decoding_scheduler = LoadBalanceDecodingScheduler(
            slo_constraints=slo_constraints
        )
        
        # System statistics
        self.system_start_time = time.time()
        self.total_requests = 0
        self.completed_requests = 0
        
    def process_prefill_request(self, 
                               request_id: str,
                               input_tokens: torch.Tensor,
                               cache_hints: List[str] = None) -> str:
        """Process a prefill request"""
        self.total_requests += 1
        
        instance_id = self.prefill_scheduler.schedule_prefill(
            request_id=request_id,
            input_tokens=input_tokens,
            cache_hints=cache_hints
        )
        
        return instance_id
    
    def process_decoding_request(self, 
                                request_id: str,
                                input_tokens: torch.Tensor,
                                cache_data: torch.Tensor = None) -> str:
        """Process a decoding request"""
        self.total_requests += 1
        
        instance_id = self.decoding_scheduler.schedule_decoding(
            request_id=request_id,
            input_tokens=input_tokens,
            cache_data=cache_data
        )
        
        return instance_id
    
    def optimize_system(self):
        """Run system optimization"""
        # Balance distributed cache pool
        self.distributed_pool.balance_load()
        
        # Balance prefill scheduler
        self.prefill_scheduler.balance_prefill_load()
        
        # Balance decoding scheduler
        self.decoding_scheduler.balance_decoding_load()
        
        # Update statistics
        self._update_system_stats()
    
    def _update_system_stats(self):
        """Update system statistics"""
        # Update completion rate
        self.completed_requests = (
            len(self.prefill_scheduler.completed_instances) +
            len(self.decoding_scheduler.completed_instances)
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        uptime = time.time() - self.system_start_time
        
        return {
            'uptime': uptime,
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'completion_rate': self.completed_requests / max(self.total_requests, 1),
            'paged_cache': self.paged_cache.get_cache_stats(),
            'distributed_pool': self.distributed_pool.get_pool_stats(),
            'prefill_scheduler': self.prefill_scheduler.get_scheduler_stats(),
            'decoding_scheduler': self.decoding_scheduler.get_scheduler_stats(),
            'slo_constraints': {
                'ttft_slo': self.slo_constraints.ttft_slo,
                'tbt_slo': self.slo_constraints.tbt_slo,
                'max_cache_dram': self.slo_constraints.max_cache_dram,
                'max_cache_vram': self.slo_constraints.max_cache_vram
            }
        }


# Factory function
def create_kvcache_centric_system(world_size: int = 1,
                                 local_rank: int = 0,
                                 enable_rdma: bool = True,
                                 ttft_slo: float = 0.1,
                                 tbt_slo: float = 0.05,
                                 max_cache_dram: int = 16 * 1024**3,
                                 max_cache_vram: int = 8 * 1024**3) -> KVCacheCentricSystem:
    """
    Factory function to create KVCache-Centric System
    
    Args:
        world_size: Number of distributed processes
        local_rank: Local process rank
        enable_rdma: Enable RDMA inter-node transfer
        ttft_slo: Time to First Token SLO (seconds)
        tbt_slo: Time Between Tokens SLO (seconds)
        max_cache_dram: Maximum cache size in DRAM (bytes)
        max_cache_vram: Maximum cache size in VRAM (bytes)
    
    Returns:
        KVCacheCentricSystem instance
    """
    slo_constraints = SLOConstraints(
        ttft_slo=ttft_slo,
        tbt_slo=tbt_slo,
        max_cache_dram=max_cache_dram,
        max_cache_vram=max_cache_vram
    )
    
    return KVCacheCentricSystem(
        world_size=world_size,
        local_rank=local_rank,
        enable_rdma=enable_rdma,
        slo_constraints=slo_constraints
    )
