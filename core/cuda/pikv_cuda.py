"""
Python bindings for PiKV CUDA kernels
Provides high-level interface to CUDA-accelerated operations
"""

import ctypes
import numpy as np
import torch
from typing import Optional, Tuple, Union
import os

# Load CUDA library
def load_cuda_library():
    """Load the CUDA kernel library"""
    lib_path = os.path.join(os.path.dirname(__file__), "libpikv_kernels.so")
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"CUDA library not found at {lib_path}. "
            "Please build the library first: ./scripts/build_cuda.sh release"
        )
    
    try:
        lib = ctypes.CDLL(lib_path)
        return lib
    except Exception as e:
        raise RuntimeError(f"Failed to load CUDA library: {e}")

# Load library
try:
    CUDA_LIB = load_cuda_library()
except Exception as e:
    print(f"Warning: CUDA library not available: {e}")
    CUDA_LIB = None

class PiKVCUDA:
    """Python interface to PiKV CUDA kernels"""
    
    def __init__(self):
        if CUDA_LIB is None:
            raise RuntimeError("CUDA library not available")
        
        # Set function argument types
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Setup function signatures for CUDA library calls"""
        # MoE routing
        CUDA_LIB.moe_routing_cuda.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # router_logits
            ctypes.c_void_p,  # router_weights
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # num_experts
            ctypes.c_void_p   # stream
        ]
        
        # Top-k experts
        CUDA_LIB.top_k_experts_cuda.argtypes = [
            ctypes.c_void_p,  # router_logits
            ctypes.c_void_p,  # expert_indices
            ctypes.c_void_p,  # expert_weights
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # num_experts
            ctypes.c_int,     # top_k
            ctypes.c_void_p   # stream
        ]
        
        # LoRA compression
        CUDA_LIB.lora_compression_cuda.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # lora_A
            ctypes.c_void_p,  # lora_B
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # rank
            ctypes.c_float,   # alpha
            ctypes.c_void_p   # stream
        ]
        
        # Pyramid compression
        CUDA_LIB.pyramid_compression_cuda.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # encoder_weights
            ctypes.c_void_p,  # decoder_weights
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # num_levels
            ctypes.c_void_p   # stream
        ]
        
        # Cache operations
        CUDA_LIB.lru_cache_update_cuda.argtypes = [
            ctypes.c_void_p,  # cache_keys
            ctypes.c_void_p,  # cache_values
            ctypes.c_void_p,  # cache_timestamps
            ctypes.c_void_p,  # new_keys
            ctypes.c_void_p,  # new_values
            ctypes.c_int,     # cache_size
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # current_timestamp
            ctypes.c_void_p   # stream
        ]
        
        CUDA_LIB.h2o_cache_eviction_cuda.argtypes = [
            ctypes.c_void_p,  # cache_keys
            ctypes.c_void_p,  # cache_values
            ctypes.c_void_p,  # cache_timestamps
            ctypes.c_void_p,  # eviction_mask
            ctypes.c_int,     # cache_size
            ctypes.c_int,     # hidden_size
            ctypes.c_float,   # importance_threshold
            ctypes.c_int,     # max_age
            ctypes.c_void_p   # stream
        ]
    
    def moe_routing(
        self,
        input_tensor: torch.Tensor,
        router_weights: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        Perform MoE routing using CUDA kernels
        
        Args:
            input_tensor: Input tensor [batch_size, seq_len, hidden_size]
            router_weights: Router weight matrix [hidden_size, num_experts]
            stream: Optional CUDA stream
            
        Returns:
            Router logits [batch_size, seq_len, num_experts]
        """
        if not input_tensor.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")
        
        batch_size, seq_len, hidden_size = input_tensor.shape
        num_experts = router_weights.shape[1]
        
        # Create output tensor
        output = torch.empty(
            batch_size, seq_len, num_experts,
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )
        
        # Get CUDA stream
        cuda_stream = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        
        # Call CUDA kernel
        CUDA_LIB.moe_routing_cuda(
            input_tensor.data_ptr(),
            output.data_ptr(),
            router_weights.data_ptr(),
            batch_size,
            seq_len,
            hidden_size,
            num_experts,
            ctypes.c_void_p(cuda_stream)
        )
        
        return output
    
    def top_k_experts(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        stream: Optional[torch.cuda.Stream] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k experts using CUDA kernels
        
        Args:
            router_logits: Router logits [batch_size, seq_len, num_experts]
            top_k: Number of top experts to select
            stream: Optional CUDA stream
            
        Returns:
            Tuple of (expert_indices, expert_weights)
        """
        if not router_logits.is_cuda:
            raise ValueError("Router logits must be on CUDA device")
        
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Create output tensors
        expert_indices = torch.empty(
            batch_size, seq_len, top_k,
            dtype=torch.int32,
            device=router_logits.device
        )
        
        expert_weights = torch.empty(
            batch_size, seq_len, top_k,
            dtype=router_logits.dtype,
            device=router_logits.device
        )
        
        # Get CUDA stream
        cuda_stream = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        
        # Call CUDA kernel
        CUDA_LIB.top_k_experts_cuda(
            router_logits.data_ptr(),
            expert_indices.data_ptr(),
            expert_weights.data_ptr(),
            batch_size,
            seq_len,
            num_experts,
            top_k,
            ctypes.c_void_p(cuda_stream)
        )
        
        return expert_indices, expert_weights
    
    def lora_compression(
        self,
        input_tensor: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        alpha: float = 32.0,
        stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        Apply LoRA compression using CUDA kernels
        
        Args:
            input_tensor: Input tensor [batch_size, seq_len, hidden_size]
            lora_A: LoRA A matrix [hidden_size, rank]
            lora_B: LoRA B matrix [rank, hidden_size]
            alpha: LoRA scaling factor
            stream: Optional CUDA stream
            
        Returns:
            Compressed tensor [batch_size, seq_len, hidden_size]
        """
        if not input_tensor.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")
        
        batch_size, seq_len, hidden_size = input_tensor.shape
        rank = lora_A.shape[1]
        
        # Create output tensor
        output = torch.empty_like(input_tensor)
        
        # Get CUDA stream
        cuda_stream = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        
        # Call CUDA kernel
        CUDA_LIB.lora_compression_cuda(
            input_tensor.data_ptr(),
            output.data_ptr(),
            lora_A.data_ptr(),
            lora_B.data_ptr(),
            batch_size,
            seq_len,
            hidden_size,
            rank,
            alpha,
            ctypes.c_void_p(cuda_stream)
        )
        
        return output
    
    def pyramid_compression(
        self,
        input_tensor: torch.Tensor,
        encoder_weights: torch.Tensor,
        decoder_weights: torch.Tensor,
        num_levels: int,
        stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        Apply pyramid compression using CUDA kernels
        
        Args:
            input_tensor: Input tensor [batch_size, seq_len, hidden_size]
            encoder_weights: Encoder weight matrices
            decoder_weights: Decoder weight matrices
            num_levels: Number of pyramid levels
            stream: Optional CUDA stream
            
        Returns:
            Compressed tensor [batch_size, seq_len, hidden_size]
        """
        if not input_tensor.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")
        
        batch_size, seq_len, hidden_size = input_tensor.shape
        
        # Create output tensor
        output = torch.empty_like(input_tensor)
        
        # Get CUDA stream
        cuda_stream = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        
        # Call CUDA kernel
        CUDA_LIB.pyramid_compression_cuda(
            input_tensor.data_ptr(),
            output.data_ptr(),
            encoder_weights.data_ptr(),
            decoder_weights.data_ptr(),
            batch_size,
            seq_len,
            hidden_size,
            num_levels,
            ctypes.c_void_p(cuda_stream)
        )
        
        return output
    
    def lru_cache_update(
        self,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        cache_timestamps: torch.Tensor,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        current_timestamp: int,
        stream: Optional[torch.cuda.Stream] = None
    ):
        """
        Update LRU cache using CUDA kernels
        
        Args:
            cache_keys: Cache key tensor [cache_size, hidden_size]
            cache_values: Cache value tensor [cache_size, hidden_size]
            cache_timestamps: Cache timestamp tensor [cache_size]
            new_keys: New key tensor [cache_size, hidden_size]
            new_values: New value tensor [cache_size, hidden_size]
            current_timestamp: Current timestamp
            stream: Optional CUDA stream
        """
        if not cache_keys.is_cuda:
            raise ValueError("Cache tensors must be on CUDA device")
        
        cache_size, hidden_size = cache_keys.shape
        
        # Get CUDA stream
        cuda_stream = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        
        # Call CUDA kernel
        CUDA_LIB.lru_cache_update_cuda(
            cache_keys.data_ptr(),
            cache_values.data_ptr(),
            cache_timestamps.data_ptr(),
            new_keys.data_ptr(),
            new_values.data_ptr(),
            cache_size,
            hidden_size,
            current_timestamp,
            ctypes.c_void_p(cuda_stream)
        )
    
    def h2o_cache_eviction(
        self,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        cache_timestamps: torch.Tensor,
        importance_threshold: float,
        max_age: int,
        stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        Perform H2O cache eviction using CUDA kernels
        
        Args:
            cache_keys: Cache key tensor [cache_size, hidden_size]
            cache_values: Cache value tensor [cache_size, hidden_size]
            cache_timestamps: Cache timestamp tensor [cache_size]
            importance_threshold: Importance threshold for eviction
            max_age: Maximum age before eviction
            stream: Optional CUDA stream
            
        Returns:
            Eviction mask [cache_size]
        """
        if not cache_keys.is_cuda:
            raise ValueError("Cache tensors must be on CUDA device")
        
        cache_size, hidden_size = cache_keys.shape
        
        # Create eviction mask
        eviction_mask = torch.empty(
            cache_size,
            dtype=torch.int32,
            device=cache_keys.device
        )
        
        # Get CUDA stream
        cuda_stream = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        
        # Call CUDA kernel
        CUDA_LIB.h2o_cache_eviction_cuda(
            cache_keys.data_ptr(),
            cache_values.data_ptr(),
            cache_timestamps.data_ptr(),
            eviction_mask.data_ptr(),
            cache_size,
            hidden_size,
            importance_threshold,
            max_age,
            ctypes.c_void_p(cuda_stream)
        )
        
        return eviction_mask


# Convenience function to check CUDA availability
def is_cuda_available() -> bool:
    """Check if CUDA kernels are available"""
    return CUDA_LIB is not None and torch.cuda.is_available()


# Example usage
if __name__ == "__main__":
    if not is_cuda_available():
        print("CUDA not available")
        exit(1)
    
    print("Testing PiKV CUDA kernels...")
    
    # Create test tensors
    batch_size, seq_len, hidden_size = 2, 64, 512
    num_experts = 8
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    router_weights = torch.randn(hidden_size, num_experts, device='cuda')
    
    # Test MoE routing
    try:
        pikv_cuda = PiKVCUDA()
        
        # Test routing
        router_logits = pikv_cuda.moe_routing(input_tensor, router_weights)
        print(f"✓ MoE routing successful: {router_logits.shape}")
        
        # Test top-k experts
        expert_indices, expert_weights = pikv_cuda.top_k_experts(router_logits, top_k=2)
        print(f"✓ Top-k experts successful: {expert_indices.shape}, {expert_weights.shape}")
        
        print("All CUDA tests passed! 🎉")
        
    except Exception as e:
        print(f"CUDA test failed: {e}")
