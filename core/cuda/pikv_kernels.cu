#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;

// CUDA kernels for MoE routing
__global__ void moe_routing_kernel(
    const float* input,
    float* router_logits,
    const float* router_weights,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_experts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (seq_len * hidden_size);
    int seq_idx = (idx % (seq_len * hidden_size)) / hidden_size;
    int hidden_idx = idx % hidden_size;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    // Compute router logits
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        int input_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + i;
        int weight_idx = i * num_experts + hidden_idx;
        sum += input[input_idx] * router_weights[weight_idx];
    }
    
    int output_idx = batch_idx * seq_len * num_experts + seq_idx * num_experts + hidden_idx;
    router_logits[output_idx] = sum;
}

__global__ void top_k_experts_kernel(
    const float* router_logits,
    int* expert_indices,
    float* expert_weights,
    const int batch_size,
    const int seq_len,
    const int num_experts,
    const int top_k
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) {
        return;
    }
    
    // Shared memory for sorting
    __shared__ float shared_logits[BLOCK_SIZE];
    __shared__ int shared_indices[BLOCK_SIZE];
    
    // Load logits into shared memory
    int base_idx = batch_idx * seq_len * num_experts + seq_idx * num_experts;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        shared_logits[i] = router_logits[base_idx + i];
        shared_indices[i] = i;
    }
    __syncthreads();
    
    // Simple bubble sort for top-k (can be optimized with bitonic sort)
    for (int i = 0; i < top_k; i++) {
        for (int j = i + 1; j < num_experts; j++) {
            if (shared_logits[j] > shared_logits[i]) {
                // Swap logits
                float temp_logit = shared_logits[i];
                shared_logits[i] = shared_logits[j];
                shared_logits[j] = temp_logit;
                
                // Swap indices
                int temp_idx = shared_indices[i];
                shared_indices[i] = shared_indices[j];
                shared_indices[j] = temp_idx;
            }
        }
    }
    
    // Store results
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        int output_idx = batch_idx * seq_len * top_k + seq_idx * top_k + i;
        expert_indices[output_idx] = shared_indices[i];
        expert_weights[output_idx] = shared_logits[i];
    }
}

// CUDA kernels for compression (legacy tensor API; full KV compression lives in
// pikv_compression_kernel.cu — LoRA/Quant/Pyramid/SVD/Hybrid/Quality)
__global__ void lora_compression_legacy_kernel(
    const float* input,
    float* output,
    const float* lora_A,
    const float* lora_B,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int rank,
    const float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (seq_len * hidden_size);
    int seq_idx = (idx % (seq_len * hidden_size)) / hidden_size;
    int hidden_idx = idx % hidden_size;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    // Compute LoRA compression
    float intermediate = 0.0f;
    for (int r = 0; r < rank; r++) {
        int input_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx;
        int a_idx = hidden_idx * rank + r;
        int b_idx = r * hidden_size + hidden_idx;
        
        intermediate += input[input_idx] * lora_A[a_idx];
    }
    
    float result = 0.0f;
    for (int r = 0; r < rank; r++) {
        int b_idx = r * hidden_size + hidden_idx;
        result += intermediate * lora_B[b_idx];
    }
    
    int output_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx;
    output[output_idx] = result * (alpha / rank);
}

__global__ void pyramid_compression_legacy_kernel(
    const float* input,
    float* output,
    const float* encoder_weights,
    const float* decoder_weights,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_levels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (seq_len * hidden_size);
    int seq_idx = (idx % (seq_len * hidden_size)) / hidden_size;
    int hidden_idx = idx % hidden_size;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    // Apply pyramid encoding and decoding
    float encoded = input[idx];
    for (int level = 0; level < num_levels; level++) {
        // Encoding
        float encoded_temp = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            int weight_idx = level * hidden_size * hidden_size + hidden_idx * hidden_size + i;
            encoded_temp += encoded * encoder_weights[weight_idx];
        }
        encoded = fmaxf(encoded_temp, 0.0f); // ReLU
    }
    
    // Decoding
    float decoded = encoded;
    for (int level = num_levels - 1; level >= 0; level--) {
        float decoded_temp = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            int weight_idx = level * hidden_size * hidden_size + hidden_idx * hidden_size + i;
            decoded_temp += decoded * decoder_weights[weight_idx];
        }
        decoded = fmaxf(decoded_temp, 0.0f); // ReLU
    }
    
    output[idx] = decoded;
}

// CUDA kernels for cache scheduling
__global__ void lru_cache_update_kernel(
    float* cache_keys,
    float* cache_values,
    int* cache_timestamps,
    const float* new_keys,
    const float* new_values,
    const int cache_size,
    const int hidden_size,
    const int current_timestamp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cache_idx = idx / hidden_size;
    int hidden_idx = idx % hidden_size;
    
    if (cache_idx >= cache_size) {
        return;
    }
    
    // Update timestamp
    cache_timestamps[cache_idx] = current_timestamp;
    
    // Update cache content
    int key_idx = cache_idx * hidden_size + hidden_idx;
    int value_idx = cache_idx * hidden_size + hidden_idx;
    
    cache_keys[key_idx] = new_keys[key_idx];
    cache_values[value_idx] = new_values[value_idx];
}

__global__ void h2o_cache_eviction_kernel(
    const float* cache_keys,
    const float* cache_values,
    const int* cache_timestamps,
    int* eviction_mask,
    const int cache_size,
    const int hidden_size,
    const float importance_threshold,
    const int max_age
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cache_idx = idx / hidden_size;
    int hidden_idx = idx % hidden_size;
    
    if (cache_idx >= cache_size) {
        return;
    }
    
    // Calculate importance score
    float importance = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        int key_idx = cache_idx * hidden_size + i;
        importance += fabsf(cache_keys[key_idx]);
    }
    importance /= hidden_size;
    
    // Calculate age
    int age = cache_timestamps[cache_idx];
    
    // Eviction decision
    if (importance < importance_threshold || age > max_age) {
        eviction_mask[cache_idx] = 1;
    } else {
        eviction_mask[cache_idx] = 0;
    }
}

// Host wrapper functions
extern "C" {

// MoE routing wrapper
void moe_routing_cuda(
    const float* input,
    float* router_logits,
    const float* router_weights,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_experts,
    cudaStream_t stream
) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(
        (batch_size * seq_len * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    moe_routing_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, router_logits, router_weights,
        batch_size, seq_len, hidden_size, num_experts
    );
}

// Top-k experts wrapper
void top_k_experts_cuda(
    const float* router_logits,
    int* expert_indices,
    float* expert_weights,
    const int batch_size,
    const int seq_len,
    const int num_experts,
    const int top_k,
    cudaStream_t stream
) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(batch_size, seq_len);
    
    top_k_experts_kernel<<<grid_dim, block_dim, 0, stream>>>(
        router_logits, expert_indices, expert_weights,
        batch_size, seq_len, num_experts, top_k
    );
}

// LoRA compression wrapper
void lora_compression_cuda(
    const float* input,
    float* output,
    const float* lora_A,
    const float* lora_B,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int rank,
    const float alpha,
    cudaStream_t stream
) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(
        (batch_size * seq_len * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    lora_compression_legacy_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, output, lora_A, lora_B,
        batch_size, seq_len, hidden_size, rank, alpha
    );
}

// Pyramid compression wrapper
void pyramid_compression_cuda(
    const float* input,
    float* output,
    const float* encoder_weights,
    const float* decoder_weights,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_levels,
    cudaStream_t stream
) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(
        (batch_size * seq_len * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    pyramid_compression_legacy_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, output, encoder_weights, decoder_weights,
        batch_size, seq_len, hidden_size, num_levels
    );
}

// LRU cache update wrapper
void lru_cache_update_cuda(
    float* cache_keys,
    float* cache_values,
    int* cache_timestamps,
    const float* new_keys,
    const float* new_values,
    const int cache_size,
    const int hidden_size,
    const int current_timestamp,
    cudaStream_t stream
) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(
        (cache_size * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    lru_cache_update_kernel<<<grid_dim, block_dim, 0, stream>>>(
        cache_keys, cache_values, cache_timestamps,
        new_keys, new_values, cache_size, hidden_size, current_timestamp
    );
}

// H2O cache eviction wrapper
void h2o_cache_eviction_cuda(
    const float* cache_keys,
    const float* cache_values,
    const int* cache_timestamps,
    int* eviction_mask,
    const int cache_size,
    const int hidden_size,
    const float importance_threshold,
    const int max_age,
    cudaStream_t stream
) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(
        (cache_size * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    h2o_cache_eviction_kernel<<<grid_dim, block_dim, 0, stream>>>(
        cache_keys, cache_values, cache_timestamps, eviction_mask,
        cache_size, hidden_size, importance_threshold, max_age
    );
}

} // extern "C"
