#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the Metal device, command queue, and compiles the compute kernels.
int metal_init(void);

// Checks if Metal GPU acceleration is initialized and available.
bool metal_is_available(void);

// Batching API: records multiple GPU commands into a single command buffer per token
void metal_begin_batch(void);
void metal_end_batch(void);

// Allocates permanent persistent GPU buffers for activations and KV-cache
int metal_alloc_buffers(uint32_t dim, uint32_t hidden_dim, uint32_t kv_dim,
                        uint32_t vocab_size, uint32_t num_layers, uint32_t max_seq);

// Pre-wrapped GPU Buffer Management
typedef void* metal_buffer_t;
metal_buffer_t metal_create_buffer(const void* ptr, size_t bytes);
void metal_release_buffer(metal_buffer_t buf);

// Direct Buffer GEMV (zero runtime buffer wrapping)
int metal_gemv_buf(int quant_type, float* y, const float* x, metal_buffer_t w_buf, uint32_t rows, uint32_t cols);

typedef struct {
    metal_buffer_t wq; int wq_type;
    metal_buffer_t wk; int wk_type;
    metal_buffer_t wv; int wv_type;
    metal_buffer_t wo; int wo_type;
    metal_buffer_t ffn_gate; int ffn_gate_type;
    metal_buffer_t ffn_up; int ffn_up_type;
    metal_buffer_t ffn_down; int ffn_down_type;
    metal_buffer_t attn_norm;
    metal_buffer_t ffn_norm;
} metal_layer_weights_t;

// Single-call forward pass across all transformer layers with GPU-resident activations
int metal_forward_transformer(
    const float* initial_x,
    float* out_logits,
    const metal_layer_weights_t* layers,
    metal_buffer_t output_norm_buf,
    metal_buffer_t output_weight_buf,
    int output_weight_type,
    uint32_t num_layers,
    uint32_t dim,
    uint32_t hidden_dim,
    uint32_t kv_dim,
    uint32_t vocab_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t pos,
    uint32_t slot,
    uint32_t max_seq,
    uint32_t active_context,
    float norm_eps,
    float rope_theta,
    float attn_scale
);

// Fused transformer single-layer dispatch (flat arguments for zero CGo pointer checks)
int metal_forward_layer(
    float* x, float* xnorm, float* q, float* k, float* v,
    float* attn_out, float* attn_proj, float* gate_act, float* up_act, float* ffn_down_act,
    const float* attn_norm, const float* ffn_norm,
    metal_buffer_t wq, int wq_type,
    metal_buffer_t wk, int wk_type,
    metal_buffer_t wv, int wv_type,
    metal_buffer_t wo, int wo_type,
    metal_buffer_t ffn_gate, int ffn_gate_type,
    metal_buffer_t ffn_up, int ffn_up_type,
    metal_buffer_t ffn_down, int ffn_down_type,
    uint32_t layer_idx, uint32_t dim, uint32_t hidden_dim, uint32_t kv_dim,
    uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t pos, uint32_t slot, uint32_t max_seq, uint32_t active_context,
    float norm_eps, float rope_theta, float attn_scale
);

// Matrix-vector multiplication functions using Apple Metal GPU
int metal_gemv_f32(float* y, const float* x, const float* w, uint32_t rows, uint32_t cols);
int metal_gemv_f16(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols);
int metal_gemv_q4_0(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols);
int metal_gemv_q8_0(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols);
int metal_gemv_q4_k(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols);
int metal_gemv_q6_k(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols);

// Batched 2D Matrix-Matrix Multiplication (GEMM) for fast prompt prefill
int metal_gemm_q4_0(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols);
int metal_gemm_q8_0(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols);
int metal_gemm_q4_k(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols);
int metal_gemm_q6_k(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols);

// Fused transformer GPU kernels
int metal_rmsnorm(float* out, const float* x, const float* weight, uint32_t dim, float eps);
int metal_rope(float* q, float* k, uint32_t pos, uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim, float theta);
int metal_kv_write(const float* k, const float* v, uint32_t layer, uint32_t slot, uint32_t max_seq, uint32_t kv_dim);
int metal_attention_gqa(float* attn_out, const float* q, const float* k_cache, const float* v_cache,
                        uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
                        uint32_t active_context, float attn_scale);
int metal_swiglu(float* gate, const float* up, uint32_t hidden_dim);
int metal_add_residual(float* x, const float* proj, uint32_t dim);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H
