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

// Allocates specialized persistent GPU buffers for Qwen 3.5 hybrid SSM / Attention architecture
int metal_alloc_qwen35_buffers(
    uint32_t dim,
    uint32_t hidden_dim,
    uint32_t ssm_inner,
    uint32_t ssm_channels,
    uint32_t ssm_rank,
    uint32_t ssm_state_size,
    uint32_t kv_dim,
    uint32_t vocab_size,
    uint32_t num_layers,
    uint32_t max_seq
);

// Resets recurrent SSM state in GPU memory
void metal_reset_ssm_state(void);

// Pre-wrapped GPU Buffer Management
typedef void* metal_buffer_t;
metal_buffer_t metal_create_buffer(const void* ptr, size_t bytes);
void metal_release_buffer(metal_buffer_t buf);

// Direct Buffer GEMV and GEMM (zero runtime buffer wrapping)
int metal_gemv_buf(int quant_type, float* y, const float* x, metal_buffer_t w_buf, uint32_t rows, uint32_t cols);
int metal_gemm_buf(int quant_type, float* y, const float* x, metal_buffer_t w_buf, uint32_t batch_size, uint32_t rows, uint32_t cols);
int metal_gemm_fused_gate_up_buf(int quant_type, float* y, const float* x, metal_buffer_t gate_buf, metal_buffer_t up_buf, uint32_t batch_size, uint32_t rows, uint32_t cols);

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
    metal_buffer_t bq;
    metal_buffer_t bk;
    metal_buffer_t bv;
} metal_layer_weights_t;

// Struct holding GPU buffer handles and quantization types for a single Qwen 3.5 hybrid layer
typedef struct {
    bool is_ssm;
    metal_buffer_t attn_norm;
    metal_buffer_t ffn_norm;

    // SSM layer weights
    metal_buffer_t ssm_gate; int ssm_gate_type;
    metal_buffer_t ssm_qkv;  int ssm_qkv_type;
    metal_buffer_t ssm_conv1d;
    metal_buffer_t ssm_alpha; int ssm_alpha_type;
    metal_buffer_t ssm_beta;  int ssm_beta_type;
    metal_buffer_t ssm_dt_bias;
    metal_buffer_t ssm_a;
    metal_buffer_t ssm_norm;
    metal_buffer_t ssm_out;   int ssm_out_type;

    // Attention layer weights
    metal_buffer_t wq; int wq_type;
    metal_buffer_t wk; int wk_type;
    metal_buffer_t wv; int wv_type;
    metal_buffer_t wo; int wo_type;
    metal_buffer_t q_norm;
    metal_buffer_t k_norm;

    // FFN weights (shared by both SSM and Attention layers)
    metal_buffer_t ffn_gate; int ffn_gate_type;
    metal_buffer_t ffn_up;   int ffn_up_type;
    metal_buffer_t ffn_down; int ffn_down_type;
} metal_qwen35_layer_weights_t;

// Single-call forward pass across all 65 hybrid layers for Qwen 3.5
int metal_forward_qwen35(
    const float* initial_x,
    metal_buffer_t token_embd_buf,
    int32_t token_id,
    float* out_logits,
    uint32_t* out_token,
    const metal_qwen35_layer_weights_t* layers,
    metal_buffer_t output_norm_buf,
    metal_buffer_t output_weight_buf,
    int output_weight_type,
    uint32_t num_layers,
    uint32_t dim,
    uint32_t hidden_dim,
    uint32_t ssm_inner,
    uint32_t ssm_channels,
    uint32_t ssm_state_size,
    uint32_t ssm_groups,
    uint32_t ssm_rank,
    uint32_t kv_dim,
    uint32_t vocab_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t pos,
    uint32_t slot,
    uint32_t max_seq,
    uint32_t active_context,
    float norm_eps,
    float rope_theta,
    float attn_scale
);

// Single-call forward pass across all transformer layers with GPU-resident activations
int metal_forward_transformer(
    const float* initial_x,
    float* out_logits,
    uint32_t* out_token,
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
int metal_rmsnorm_batch(float* out, const float* x, const float* weight, uint32_t dim, float eps, uint32_t batch_size);
int metal_residual_rmsnorm_batch(float* x, const float* proj, float* out_norm, const float* weight, uint32_t dim, float eps, uint32_t batch_size);
int metal_embed_lookup_q4_k_batch(float* out_x, metal_buffer_t w_embd, const uint32_t* token_ids, uint32_t dim, uint32_t batch_size);

// Batched prefill GPU operations for Qwen 3.5 Hybrid architecture
int metal_conv1d_batch(float* out, const float* in, float* state, const float* conv_weight,
                       uint32_t kernel_size, uint32_t channels, uint32_t batch_size);
int metal_ssm_recurrence_batch(
    float* ssm_out, const float* conv_out, const float* ssm_alpha, const float* ssm_beta,
    const float* dt_bias, const float* ssm_a, const float* ssm_norm_w, const float* ssm_gate,
    float* ssm_state, uint32_t ssm_inner, uint32_t ssm_state_size, uint32_t ssm_groups,
    uint32_t ssm_rank, float eps, uint32_t batch_size, uint32_t ssm_channels);
int metal_attention_gqa_batch(
    float* attn_out, const float* q, const float* k_cache, const float* v_cache,
    uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t start_pos, uint32_t max_seq, float attn_scale, uint32_t batch_size);
int metal_kv_write_batch(
    float* k_cache, float* v_cache, const float* k, const float* v,
    uint32_t start_pos, uint32_t max_seq, uint32_t kv_dim, uint32_t batch_size);
int metal_qwen35_split_q_gate_batch(
    float* q_out, float* gate_out, const float* q_gate_in, uint32_t num_tokens);
int metal_rope_norm_batch(
    float* q, float* k, const float* q_norm_w, const float* k_norm_w,
    uint32_t start_pos, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t rope_dim, float theta, float eps, uint32_t batch_size);
int metal_qwen35_attn_gate(float* attn_out, const float* gate, uint32_t total_elements);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H
