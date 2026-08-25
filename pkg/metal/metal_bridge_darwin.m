#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_bridge.h"

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;

static id<MTLComputePipelineState> g_pipeline_f32        = nil;
static id<MTLComputePipelineState> g_pipeline_f16        = nil;
static id<MTLComputePipelineState> g_pipeline_q4_0       = nil;
static id<MTLComputePipelineState> g_pipeline_q8_0       = nil;
static id<MTLComputePipelineState> g_pipeline_q4_k       = nil;
static id<MTLComputePipelineState> g_pipeline_q6_k       = nil;
static id<MTLComputePipelineState> g_pipeline_q2_k       = nil;
static id<MTLComputePipelineState> g_pipeline_q3_k       = nil;

static id<MTLComputePipelineState> g_pipeline_gemm_q4_0  = nil;
static id<MTLComputePipelineState> g_pipeline_gemm_q8_0  = nil;
static id<MTLComputePipelineState> g_pipeline_gemm_q4_k  = nil;
static id<MTLComputePipelineState> g_pipeline_gemm_q6_k  = nil;

static id<MTLComputePipelineState> g_pipeline_fused_gate_up_q4_0 = nil;
static id<MTLComputePipelineState> g_pipeline_fused_gate_up_q8_0 = nil;
static id<MTLComputePipelineState> g_pipeline_fused_gate_up_q4_k = nil;
static id<MTLComputePipelineState> g_pipeline_fused_gate_up_q6_k = nil;
static id<MTLComputePipelineState> g_pipeline_sample_argmax      = nil;

static id<MTLComputePipelineState> g_pipeline_rmsnorm    = nil;
static id<MTLComputePipelineState> g_pipeline_rope       = nil;
static id<MTLComputePipelineState> g_pipeline_attn       = nil;
static id<MTLComputePipelineState> g_pipeline_kv_write   = nil;
static id<MTLComputePipelineState> g_pipeline_swiglu     = nil;
static id<MTLComputePipelineState> g_pipeline_residual   = nil;

static id<MTLBuffer> g_k_cache = nil;
static id<MTLBuffer> g_v_cache = nil;

static id<MTLBuffer> g_buf_x = nil;
static id<MTLBuffer> g_buf_xb = nil;
static id<MTLBuffer> g_buf_q = nil;
static id<MTLBuffer> g_buf_k = nil;
static id<MTLBuffer> g_buf_v = nil;
static id<MTLBuffer> g_buf_attn_out = nil;
static id<MTLBuffer> g_buf_attn_proj = nil;
static id<MTLBuffer> g_buf_gate = nil;
static id<MTLBuffer> g_buf_up = nil;
static id<MTLBuffer> g_buf_down = nil;
static id<MTLBuffer> g_buf_logits = nil;

static id<MTLCommandBuffer> g_batch_cmd = nil;
static id<MTLComputeCommandEncoder> g_batch_encoder = nil;

static const char* METAL_SOURCE = R"(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

struct block_q4_k {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q6_k {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half    d;
};

// --- 8-Way Multi-Row SIMD Vectorized GEMV Kernels (8x L1 Activation Reuse) ---

kernel void gemv_f32(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const float* w            [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    device const float* r_ptrs[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_ptrs[i] = (r < rows) ? (w + r * cols) : (w + r0 * cols);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint col = tid; col < cols; col += 32) {
        float x_val = x[col];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sums[i] += r_ptrs[i][col] * x_val;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) y[r0 + i] = sums[i];
        }
    }
}

kernel void gemv_f16(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const half* w             [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    device const half* r_ptrs[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_ptrs[i] = (r < rows) ? (w + r * cols) : (w + r0 * cols);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint col = tid; col < cols; col += 32) {
        float x_val = x[col];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sums[i] += float(r_ptrs[i][col]) * x_val;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) y[r0 + i] = sums[i];
        }
    }
}

kernel void gemv_q4_0(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_0* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    uint num_blocks = cols / 32;
    device const block_q4_0* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 32) {
        uint x_off = b * 32;
        float x_low[16], x_high[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            x_low[j]  = x[x_off + j];
            x_high[j] = x[x_off + j + 16];
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) {
                device const block_q4_0& blk = r_blocks[i][b];
                float d = float(blk.d);
                device const uint8_t* qs = blk.qs;
                float b_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t val = qs[j];
                    b_sum += float(int(val & 0x0F) - 8) * x_low[j] + float(int((val >> 4) & 0x0F) - 8) * x_high[j];
                }
                sums[i] += b_sum * d;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) y[r0 + i] = sums[i];
        }
    }
}

kernel void gemv_q8_0(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q8_0* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    uint num_blocks = cols / 32;
    device const block_q8_0* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 32) {
        uint x_off = b * 32;
        float x_vals[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            x_vals[j] = x[x_off + j];
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) {
                device const block_q8_0& blk = r_blocks[i][b];
                float d = float(blk.d);
                device const int8_t* qs = blk.qs;
                float b_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    b_sum += float(qs[j]) * x_vals[j];
                }
                sums[i] += b_sum * d;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) y[r0 + i] = sums[i];
        }
    }
}

kernel void gemv_q4_k(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_k* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    uint num_blocks = cols / 256;
    device const block_q4_k* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 32) {
        uint x_off = b * 256;

        for (int sb = 0; sb < 8; sb++) {
            int q_off = sb * 16;
            int sb_x_off = x_off + sb * 32;

            float x_low[16], x_high[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                x_low[j]  = x[sb_x_off + j];
                x_high[j] = x[sb_x_off + j + 16];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (r0 + i < rows) {
                    device const block_q4_k& blk = r_blocks[i][b];
                    float d = float(blk.d), dmin = float(blk.dmin);
                    float sc, m;
                    if (sb < 4) {
                        sc = float(blk.scales[sb] & 63) * d;
                        m  = float(blk.scales[sb + 4] & 63) * dmin;
                    } else {
                        sc = float((blk.scales[sb + 4] & 0xF) | ((blk.scales[sb - 4] >> 6) << 4)) * d;
                        m  = float((blk.scales[sb + 4] >> 4) | ((blk.scales[sb] >> 6) << 4)) * dmin;
                    }
                    device const uint8_t* qs = blk.qs + q_off;
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint8_t byte_val = qs[j];
                        sums[i] += (float(byte_val & 0x0F) * sc - m) * x_low[j] + (float((byte_val >> 4) & 0x0F) * sc - m) * x_high[j];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) y[r0 + i] = sums[i];
        }
    }
}

kernel void gemv_q6_k(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q6_k* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    uint num_blocks = cols / 256;
    device const block_q6_k* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 32) {
        uint x_off = b * 256;

        for (int sb = 0; sb < 16; sb++) {
            int sb_x_off = x_off + sb * 16;
            float x_vals[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                x_vals[j] = x[sb_x_off + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (r0 + i < rows) {
                    device const block_q6_k& blk = r_blocks[i][b];
                    float sc = float(blk.scales[sb]) * float(blk.d);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        int idx = sb * 16 + j;
                        uint8_t l = blk.ql[idx / 2];
                        int q_val = (idx % 2 == 0) ? int(l & 0x0F) : int((l >> 4) & 0x0F);
                        uint8_t h = (blk.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                        q_val = (q_val | (int(h) << 4)) - 32;
                        sums[i] += (float(q_val) * sc) * x_vals[j];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) y[r0 + i] = sums[i];
        }
    }
}

// --- Batched 2D GEMM Kernels for Prompt Prefill ---

kernel void gemm_q4_0_batched(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_0* w       [[buffer(2)]],
    constant uint& batch_size        [[buffer(3)]],
    constant uint& rows              [[buffer(4)]],
    constant uint& cols              [[buffer(5)]],
    uint3 threadgroup_pos            [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint batch_idx = threadgroup_pos.x;
    uint row_idx   = threadgroup_pos.y;
    if (batch_idx >= batch_size || row_idx >= rows) return;

    device const float* cur_x = x + batch_idx * cols;
    uint num_blocks = cols / 32;
    device const block_q4_0* row_blocks = w + row_idx * num_blocks;

    float sum = 0.0f;
    for (uint b = tid; b < num_blocks; b += 32) {
        device const block_q4_0& blk = row_blocks[b];
        float d = float(blk.d);
        uint x_off = b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t val = blk.qs[j];
            int v0 = int(val & 0x0F) - 8;
            int v1 = int((val >> 4) & 0x0F) - 8;
            block_sum += float(v0) * cur_x[x_off + j] + float(v1) * cur_x[x_off + j + 16];
        }
        sum += block_sum * d;
    }
    sum = simd_sum(sum);
    if (tid == 0) {
        y[batch_idx * rows + row_idx] = sum;
    }
}

kernel void gemm_q8_0_batched(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q8_0* w       [[buffer(2)]],
    constant uint& batch_size        [[buffer(3)]],
    constant uint& rows              [[buffer(4)]],
    constant uint& cols              [[buffer(5)]],
    uint3 threadgroup_pos            [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint batch_idx = threadgroup_pos.x;
    uint row_idx   = threadgroup_pos.y;
    if (batch_idx >= batch_size || row_idx >= rows) return;

    device const float* cur_x = x + batch_idx * cols;
    uint num_blocks = cols / 32;
    device const block_q8_0* row_blocks = w + row_idx * num_blocks;

    float sum = 0.0f;
    for (uint b = tid; b < num_blocks; b += 32) {
        device const block_q8_0& blk = row_blocks[b];
        float d = float(blk.d);
        uint x_off = b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            block_sum += float(blk.qs[j]) * cur_x[x_off + j];
        }
        sum += block_sum * d;
    }
    sum = simd_sum(sum);
    if (tid == 0) {
        y[batch_idx * rows + row_idx] = sum;
    }
}

kernel void gemm_q4_k_batched(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_k* w       [[buffer(2)]],
    constant uint& batch_size        [[buffer(3)]],
    constant uint& rows              [[buffer(4)]],
    constant uint& cols              [[buffer(5)]],
    uint3 threadgroup_pos            [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint batch_idx = threadgroup_pos.x;
    uint row_idx   = threadgroup_pos.y;
    if (batch_idx >= batch_size || row_idx >= rows) return;

    device const float* cur_x = x + batch_idx * cols;
    uint num_blocks = cols / 256;
    device const block_q4_k* row_blocks = w + row_idx * num_blocks;

    float sum = 0.0f;
    for (uint b = tid; b < num_blocks; b += 32) {
        device const block_q4_k& blk = row_blocks[b];
        float d = float(blk.d);
        float dmin = float(blk.dmin);
        uint x_off = b * 256;

        for (int sb = 0; sb < 8; sb++) {
            float sc, m;
            if (sb < 4) {
                sc = float(blk.scales[sb] & 63) * d;
                m  = float(blk.scales[sb + 4] & 63) * dmin;
            } else {
                sc = float((blk.scales[sb + 4] & 0xF) | ((blk.scales[sb - 4] >> 6) << 4)) * d;
                m  = float((blk.scales[sb + 4] >> 4) | ((blk.scales[sb] >> 6) << 4)) * dmin;
            }

            int q_off = sb * 16;
            int sb_x_off = x_off + sb * 32;
            for (int j = 0; j < 16; j++) {
                uint8_t byte_val = blk.qs[q_off + j];
                float x0 = float(byte_val & 0x0F);
                float x1 = float((byte_val >> 4) & 0x0F);
                sum += (x0 * sc - m) * cur_x[sb_x_off + j];
                sum += (x1 * sc - m) * cur_x[sb_x_off + j + 16];
            }
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) {
        y[batch_idx * rows + row_idx] = sum;
    }
}

kernel void gemm_q6_k_batched(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q6_k* w       [[buffer(2)]],
    constant uint& batch_size        [[buffer(3)]],
    constant uint& rows              [[buffer(4)]],
    constant uint& cols              [[buffer(5)]],
    uint3 threadgroup_pos            [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint batch_idx = threadgroup_pos.x;
    uint row_idx   = threadgroup_pos.y;
    if (batch_idx >= batch_size || row_idx >= rows) return;

    device const float* cur_x = x + batch_idx * cols;
    uint num_blocks = cols / 256;
    device const block_q6_k* row_blocks = w + row_idx * num_blocks;

    float sum = 0.0f;
    for (uint b = tid; b < num_blocks; b += 32) {
        device const block_q6_k& blk = row_blocks[b];
        float d = float(blk.d);
        uint x_off = b * 256;

        for (int sb = 0; sb < 16; sb++) {
            float sc = float(blk.scales[sb]) * d;
            int sb_x_off = x_off + sb * 16;
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                uint8_t l = blk.ql[idx / 2];
                int q_val = (idx % 2 == 0) ? int(l & 0x0F) : int((l >> 4) & 0x0F);
                uint8_t h = (blk.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                q_val = (q_val | (int(h) << 4)) - 32;
                sum += (float(q_val) * sc) * cur_x[sb_x_off + j];
            }
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) {
        y[batch_idx * rows + row_idx] = sum;
    }
}

// --- Fused Transformer Math Kernels ---

kernel void kernel_rmsnorm(
    device float* out                [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const float* weight       [[buffer(2)]],
    constant uint& dim               [[buffer(3)]],
    constant float& eps              [[buffer(4)]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += 32) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float scale = 1.0f / sqrt((sum_sq / float(dim)) + eps);

    for (uint i = tid; i < dim; i += 32) {
        out[i] = x[i] * scale * weight[i];
    }
}

kernel void kernel_rope(
    device float* q                  [[buffer(0)]],
    device float* k                  [[buffer(1)]],
    constant uint& pos               [[buffer(2)]],
    constant uint& num_heads         [[buffer(3)]],
    constant uint& num_kv_heads      [[buffer(4)]],
    constant uint& head_dim          [[buffer(5)]],
    constant float& theta            [[buffer(6)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint total_q_elems = num_heads * (head_dim / 2);
    uint total_kv_elems = num_kv_heads * (head_dim / 2);

    if (tid < total_q_elems) {
        uint h = tid / (head_dim / 2);
        uint i = tid % (head_dim / 2);
        uint half_dim = head_dim / 2;

        float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
        float val = float(pos) * freq;
        float cos_val = cos(val);
        float sin_val = sin(val);

        uint base = h * head_dim;
        float v0 = q[base + i];
        float v1 = q[base + i + half_dim];
        q[base + i]            = v0 * cos_val - v1 * sin_val;
        q[base + i + half_dim] = v0 * sin_val + v1 * cos_val;
    }

    if (tid < total_kv_elems) {
        uint h = tid / (head_dim / 2);
        uint i = tid % (head_dim / 2);
        uint half_dim = head_dim / 2;

        float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
        float val = float(pos) * freq;
        float cos_val = cos(val);
        float sin_val = sin(val);

        uint base = h * head_dim;
        float v0 = k[base + i];
        float v1 = k[base + i + half_dim];
        k[base + i]            = v0 * cos_val - v1 * sin_val;
        k[base + i + half_dim] = v0 * sin_val + v1 * cos_val;
    }
}

// Fused Multi-Head / Grouped-Query FlashAttention Kernel
kernel void kernel_attention_gqa(
    device float* attn_out           [[buffer(0)]],
    device const float* q            [[buffer(1)]],
    device const float* k_cache      [[buffer(2)]],
    device const float* v_cache      [[buffer(3)]],
    constant uint& num_heads         [[buffer(4)]],
    constant uint& num_kv_heads      [[buffer(5)]],
    constant uint& head_dim          [[buffer(6)]],
    constant uint& active_context    [[buffer(7)]],
    constant float& attn_scale       [[buffer(8)]],
    uint h                           [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    if (h >= num_heads) return;

    uint kv_mul = num_heads / num_kv_heads;
    uint kv_head = h / kv_mul;
    uint kv_dim = num_kv_heads * head_dim;

    device const float* q_h = q + h * head_dim;
    device float* out_h = attn_out + h * head_dim;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    thread float thread_accum[128] = {0.0f};

    for (uint t = 0; t < active_context; t++) {
        device const float* k_t = k_cache + t * kv_dim + kv_head * head_dim;
        device const float* v_t = v_cache + t * kv_dim + kv_head * head_dim;

        float score = 0.0f;
        for (uint d = tid; d < head_dim; d += 32) {
            score += q_h[d] * k_t[d];
        }
        score = simd_sum(score) * attn_scale;

        float prev_max = max_score;
        max_score = max(max_score, score);
        float exp_val = exp(score - max_score);
        float scale_prev = exp(prev_max - max_score);
        sum_exp = sum_exp * scale_prev + exp_val;

        for (uint d = tid; d < head_dim; d += 32) {
            thread_accum[d] = thread_accum[d] * scale_prev + exp_val * v_t[d];
        }
    }

    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    for (uint d = tid; d < head_dim; d += 32) {
        out_h[d] = thread_accum[d] * inv_sum;
    }
}

kernel void kernel_swiglu(
    device float* gate               [[buffer(0)]],
    device const float* up           [[buffer(1)]],
    constant uint& hidden_dim        [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid < hidden_dim) {
        float g = gate[tid];
        float silu = g / (1.0f + exp(-g));
        gate[tid] = silu * up[tid];
    }
}

kernel void kernel_add_residual(
    device float* x                  [[buffer(0)]],
    device const float* proj         [[buffer(1)]],
    constant uint& dim               [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid < dim) {
        x[tid] += proj[tid];
    }
}

kernel void kernel_kv_write(
    device float* k_cache            [[buffer(0)]],
    device float* v_cache            [[buffer(1)]],
    device const float* k            [[buffer(2)]],
    device const float* v            [[buffer(3)]],
    constant uint& layer             [[buffer(4)]],
    constant uint& slot              [[buffer(5)]],
    constant uint& max_seq           [[buffer(6)]],
    constant uint& kv_dim            [[buffer(7)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid < kv_dim) {
        uint offset = layer * max_seq * kv_dim + slot * kv_dim + tid;
        k_cache[offset] = k[tid];
        v_cache[offset] = v[tid];
    }
}
)";

int metal_init(void) {
    if (g_device != nil) {
        return 0;
    }

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        return -1;
    }

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        return -2;
    }

    id<MTLLibrary> library = nil;
    NSError* error = nil;

    // 1. Try loading pre-compiled kernels.metallib
    NSFileManager* fm = [NSFileManager defaultManager];
    NSArray* possiblePaths = @[
        @"pkg/metal/kernels.metallib",
        @"kernels.metallib",
        [[[NSBundle mainBundle] bundlePath] stringByAppendingPathComponent:@"kernels.metallib"]
    ];

    for (NSString* path in possiblePaths) {
        if ([fm fileExistsAtPath:path]) {
            NSURL* url = [NSURL fileURLWithPath:path];
            library = [g_device newLibraryWithURL:url error:&error];
            if (library) break;
        }
    }

    // 2. Fallback to JIT runtime compilation
    if (!library) {
        NSString* src = [NSString stringWithUTF8String:METAL_SOURCE];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop
        library = [g_device newLibraryWithSource:src options:options error:&error];
    }

    if (!library) {
        NSLog(@"Metal shader compilation failed: %@", error);
        return -3;
    }

    id<MTLFunction> fn_f32      = [library newFunctionWithName:@"gemv_f32"];
    id<MTLFunction> fn_f16      = [library newFunctionWithName:@"gemv_f16"];
    id<MTLFunction> fn_q4_0     = [library newFunctionWithName:@"gemv_q4_0"];
    id<MTLFunction> fn_q8_0     = [library newFunctionWithName:@"gemv_q8_0"];
    id<MTLFunction> fn_q4_k     = [library newFunctionWithName:@"gemv_q4_k"];
    id<MTLFunction> fn_q6_k     = [library newFunctionWithName:@"gemv_q6_k"];
    id<MTLFunction> fn_q2_k     = [library newFunctionWithName:@"gemv_q2_k"];
    id<MTLFunction> fn_q3_k     = [library newFunctionWithName:@"gemv_q3_k"];

    id<MTLFunction> fn_gemm_q4_0 = [library newFunctionWithName:@"gemm_q4_0_batched"];
    id<MTLFunction> fn_gemm_q8_0 = [library newFunctionWithName:@"gemm_q8_0_batched"];
    id<MTLFunction> fn_gemm_q4_k = [library newFunctionWithName:@"gemm_q4_k_batched"];
    id<MTLFunction> fn_gemm_q6_k = [library newFunctionWithName:@"gemm_q6_k_batched"];

    id<MTLFunction> fn_rmsnorm  = [library newFunctionWithName:@"kernel_rmsnorm"];
    id<MTLFunction> fn_rope     = [library newFunctionWithName:@"kernel_rope"];
    id<MTLFunction> fn_attn     = [library newFunctionWithName:@"kernel_attention_gqa"];
    id<MTLFunction> fn_kv_write = [library newFunctionWithName:@"kernel_kv_write"];
    id<MTLFunction> fn_swiglu   = [library newFunctionWithName:@"kernel_swiglu"];
    id<MTLFunction> fn_residual = [library newFunctionWithName:@"kernel_add_residual"];

    g_pipeline_f32      = [g_device newComputePipelineStateWithFunction:fn_f32 error:&error];
    g_pipeline_f16      = [g_device newComputePipelineStateWithFunction:fn_f16 error:&error];
    g_pipeline_q4_0     = [g_device newComputePipelineStateWithFunction:fn_q4_0 error:&error];
    g_pipeline_q8_0     = [g_device newComputePipelineStateWithFunction:fn_q8_0 error:&error];
    g_pipeline_q4_k     = [g_device newComputePipelineStateWithFunction:fn_q4_k error:&error];
    g_pipeline_q6_k     = [g_device newComputePipelineStateWithFunction:fn_q6_k error:&error];
    if (fn_q2_k) g_pipeline_q2_k = [g_device newComputePipelineStateWithFunction:fn_q2_k error:&error];
    if (fn_q3_k) g_pipeline_q3_k = [g_device newComputePipelineStateWithFunction:fn_q3_k error:&error];

    g_pipeline_gemm_q4_0 = [g_device newComputePipelineStateWithFunction:fn_gemm_q4_0 error:&error];
    g_pipeline_gemm_q8_0 = [g_device newComputePipelineStateWithFunction:fn_gemm_q8_0 error:&error];
    g_pipeline_gemm_q4_k = [g_device newComputePipelineStateWithFunction:fn_gemm_q4_k error:&error];
    g_pipeline_gemm_q6_k = [g_device newComputePipelineStateWithFunction:fn_gemm_q6_k error:&error];

    id<MTLFunction> fn_fused_q4_0 = [library newFunctionWithName:@"gemv_fused_gate_up_swiglu_q4_0"];
    id<MTLFunction> fn_fused_q8_0 = [library newFunctionWithName:@"gemv_fused_gate_up_swiglu_q8_0"];
    id<MTLFunction> fn_fused_q4_k = [library newFunctionWithName:@"gemv_fused_gate_up_swiglu_q4_k"];
    id<MTLFunction> fn_fused_q6_k = [library newFunctionWithName:@"gemv_fused_gate_up_swiglu_q6_k"];
    id<MTLFunction> fn_argmax     = [library newFunctionWithName:@"kernel_sample_argmax"];

    if (fn_fused_q4_0) g_pipeline_fused_gate_up_q4_0 = [g_device newComputePipelineStateWithFunction:fn_fused_q4_0 error:&error];
    if (fn_fused_q8_0) g_pipeline_fused_gate_up_q8_0 = [g_device newComputePipelineStateWithFunction:fn_fused_q8_0 error:&error];
    if (fn_fused_q4_k) g_pipeline_fused_gate_up_q4_k = [g_device newComputePipelineStateWithFunction:fn_fused_q4_k error:&error];
    if (fn_fused_q6_k) g_pipeline_fused_gate_up_q6_k = [g_device newComputePipelineStateWithFunction:fn_fused_q6_k error:&error];
    if (fn_argmax)     g_pipeline_sample_argmax      = [g_device newComputePipelineStateWithFunction:fn_argmax error:&error];

    g_pipeline_rmsnorm  = [g_device newComputePipelineStateWithFunction:fn_rmsnorm error:&error];
    g_pipeline_rope     = [g_device newComputePipelineStateWithFunction:fn_rope error:&error];
    if (fn_attn) {
        g_pipeline_attn = [g_device newComputePipelineStateWithFunction:fn_attn error:&error];
    }
    if (fn_kv_write) {
        g_pipeline_kv_write = [g_device newComputePipelineStateWithFunction:fn_kv_write error:&error];
    }
    g_pipeline_swiglu   = [g_device newComputePipelineStateWithFunction:fn_swiglu error:&error];
    g_pipeline_residual = [g_device newComputePipelineStateWithFunction:fn_residual error:&error];

    if (!g_pipeline_f32 || !g_pipeline_q4_0 || !g_pipeline_rmsnorm) {
        return -4;
    }

    return 0;
}

bool metal_is_available(void) {
    return g_device != nil && g_queue != nil && g_pipeline_q4_0 != nil;
}

int metal_alloc_buffers(uint32_t dim, uint32_t hidden_dim, uint32_t kv_dim,
                        uint32_t vocab_size, uint32_t num_layers, uint32_t max_seq) {
    if (!metal_is_available()) return -1;

    size_t kv_cache_bytes = (size_t)num_layers * max_seq * kv_dim * sizeof(float);
    g_k_cache = [g_device newBufferWithLength:kv_cache_bytes options:MTLResourceStorageModeShared];
    g_v_cache = [g_device newBufferWithLength:kv_cache_bytes options:MTLResourceStorageModeShared];

    g_buf_x         = [g_device newBufferWithLength:dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_xb        = [g_device newBufferWithLength:dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_q         = [g_device newBufferWithLength:dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_k         = [g_device newBufferWithLength:kv_dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_v         = [g_device newBufferWithLength:kv_dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_attn_out  = [g_device newBufferWithLength:dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_attn_proj = [g_device newBufferWithLength:dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_gate      = [g_device newBufferWithLength:hidden_dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_up        = [g_device newBufferWithLength:hidden_dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_down      = [g_device newBufferWithLength:dim * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_logits    = [g_device newBufferWithLength:vocab_size * sizeof(float) options:MTLResourceStorageModeShared];

    if (!g_k_cache || !g_v_cache || !g_buf_x || !g_buf_logits) return -2;
    return 0;
}

int metal_kv_write(const float* k, const float* v, uint32_t layer, uint32_t slot, uint32_t max_seq, uint32_t kv_dim) {
    if (!metal_is_available() || !g_pipeline_kv_write || !g_k_cache || !g_v_cache) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_k = [g_device newBufferWithBytesNoCopy:(void*)k length:kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_v = [g_device newBufferWithBytesNoCopy:(void*)v length:kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        if (!buf_k || !buf_v) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_kv_write];
        [enc setBuffer:g_k_cache offset:0 atIndex:0];
        [enc setBuffer:g_v_cache offset:0 atIndex:1];
        [enc setBuffer:buf_k offset:0 atIndex:2];
        [enc setBuffer:buf_v offset:0 atIndex:3];
        [enc setBytes:&layer length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&slot length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&max_seq length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&kv_dim length:sizeof(uint32_t) atIndex:7];

        MTLSize tgs = MTLSizeMake((kv_dim + 31) / 32, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

void metal_begin_batch(void) {
    if (!g_batch_cmd && g_queue) {
        g_batch_cmd = [g_queue commandBuffer];
        g_batch_encoder = [g_batch_cmd computeCommandEncoder];
    }
}

void metal_end_batch(void) {
    if (g_batch_encoder) {
        [g_batch_encoder endEncoding];
        g_batch_encoder = nil;
    }
    if (g_batch_cmd) {
        [g_batch_cmd commit];
        [g_batch_cmd waitUntilCompleted];
        g_batch_cmd = nil;
    }
}

static inline int run_gemv(id<MTLComputePipelineState> pipeline,
                          float* y, const float* x, const void* w,
                          size_t w_bytes, uint32_t rows, uint32_t cols) {
    if (!metal_is_available()) {
        return -1;
    }

    @autoreleasepool {
        id<MTLBuffer> buf_y = [g_device newBufferWithBytesNoCopy:y
                                                          length:rows * sizeof(float)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        id<MTLBuffer> buf_x = [g_device newBufferWithBytesNoCopy:(void*)x
                                                          length:cols * sizeof(float)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        id<MTLBuffer> buf_w = [g_device newBufferWithBytesNoCopy:(void*)w
                                                          length:w_bytes
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        if (!buf_y || !buf_x || !buf_w) {
            return -2;
        }

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmdBuffer = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = is_batched ? g_batch_encoder : [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_y offset:0 atIndex:0];
        [encoder setBuffer:buf_x offset:0 atIndex:1];
        [encoder setBuffer:buf_w offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&cols length:sizeof(uint32_t) atIndex:4];

        MTLSize threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(128, 1, 1);

        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

        if (!is_batched) {
            [encoder endEncoding];
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }
    }
    return 0;
}

metal_buffer_t metal_create_buffer(const void* ptr, size_t bytes) {
    if (!metal_is_available() || !ptr || bytes == 0) return NULL;
    id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:(void*)ptr
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    return (__bridge_retained void*)buf;
}

void metal_release_buffer(metal_buffer_t buf) {
    if (buf) {
        id<MTLBuffer> mtl_buf = (__bridge_transfer id<MTLBuffer>)buf;
        mtl_buf = nil;
    }
}

int metal_gemv_buf(int quant_type, float* y, const float* x, metal_buffer_t w_buf, uint32_t rows, uint32_t cols) {
    if (!metal_is_available() || !w_buf) return -1;

    id<MTLComputePipelineState> pipeline = nil;
    switch (quant_type) {
        case 0: pipeline = g_pipeline_f32; break;
        case 1: pipeline = g_pipeline_f16; break;
        case 2: pipeline = g_pipeline_q4_0; break;
        case 3: pipeline = g_pipeline_q8_0; break;
        case 10: pipeline = g_pipeline_q2_k; break;
        case 11: pipeline = g_pipeline_q3_k; break;
        case 12: pipeline = g_pipeline_q4_k; break;
        case 14: pipeline = g_pipeline_q6_k; break;
        default: return -2;
    }

    @autoreleasepool {
        id<MTLBuffer> buf_y = [g_device newBufferWithBytesNoCopy:y
                                                          length:rows * sizeof(float)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        id<MTLBuffer> buf_x = [g_device newBufferWithBytesNoCopy:(void*)x
                                                          length:cols * sizeof(float)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        id<MTLBuffer> buf_w = (__bridge id<MTLBuffer>)w_buf;

        if (!buf_y || !buf_x || !buf_w) return -3;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmdBuffer = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = is_batched ? g_batch_encoder : [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_y offset:0 atIndex:0];
        [encoder setBuffer:buf_x offset:0 atIndex:1];
        [encoder setBuffer:buf_w offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&cols length:sizeof(uint32_t) atIndex:4];

        MTLSize threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(128, 1, 1);

        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

        if (!is_batched) {
            [encoder endEncoding];
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }
    }
    return 0;
}

int metal_gemv_f32(float* y, const float* x, const float* w, uint32_t rows, uint32_t cols) {
    return run_gemv(g_pipeline_f32, y, x, w, rows * cols * sizeof(float), rows, cols);
}

int metal_gemv_f16(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols) {
    return run_gemv(g_pipeline_f16, y, x, w, rows * cols * 2, rows, cols);
}

int metal_gemv_q4_0(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 32) * 18);
    return run_gemv(g_pipeline_q4_0, y, x, w, w_bytes, rows, cols);
}

int metal_gemv_q8_0(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 32) * 34);
    return run_gemv(g_pipeline_q8_0, y, x, w, w_bytes, rows, cols);
}

int metal_gemv_q4_k(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 256) * 144);
    return run_gemv(g_pipeline_q4_k, y, x, w, w_bytes, rows, cols);
}

int metal_gemv_q6_k(float* y, const float* x, const void* w, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 256) * 210);
    return run_gemv(g_pipeline_q6_k, y, x, w, w_bytes, rows, cols);
}

static inline int run_gemm(id<MTLComputePipelineState> pipeline,
                           float* y, const float* x, const void* w,
                           size_t w_bytes, uint32_t batch_size, uint32_t rows, uint32_t cols) {
    if (!metal_is_available()) {
        return -1;
    }

    @autoreleasepool {
        id<MTLBuffer> buf_y = [g_device newBufferWithBytesNoCopy:y length:batch_size * rows * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_x = [g_device newBufferWithBytesNoCopy:(void*)x length:batch_size * cols * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_w = [g_device newBufferWithBytesNoCopy:(void*)w length:w_bytes options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_y || !buf_x || !buf_w) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_y offset:0 atIndex:0];
        [enc setBuffer:buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_w offset:0 atIndex:2];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&rows length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&cols length:sizeof(uint32_t) atIndex:5];

        MTLSize tgs = MTLSizeMake(batch_size, rows, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_gemm_q4_0(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 32) * 18);
    return run_gemm(g_pipeline_gemm_q4_0, y, x, w, w_bytes, batch_size, rows, cols);
}

int metal_gemm_q8_0(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 32) * 34);
    return run_gemm(g_pipeline_gemm_q8_0, y, x, w, w_bytes, batch_size, rows, cols);
}

int metal_gemm_q4_k(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 256) * 144);
    return run_gemm(g_pipeline_gemm_q4_k, y, x, w, w_bytes, batch_size, rows, cols);
}

int metal_gemm_q6_k(float* y, const float* x, const void* w, uint32_t batch_size, uint32_t rows, uint32_t cols) {
    size_t w_bytes = (size_t)rows * ((cols / 256) * 210);
    return run_gemm(g_pipeline_gemm_q6_k, y, x, w, w_bytes, batch_size, rows, cols);
}

int metal_rmsnorm(float* out, const float* x, const float* weight, uint32_t dim, float eps) {
    if (!metal_is_available()) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_out = [g_device newBufferWithBytesNoCopy:out length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_x   = [g_device newBufferWithBytesNoCopy:(void*)x length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_w   = [g_device newBufferWithBytesNoCopy:(void*)weight length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_rmsnorm];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_w offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];

        MTLSize tgs = MTLSizeMake(1, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_attention_gqa(float* attn_out, const float* q, const float* k_cache, const float* v_cache,
                        uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
                        uint32_t active_context, float attn_scale) {
    if (!metal_is_available() || !g_pipeline_attn) return -1;

    @autoreleasepool {
        uint32_t q_size = num_heads * head_dim * sizeof(float);
        uint32_t kv_size = active_context * num_kv_heads * head_dim * sizeof(float);

        id<MTLBuffer> buf_out = [g_device newBufferWithBytesNoCopy:attn_out length:q_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_q   = [g_device newBufferWithBytesNoCopy:(void*)q length:q_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_k   = g_k_cache ? g_k_cache : [g_device newBufferWithBytesNoCopy:(void*)k_cache length:kv_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_v   = g_v_cache ? g_v_cache : [g_device newBufferWithBytesNoCopy:(void*)v_cache length:kv_size options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_out || !buf_q || !buf_k || !buf_v) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_attn];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_q offset:0 atIndex:1];
        [enc setBuffer:buf_k offset:0 atIndex:2];
        [enc setBuffer:buf_v offset:0 atIndex:3];
        [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&active_context length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&attn_scale length:sizeof(float) atIndex:8];

        MTLSize tgs = MTLSizeMake(num_heads, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_rope(float* q, float* k, uint32_t pos, uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim, float theta) {
    if (!metal_is_available()) return -1;

    @autoreleasepool {
        uint32_t q_size = num_heads * head_dim * sizeof(float);
        uint32_t k_size = num_kv_heads * head_dim * sizeof(float);

        id<MTLBuffer> buf_q = [g_device newBufferWithBytesNoCopy:q length:q_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_k = [g_device newBufferWithBytesNoCopy:k length:k_size options:MTLResourceStorageModeShared deallocator:nil];

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_rope];
        [enc setBuffer:buf_q offset:0 atIndex:0];
        [enc setBuffer:buf_k offset:0 atIndex:1];
        [enc setBytes:&pos length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&theta length:sizeof(float) atIndex:6];

        uint total_threads = (num_heads > num_kv_heads ? num_heads : num_kv_heads) * (head_dim / 2);
        MTLSize tgs = MTLSizeMake((total_threads + 31) / 32, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_swiglu(float* gate, const float* up, uint32_t hidden_dim) {
    if (!metal_is_available()) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_g = [g_device newBufferWithBytesNoCopy:gate length:hidden_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_u = [g_device newBufferWithBytesNoCopy:(void*)up length:hidden_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_swiglu];
        [enc setBuffer:buf_g offset:0 atIndex:0];
        [enc setBuffer:buf_u offset:0 atIndex:1];
        [enc setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:2];

        MTLSize tgs = MTLSizeMake((hidden_dim + 31) / 32, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_add_residual(float* x, const float* proj, uint32_t dim) {
    if (!metal_is_available()) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_x = [g_device newBufferWithBytesNoCopy:x length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_p = [g_device newBufferWithBytesNoCopy:(void*)proj length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_residual];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_p offset:0 atIndex:1];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:2];

        MTLSize tgs = MTLSizeMake((dim + 31) / 32, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

static inline id<MTLComputePipelineState> get_pipeline(int quant_type) {
    switch (quant_type) {
        case 0: return g_pipeline_f32;
        case 1: return g_pipeline_f16;
        case 2: return g_pipeline_q4_0;
        case 3: return g_pipeline_q8_0;
        case 12: return g_pipeline_q4_k;
        case 14: return g_pipeline_q6_k;
        default: return nil;
    }
}

static inline id<MTLComputePipelineState> get_fused_gate_up_pipeline(int quant_type) {
    switch (quant_type) {
        case 2: return g_pipeline_fused_gate_up_q4_0;
        case 3: return g_pipeline_fused_gate_up_q8_0;
        case 12: return g_pipeline_fused_gate_up_q4_k;
        case 14: return g_pipeline_fused_gate_up_q6_k;
        default: return nil;
    }
}

static inline void encode_gemv_buf(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> buf_y, id<MTLBuffer> buf_x, id<MTLBuffer> buf_w,
                                  uint32_t rows, uint32_t cols) {
    if (!pipeline || !buf_y || !buf_x || !buf_w) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_y offset:0 atIndex:0];
    [enc setBuffer:buf_x offset:0 atIndex:1];
    [enc setBuffer:buf_w offset:0 atIndex:2];
    [enc setBytes:&rows length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&cols length:sizeof(uint32_t) atIndex:4];

    MTLSize threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(128, 1, 1);
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
}

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
) {
    if (!metal_is_available() || !initial_x || !out_logits || !layers) return -1;
    if (!g_buf_x || !g_buf_logits || !output_norm_buf || !output_weight_buf) return -2;

    @autoreleasepool {
        memcpy(g_buf_x.contents, initial_x, dim * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint32_t total_rope_threads = (num_heads > num_kv_heads ? num_heads : num_kv_heads) * (head_dim / 2);

        for (uint32_t l = 0; l < num_layers; l++) {
            const metal_layer_weights_t* lw = &layers[l];
            id<MTLBuffer> buf_attn_norm = (__bridge id<MTLBuffer>)lw->attn_norm;
            id<MTLBuffer> buf_ffn_norm  = (__bridge id<MTLBuffer>)lw->ffn_norm;

            // 1. RMSNorm on input x -> xb
            [enc setComputePipelineState:g_pipeline_rmsnorm];
            [enc setBuffer:g_buf_xb offset:0 atIndex:0];
            [enc setBuffer:g_buf_x offset:0 atIndex:1];
            [enc setBuffer:buf_attn_norm offset:0 atIndex:2];
            [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:(void*)&norm_eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 2. Q, K, V Projections
            encode_gemv_buf(enc, get_pipeline(lw->wq_type), g_buf_q, g_buf_xb, (__bridge id<MTLBuffer>)lw->wq, dim, dim);
            encode_gemv_buf(enc, get_pipeline(lw->wk_type), g_buf_k, g_buf_xb, (__bridge id<MTLBuffer>)lw->wk, kv_dim, dim);
            encode_gemv_buf(enc, get_pipeline(lw->wv_type), g_buf_v, g_buf_xb, (__bridge id<MTLBuffer>)lw->wv, kv_dim, dim);

            // 3. RoPE
            [enc setComputePipelineState:g_pipeline_rope];
            [enc setBuffer:g_buf_q offset:0 atIndex:0];
            [enc setBuffer:g_buf_k offset:0 atIndex:1];
            [enc setBytes:(void*)&pos length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:(void*)&num_heads length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:(void*)&num_kv_heads length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:(void*)&head_dim length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:(void*)&rope_theta length:sizeof(float) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((total_rope_threads + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 4. KVWrite to GPU resident cache
            [enc setComputePipelineState:g_pipeline_kv_write];
            [enc setBuffer:g_k_cache offset:0 atIndex:0];
            [enc setBuffer:g_v_cache offset:0 atIndex:1];
            [enc setBuffer:g_buf_k offset:0 atIndex:2];
            [enc setBuffer:g_buf_v offset:0 atIndex:3];
            [enc setBytes:(void*)&l length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:(void*)&slot length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:(void*)&max_seq length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:(void*)&kv_dim length:sizeof(uint32_t) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 5. FlashAttention (GQA)
            [enc setComputePipelineState:g_pipeline_attn];
            [enc setBuffer:g_buf_attn_out offset:0 atIndex:0];
            [enc setBuffer:g_buf_q offset:0 atIndex:1];
            [enc setBuffer:g_k_cache offset:0 atIndex:2];
            [enc setBuffer:g_v_cache offset:0 atIndex:3];
            [enc setBytes:(void*)&num_heads length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:(void*)&num_kv_heads length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:(void*)&head_dim length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:(void*)&active_context length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:(void*)&attn_scale length:sizeof(float) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 6. WO Projection
            encode_gemv_buf(enc, get_pipeline(lw->wo_type), g_buf_attn_proj, g_buf_attn_out, (__bridge id<MTLBuffer>)lw->wo, dim, dim);

            // 7. AddResidual (x += attn_proj)
            [enc setComputePipelineState:g_pipeline_residual];
            [enc setBuffer:g_buf_x offset:0 atIndex:0];
            [enc setBuffer:g_buf_attn_proj offset:0 atIndex:1];
            [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 8. FFN RMSNorm (x -> xb)
            [enc setComputePipelineState:g_pipeline_rmsnorm];
            [enc setBuffer:g_buf_xb offset:0 atIndex:0];
            [enc setBuffer:g_buf_x offset:0 atIndex:1];
            [enc setBuffer:buf_ffn_norm offset:0 atIndex:2];
            [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:(void*)&norm_eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 9. Fused Gate-Up + SwiGLU in 1 kernel pass directly into g_buf_gate
            id<MTLComputePipelineState> fused_gate_up = get_fused_gate_up_pipeline(lw->ffn_gate_type);
            if (fused_gate_up && lw->ffn_gate_type == lw->ffn_up_type) {
                [enc setComputePipelineState:fused_gate_up];
                [enc setBuffer:g_buf_gate offset:0 atIndex:0];
                [enc setBuffer:g_buf_xb offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ffn_gate offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ffn_up offset:0 atIndex:3];
                [enc setBytes:(void*)&hidden_dim length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:5];
                MTLSize tgs = MTLSizeMake((hidden_dim + 7) / 8, 1, 1);
                MTLSize tpg = MTLSizeMake(128, 1, 1);
                [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];
            } else {
                encode_gemv_buf(enc, get_pipeline(lw->ffn_gate_type), g_buf_gate, g_buf_xb, (__bridge id<MTLBuffer>)lw->ffn_gate, hidden_dim, dim);
                encode_gemv_buf(enc, get_pipeline(lw->ffn_up_type), g_buf_up, g_buf_xb, (__bridge id<MTLBuffer>)lw->ffn_up, hidden_dim, dim);
                [enc setComputePipelineState:g_pipeline_swiglu];
                [enc setBuffer:g_buf_gate offset:0 atIndex:0];
                [enc setBuffer:g_buf_up offset:0 atIndex:1];
                [enc setBytes:(void*)&hidden_dim length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((hidden_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            // 11. FFN Down Projection
            encode_gemv_buf(enc, get_pipeline(lw->ffn_down_type), g_buf_down, g_buf_gate, (__bridge id<MTLBuffer>)lw->ffn_down, dim, hidden_dim);

            // 12. AddResidual (x += ffn_down)
            [enc setComputePipelineState:g_pipeline_residual];
            [enc setBuffer:g_buf_x offset:0 atIndex:0];
            [enc setBuffer:g_buf_down offset:0 atIndex:1];
            [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        // Final RMSNorm on x -> xb
        id<MTLBuffer> buf_out_norm = (__bridge id<MTLBuffer>)output_norm_buf;
        [enc setComputePipelineState:g_pipeline_rmsnorm];
        [enc setBuffer:g_buf_xb offset:0 atIndex:0];
        [enc setBuffer:g_buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_out_norm offset:0 atIndex:2];
        [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:(void*)&norm_eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // Final Output Logits Projection (Zero-copy directly to host out_logits)
        id<MTLBuffer> buf_logits = [g_device newBufferWithBytesNoCopy:(void*)out_logits length:vocab_size * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        encode_gemv_buf(enc, get_pipeline(output_weight_type), buf_logits, g_buf_xb, (__bridge id<MTLBuffer>)output_weight_buf, vocab_size, dim);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return 0;
}

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
) {
    if (!metal_is_available()) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_x         = [g_device newBufferWithBytesNoCopy:x length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_xnorm     = [g_device newBufferWithBytesNoCopy:xnorm length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_q         = [g_device newBufferWithBytesNoCopy:q length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_k         = [g_device newBufferWithBytesNoCopy:k length:kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_v         = [g_device newBufferWithBytesNoCopy:v length:kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_attn_out  = [g_device newBufferWithBytesNoCopy:attn_out length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_attn_proj = [g_device newBufferWithBytesNoCopy:attn_proj length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_ffn_gate  = [g_device newBufferWithBytesNoCopy:gate_act length:hidden_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_ffn_up    = [g_device newBufferWithBytesNoCopy:up_act length:hidden_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_ffn_down  = [g_device newBufferWithBytesNoCopy:ffn_down_act length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        id<MTLBuffer> buf_attn_norm = [g_device newBufferWithBytesNoCopy:(void*)attn_norm length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_ffn_norm  = [g_device newBufferWithBytesNoCopy:(void*)ffn_norm length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        // 1. RMSNorm on input x -> xnorm
        [enc setComputePipelineState:g_pipeline_rmsnorm];
        [enc setBuffer:buf_xnorm offset:0 atIndex:0];
        [enc setBuffer:buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_attn_norm offset:0 atIndex:2];
        [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:(void*)&norm_eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // 2. Q, K, V Projections
        encode_gemv_buf(enc, get_pipeline(wq_type), buf_q, buf_xnorm, (__bridge id<MTLBuffer>)wq, dim, dim);
        encode_gemv_buf(enc, get_pipeline(wk_type), buf_k, buf_xnorm, (__bridge id<MTLBuffer>)wk, kv_dim, dim);
        encode_gemv_buf(enc, get_pipeline(wv_type), buf_v, buf_xnorm, (__bridge id<MTLBuffer>)wv, kv_dim, dim);

        // 3. RoPE
        [enc setComputePipelineState:g_pipeline_rope];
        [enc setBuffer:buf_q offset:0 atIndex:0];
        [enc setBuffer:buf_k offset:0 atIndex:1];
        [enc setBytes:(void*)&pos length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:(void*)&num_heads length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:(void*)&num_kv_heads length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:(void*)&head_dim length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:(void*)&rope_theta length:sizeof(float) atIndex:6];
        uint32_t total_rope_threads = (num_heads > num_kv_heads ? num_heads : num_kv_heads) * (head_dim / 2);
        [enc dispatchThreadgroups:MTLSizeMake((total_rope_threads + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // 4. KVWrite to GPU resident cache
        [enc setComputePipelineState:g_pipeline_kv_write];
        [enc setBuffer:g_k_cache offset:0 atIndex:0];
        [enc setBuffer:g_v_cache offset:0 atIndex:1];
        [enc setBuffer:buf_k offset:0 atIndex:2];
        [enc setBuffer:buf_v offset:0 atIndex:3];
        [enc setBytes:(void*)&layer_idx length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:(void*)&slot length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:(void*)&max_seq length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:(void*)&kv_dim length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // 5. FlashAttention (GQA)
        [enc setComputePipelineState:g_pipeline_attn];
        [enc setBuffer:buf_attn_out offset:0 atIndex:0];
        [enc setBuffer:buf_q offset:0 atIndex:1];
        [enc setBuffer:g_k_cache offset:0 atIndex:2];
        [enc setBuffer:g_v_cache offset:0 atIndex:3];
        [enc setBytes:(void*)&num_heads length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:(void*)&num_kv_heads length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:(void*)&head_dim length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:(void*)&active_context length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:(void*)&attn_scale length:sizeof(float) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // 6. WO Projection
        encode_gemv_buf(enc, get_pipeline(wo_type), buf_attn_proj, buf_attn_out, (__bridge id<MTLBuffer>)wo, dim, dim);

        // 7. AddResidual (x += attn_proj)
        [enc setComputePipelineState:g_pipeline_residual];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_attn_proj offset:0 atIndex:1];
        [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // 8. FFN RMSNorm (x -> xnorm)
        [enc setComputePipelineState:g_pipeline_rmsnorm];
        [enc setBuffer:buf_xnorm offset:0 atIndex:0];
        [enc setBuffer:buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_ffn_norm offset:0 atIndex:2];
        [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:(void*)&norm_eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // 9. FFN Gate & Up Projections
        encode_gemv_buf(enc, get_pipeline(ffn_gate_type), buf_ffn_gate, buf_xnorm, (__bridge id<MTLBuffer>)ffn_gate, hidden_dim, dim);
        encode_gemv_buf(enc, get_pipeline(ffn_up_type), buf_ffn_up, buf_xnorm, (__bridge id<MTLBuffer>)ffn_up, hidden_dim, dim);

        // 10. SwiGLU (gate = silu(gate) * up)
        [enc setComputePipelineState:g_pipeline_swiglu];
        [enc setBuffer:buf_ffn_gate offset:0 atIndex:0];
        [enc setBuffer:buf_ffn_up offset:0 atIndex:1];
        [enc setBytes:(void*)&hidden_dim length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((hidden_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // 11. FFN Down Projection
        encode_gemv_buf(enc, get_pipeline(ffn_down_type), buf_ffn_down, buf_ffn_gate, (__bridge id<MTLBuffer>)ffn_down, dim, hidden_dim);

        // 12. AddResidual (x += ffn_down)
        [enc setComputePipelineState:g_pipeline_residual];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_ffn_down offset:0 atIndex:1];
        [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}
