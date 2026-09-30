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
static id<MTLComputePipelineState> g_pipeline_gemm_fused_gate_up_q4_k = nil;

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
static id<MTLComputePipelineState> g_pipeline_residual_rmsnorm = nil;

static id<MTLComputePipelineState> g_pipeline_conv1d_step         = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_step            = nil;
static id<MTLComputePipelineState> g_pipeline_qwen35_split_q_gate = nil;
static id<MTLComputePipelineState> g_pipeline_qwen35_attn_gate     = nil;
static id<MTLComputePipelineState> g_pipeline_qwen35_norm_rope    = nil;
static id<MTLComputePipelineState> g_pipeline_embed_lookup_q4_k = nil;
static id<MTLComputePipelineState> g_pipeline_qwen35_fused_split_norm_rope = nil;

static id<MTLComputePipelineState> g_pipeline_conv1d_batch          = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_batch             = nil;
static id<MTLComputePipelineState> g_pipeline_attention_gqa_batch   = nil;
static id<MTLComputePipelineState> g_pipeline_kv_write_batch        = nil;
static id<MTLComputePipelineState> g_pipeline_split_q_gate_batch    = nil;
static id<MTLComputePipelineState> g_pipeline_rope_norm_batch       = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_batch         = nil;
static id<MTLComputePipelineState> g_pipeline_residual_rmsnorm_batch = nil;
static id<MTLComputePipelineState> g_pipeline_embed_lookup_q4_k_batch = nil;

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
static id<MTLBuffer> g_buf_token  = nil;

static id<MTLBuffer> g_buf_ssm_gate         = nil;
static id<MTLBuffer> g_buf_ssm_qkv          = nil;
static id<MTLBuffer> g_buf_ssm_conv_out     = nil;
static id<MTLBuffer> g_buf_ssm_alpha        = nil;
static id<MTLBuffer> g_buf_ssm_beta         = nil;
static id<MTLBuffer> g_buf_ssm_out          = nil;
static id<MTLBuffer> g_buf_q_raw            = nil;
static id<MTLBuffer> g_buf_qwen35_attn_gate = nil;
static id<MTLBuffer> g_ssm_conv_state       = nil;
static id<MTLBuffer> g_ssm_state            = nil;

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

struct block_q2_k {
    uint8_t scales[16];
    uint8_t qs[64];
    half    d;
    half    dmin;
};

struct block_q3_k {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half    d;
};

static inline void get_scale_min_k4(int j, device const uint8_t* q, thread float& d_val, thread float& m_val, float d, float dmin) {
    if (j < 4) {
        d_val = float(q[j] & 63) * d;
        m_val = float(q[j + 4] & 63) * dmin;
    } else {
        d_val = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) * d;
        m_val = float((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)) * dmin;
    }
}

// --- 128-Thread Cooperative 8-Row SIMD Vectorized GEMV Kernels ---

// 128-Thread 8-Row Vectorized F32 GEMV (128-bit Vector Loads)
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

    threadgroup float tg_sums[4][8];

    uint cols4 = cols / 4;
    device const float4* w4 = (device const float4*)w;
    device const float4* r_ptrs4[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_ptrs4[i] = (r < rows) ? (w4 + r * cols4) : (w4 + r0 * cols4);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint c = tid; c < cols4; c += 128) {
        uint col = c * 4;
        float4 x_val = float4(x[col], x[col+1], x[col+2], x[col+3]);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sums[i] += dot(r_ptrs4[i][c], x_val);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_sums[simd_id][i] = sums[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float total = tg_sums[0][i] + tg_sums[1][i] + tg_sums[2][i] + tg_sums[3][i];
            if (r0 + i < rows) y[r0 + i] = total;
        }
    }
}

// 128-Thread 8-Row Vectorized F16 GEMV (64-bit/128-bit Vector Loads)
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

    threadgroup float tg_sums[4][8];

    uint cols4 = cols / 4;
    device const half4* w4 = (device const half4*)w;
    device const half4* r_ptrs4[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_ptrs4[i] = (r < rows) ? (w4 + r * cols4) : (w4 + r0 * cols4);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint c = tid; c < cols4; c += 128) {
        uint col = c * 4;
        float4 x_val = float4(x[col], x[col+1], x[col+2], x[col+3]);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 w_val = float4(r_ptrs4[i][c]);
            sums[i] += dot(w_val, x_val);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_sums[simd_id][i] = sums[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float total = tg_sums[0][i] + tg_sums[1][i] + tg_sums[2][i] + tg_sums[3][i];
            if (r0 + i < rows) y[r0 + i] = total;
        }
    }
}

// 128-Thread 8-Row Vectorized Q4_0 GEMV
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

    threadgroup float tg_sums[4][8];

    uint num_blocks = cols / 32;
    device const block_q4_0* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 128) {
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

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_sums[simd_id][i] = sums[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float total = tg_sums[0][i] + tg_sums[1][i] + tg_sums[2][i] + tg_sums[3][i];
            if (r0 + i < rows) y[r0 + i] = total;
        }
    }
}

// 128-Thread 8-Row Vectorized Q8_0 GEMV
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

    threadgroup float tg_sums[4][8];

    uint num_blocks = cols / 32;
    device const block_q8_0* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 128) {
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

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_sums[simd_id][i] = sums[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float total = tg_sums[0][i] + tg_sums[1][i] + tg_sums[2][i] + tg_sums[3][i];
            if (r0 + i < rows) y[r0 + i] = total;
        }
    }
}

// High-efficiency 8-thread/block parallel Q4_K GEMV (2 SIMDgroups, 4 rows/SIMDgroup = 8 rows/TG)
kernel void gemv_q4_k(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_k* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    const short NSG = 2;
    const short nr0 = 4;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short sgitg = tid / 32;
    const short tiisg = tid % 32;

    const short ix = tiisg / 8;  // 0...3 (which block)
    const short it = tiisg % 8;  // 0...7 (which 32-element chunk)
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0...3

    const int nb = cols / 256;

    const int first_row = (tg_idx * NSG + sgitg) * nr0;
    if (first_row >= rows) return;

    device const block_q4_k * r_w = w + first_row * nb;
    device const float      * y_in = x;

    float yl[16];
    float yh[16];

    float sumf[4] = {0.f, 0.f, 0.f, 0.f};

    device const float * y4 = y_in + ix * 256 + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        #pragma unroll
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)r_w[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)r_w[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &r_w[ib].d;

        for (short row = 0; row < nr0; row++) {
            if (first_row + row >= rows) break;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * ((acc1[0] + (1.f/256.f) * acc1[1]) * sc8[0] +
                                         (acc1[2] + (1.f/256.f) * acc1[3]) * sc8[1] * (1.f/16.f) +
                                         (acc2[0] + (1.f/256.f) * acc2[1]) * sc8[4] +
                                         (acc2[2] + (1.f/256.f) * acc2[3]) * sc8[5] * (1.f/16.f)) -
                         float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += (nb * sizeof(block_q4_k)) / 2;
            sc += (nb * sizeof(block_q4_k)) / 2;
            dh += (nb * sizeof(block_q4_k)) / 2;
        }

        y4 += 4 * 256;
    }

    for (int row = 0; row < nr0; ++row) {
        if (first_row + row >= rows) break;
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            y[first_row + row] = sum_all;
        }
    }
}

// Cooperative 32-thread SIMDgroup Q6_K kernel (handles 4 rows per SIMDgroup, 4 SIMDgroups per threadgroup = 16 rows)
kernel void gemv_q6_k(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q6_k* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    const uint NSG = 4;
    const uint nr0 = 4;
    uint sgitg = tid / 32;
    uint tiisg = tid % 32;

    uint first_row = (tg_idx * NSG + sgitg) * nr0;
    if (first_row >= rows) return;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const uint nb = cols / 256;
    float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;

    const short s_tid = tiisg / 2;
    const short ix    = tiisg % 2;
    const short ip    = s_tid / 8;
    const short il    = s_tid % 8;
    const short l0    = 4 * il;
    const short is    = 8 * ip + l0 / 16;
    const short y_offset   = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    device const block_q6_k* r_w0 = w + (first_row + 0) * nb;
    device const block_q6_k* r_w1 = (first_row + 1 < rows) ? (w + (first_row + 1) * nb) : r_w0;
    device const block_q6_k* r_w2 = (first_row + 2 < rows) ? (w + (first_row + 2) * nb) : r_w0;
    device const block_q6_k* r_w3 = (first_row + 3 < rows) ? (w + (first_row + 3) * nb) : r_w0;

    for (uint i = ix; i < nb; i += 2) {
        device const float* y_vec = x + i * 256 + y_offset;
        float4 vy0 = *(device const float4*)(y_vec + 0);
        float4 vy1 = *(device const float4*)(y_vec + 32);
        float4 vy2 = *(device const float4*)(y_vec + 64);
        float4 vy3 = *(device const float4*)(y_vec + 96);

        // Row 0
        {
            device const block_q6_k& b0 = r_w0[i];
            device const uint8_t* q1 = b0.ql + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = b0.qh + q_offset_h;

            uchar4 val_q1 = *(device const packed_uchar4*)q1;
            uchar4 val_q2 = *(device const packed_uchar4*)q2;
            uchar4 val_qh = *(device const packed_uchar4*)qh;
            device const int8_t* sc = b0.scales + is;
            float d = float(b0.d);

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short l = 0; l < 4; ++l) {
                uint8_t q1_b = val_q1[l];
                uint8_t q2_b = val_q2[l];
                uint8_t qh_b = val_qh[l];
                sums[0] += vy0[l] * float((int8_t)((q1_b & 0xF) | ((qh_b & kmask1) << 4)) - 32);
                sums[1] += vy1[l] * float((int8_t)((q2_b & 0xF) | ((qh_b & kmask2) << 2)) - 32);
                sums[2] += vy2[l] * float((int8_t)((q1_b >> 4)  | ((qh_b & kmask3) << 0)) - 32);
                sums[3] += vy3[l] * float((int8_t)((q2_b >> 4)  | ((qh_b & kmask4) >> 2)) - 32);
            }
            sumf0 += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
        }

        // Row 1
        if (first_row + 1 < rows) {
            device const block_q6_k& b1 = r_w1[i];
            device const uint8_t* q1 = b1.ql + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = b1.qh + q_offset_h;

            uchar4 val_q1 = *(device const packed_uchar4*)q1;
            uchar4 val_q2 = *(device const packed_uchar4*)q2;
            uchar4 val_qh = *(device const packed_uchar4*)qh;
            device const int8_t* sc = b1.scales + is;
            float d = float(b1.d);

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short l = 0; l < 4; ++l) {
                uint8_t q1_b = val_q1[l];
                uint8_t q2_b = val_q2[l];
                uint8_t qh_b = val_qh[l];
                sums[0] += vy0[l] * float((int8_t)((q1_b & 0xF) | ((qh_b & kmask1) << 4)) - 32);
                sums[1] += vy1[l] * float((int8_t)((q2_b & 0xF) | ((qh_b & kmask2) << 2)) - 32);
                sums[2] += vy2[l] * float((int8_t)((q1_b >> 4)  | ((qh_b & kmask3) << 0)) - 32);
                sums[3] += vy3[l] * float((int8_t)((q2_b >> 4)  | ((qh_b & kmask4) >> 2)) - 32);
            }
            sumf1 += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
        }

        // Row 2
        if (first_row + 2 < rows) {
            device const block_q6_k& b2 = r_w2[i];
            device const uint8_t* q1 = b2.ql + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = b2.qh + q_offset_h;

            uchar4 val_q1 = *(device const packed_uchar4*)q1;
            uchar4 val_q2 = *(device const packed_uchar4*)q2;
            uchar4 val_qh = *(device const packed_uchar4*)qh;
            device const int8_t* sc = b2.scales + is;
            float d = float(b2.d);

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short l = 0; l < 4; ++l) {
                uint8_t q1_b = val_q1[l];
                uint8_t q2_b = val_q2[l];
                uint8_t qh_b = val_qh[l];
                sums[0] += vy0[l] * float((int8_t)((q1_b & 0xF) | ((qh_b & kmask1) << 4)) - 32);
                sums[1] += vy1[l] * float((int8_t)((q2_b & 0xF) | ((qh_b & kmask2) << 2)) - 32);
                sums[2] += vy2[l] * float((int8_t)((q1_b >> 4)  | ((qh_b & kmask3) << 0)) - 32);
                sums[3] += vy3[l] * float((int8_t)((q2_b >> 4)  | ((qh_b & kmask4) >> 2)) - 32);
            }
            sumf2 += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
        }

        // Row 3
        if (first_row + 3 < rows) {
            device const block_q6_k& b3 = r_w3[i];
            device const uint8_t* q1 = b3.ql + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = b3.qh + q_offset_h;

            uchar4 val_q1 = *(device const packed_uchar4*)q1;
            uchar4 val_q2 = *(device const packed_uchar4*)q2;
            uchar4 val_qh = *(device const packed_uchar4*)qh;
            device const int8_t* sc = b3.scales + is;
            float d = float(b3.d);

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short l = 0; l < 4; ++l) {
                uint8_t q1_b = val_q1[l];
                uint8_t q2_b = val_q2[l];
                uint8_t qh_b = val_qh[l];
                sums[0] += vy0[l] * float((int8_t)((q1_b & 0xF) | ((qh_b & kmask1) << 4)) - 32);
                sums[1] += vy1[l] * float((int8_t)((q2_b & 0xF) | ((qh_b & kmask2) << 2)) - 32);
                sums[2] += vy2[l] * float((int8_t)((q1_b >> 4)  | ((qh_b & kmask3) << 0)) - 32);
                sums[3] += vy3[l] * float((int8_t)((q2_b >> 4)  | ((qh_b & kmask4) >> 2)) - 32);
            }
            sumf3 += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
        }
    }

    float total0 = simd_sum(sumf0);
    float total1 = simd_sum(sumf1);
    float total2 = simd_sum(sumf2);
    float total3 = simd_sum(sumf3);

    if (tiisg == 0) {
        y[first_row + 0] = total0;
        if (first_row + 1 < rows) y[first_row + 1] = total1;
        if (first_row + 2 < rows) y[first_row + 2] = total2;
        if (first_row + 3 < rows) y[first_row + 3] = total3;
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

// 128-thread TG (4 SG x 4 rows = 16 rows per TG) with 32-bit word loads, float4 loads, factored math
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
    uint batch_idx0 = threadgroup_pos.x * 4;
    uint sg_id = tid / 32;
    uint lane_id = tid % 32;

    uint first_row = threadgroup_pos.y * 16 + sg_id * 4;
    if (batch_idx0 >= batch_size || first_row >= rows) return;

    device const float* cur_x0 = x + (batch_idx0 + 0) * cols;
    device const float* cur_x1 = (batch_idx0 + 1 < batch_size) ? (x + (batch_idx0 + 1) * cols) : cur_x0;
    device const float* cur_x2 = (batch_idx0 + 2 < batch_size) ? (x + (batch_idx0 + 2) * cols) : cur_x0;
    device const float* cur_x3 = (batch_idx0 + 3 < batch_size) ? (x + (batch_idx0 + 3) * cols) : cur_x0;
    const uint nb = cols / 256;

    float sumf0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sumf1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sumf2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sumf3[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint c = lane_id / 8;
    uint l_sub = (lane_id % 8) * 4;
    int is = c * 2;
    int q_off = c * 32 + l_sub;
    int x_chunk_off = c * 64 + l_sub;

    device const block_q4_k* r_w0 = w + (first_row + 0) * nb;
    device const block_q4_k* r_w1 = (first_row + 1 < rows) ? (w + (first_row + 1) * nb) : r_w0;
    device const block_q4_k* r_w2 = (first_row + 2 < rows) ? (w + (first_row + 2) * nb) : r_w0;
    device const block_q4_k* r_w3 = (first_row + 3 < rows) ? (w + (first_row + 3) * nb) : r_w0;

    for (uint b = 0; b < nb; b++) {
        uint x_off = b * 256 + x_chunk_off;
        float4 x0_0 = *(device const float4*)(cur_x0 + x_off);
        float4 x1_0 = *(device const float4*)(cur_x0 + x_off + 32);
        float4 x0_1 = *(device const float4*)(cur_x1 + x_off);
        float4 x1_1 = *(device const float4*)(cur_x1 + x_off + 32);
        float4 x0_2 = *(device const float4*)(cur_x2 + x_off);
        float4 x1_2 = *(device const float4*)(cur_x2 + x_off + 32);
        float4 x0_3 = *(device const float4*)(cur_x3 + x_off);
        float4 x1_3 = *(device const float4*)(cur_x3 + x_off + 32);

        float sum_x0[4] = {
            x0_0[0] + x0_0[1] + x0_0[2] + x0_0[3],
            x0_1[0] + x0_1[1] + x0_1[2] + x0_1[3],
            x0_2[0] + x0_2[1] + x0_2[2] + x0_2[3],
            x0_3[0] + x0_3[1] + x0_3[2] + x0_3[3]
        };
        float sum_x1[4] = {
            x1_0[0] + x1_0[1] + x1_0[2] + x1_0[3],
            x1_1[0] + x1_1[1] + x1_1[2] + x1_1[3],
            x1_2[0] + x1_2[1] + x1_2[2] + x1_2[3],
            x1_3[0] + x1_3[1] + x1_3[2] + x1_3[3]
        };

        // Row 0
        {
            device const block_q4_k& blk0 = r_w0[b];
            float sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk0.scales, sc1, m1, float(blk0.d), float(blk0.dmin));
            get_scale_min_k4(is + 1, blk0.scales, sc2, m2, float(blk0.d), float(blk0.dmin));
            uint32_t q0 = *(device const uint32_t*)(blk0.qs + q_off);
            float q_lo[4] = { float((q0 >>  0) & 0x0F), float((q0 >>  8) & 0x0F), float((q0 >> 16) & 0x0F), float((q0 >> 24) & 0x0F) };
            float q_hi[4] = { float((q0 >>  4) & 0x0F), float((q0 >> 12) & 0x0F), float((q0 >> 20) & 0x0F), float((q0 >> 28) & 0x0F) };

            sumf0[0] += ((q_lo[0]*x0_0[0] + q_lo[1]*x0_0[1] + q_lo[2]*x0_0[2] + q_lo[3]*x0_0[3]) * sc1 - sum_x0[0] * m1) + ((q_hi[0]*x1_0[0] + q_hi[1]*x1_0[1] + q_hi[2]*x1_0[2] + q_hi[3]*x1_0[3]) * sc2 - sum_x1[0] * m2);
            sumf0[1] += ((q_lo[0]*x0_1[0] + q_lo[1]*x0_1[1] + q_lo[2]*x0_1[2] + q_lo[3]*x0_1[3]) * sc1 - sum_x0[1] * m1) + ((q_hi[0]*x1_1[0] + q_hi[1]*x1_1[1] + q_hi[2]*x1_1[2] + q_hi[3]*x1_1[3]) * sc2 - sum_x1[1] * m2);
            sumf0[2] += ((q_lo[0]*x0_2[0] + q_lo[1]*x0_2[1] + q_lo[2]*x0_2[2] + q_lo[3]*x0_2[3]) * sc1 - sum_x0[2] * m1) + ((q_hi[0]*x1_2[0] + q_hi[1]*x1_2[1] + q_hi[2]*x1_2[2] + q_hi[3]*x1_2[3]) * sc2 - sum_x1[2] * m2);
            sumf0[3] += ((q_lo[0]*x0_3[0] + q_lo[1]*x0_3[1] + q_lo[2]*x0_3[2] + q_lo[3]*x0_3[3]) * sc1 - sum_x0[3] * m1) + ((q_hi[0]*x1_3[0] + q_hi[1]*x1_3[1] + q_hi[2]*x1_3[2] + q_hi[3]*x1_3[3]) * sc2 - sum_x1[3] * m2);
        }

        // Row 1
        if (first_row + 1 < rows) {
            device const block_q4_k& blk1 = r_w1[b];
            float sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk1.scales, sc1, m1, float(blk1.d), float(blk1.dmin));
            get_scale_min_k4(is + 1, blk1.scales, sc2, m2, float(blk1.d), float(blk1.dmin));
            uint32_t q1 = *(device const uint32_t*)(blk1.qs + q_off);
            float q_lo[4] = { float((q1 >>  0) & 0x0F), float((q1 >>  8) & 0x0F), float((q1 >> 16) & 0x0F), float((q1 >> 24) & 0x0F) };
            float q_hi[4] = { float((q1 >>  4) & 0x0F), float((q1 >> 12) & 0x0F), float((q1 >> 20) & 0x0F), float((q1 >> 28) & 0x0F) };

            sumf1[0] += ((q_lo[0]*x0_0[0] + q_lo[1]*x0_0[1] + q_lo[2]*x0_0[2] + q_lo[3]*x0_0[3]) * sc1 - sum_x0[0] * m1) + ((q_hi[0]*x1_0[0] + q_hi[1]*x1_0[1] + q_hi[2]*x1_0[2] + q_hi[3]*x1_0[3]) * sc2 - sum_x1[0] * m2);
            sumf1[1] += ((q_lo[0]*x0_1[0] + q_lo[1]*x0_1[1] + q_lo[2]*x0_1[2] + q_lo[3]*x0_1[3]) * sc1 - sum_x0[1] * m1) + ((q_hi[0]*x1_1[0] + q_hi[1]*x1_1[1] + q_hi[2]*x1_1[2] + q_hi[3]*x1_1[3]) * sc2 - sum_x1[1] * m2);
            sumf1[2] += ((q_lo[0]*x0_2[0] + q_lo[1]*x0_2[1] + q_lo[2]*x0_2[2] + q_lo[3]*x0_2[3]) * sc1 - sum_x0[2] * m1) + ((q_hi[0]*x1_2[0] + q_hi[1]*x1_2[1] + q_hi[2]*x1_2[2] + q_hi[3]*x1_2[3]) * sc2 - sum_x1[2] * m2);
            sumf1[3] += ((q_lo[0]*x0_3[0] + q_lo[1]*x0_3[1] + q_lo[2]*x0_3[2] + q_lo[3]*x0_3[3]) * sc1 - sum_x0[3] * m1) + ((q_hi[0]*x1_3[0] + q_hi[1]*x1_3[1] + q_hi[2]*x1_3[2] + q_hi[3]*x1_3[3]) * sc2 - sum_x1[3] * m2);
        }

        // Row 2
        if (first_row + 2 < rows) {
            device const block_q4_k& blk2 = r_w2[b];
            float sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk2.scales, sc1, m1, float(blk2.d), float(blk2.dmin));
            get_scale_min_k4(is + 1, blk2.scales, sc2, m2, float(blk2.d), float(blk2.dmin));
            uint32_t q2 = *(device const uint32_t*)(blk2.qs + q_off);
            float q_lo[4] = { float((q2 >>  0) & 0x0F), float((q2 >>  8) & 0x0F), float((q2 >> 16) & 0x0F), float((q2 >> 24) & 0x0F) };
            float q_hi[4] = { float((q2 >>  4) & 0x0F), float((q2 >> 12) & 0x0F), float((q2 >> 20) & 0x0F), float((q2 >> 28) & 0x0F) };

            sumf2[0] += ((q_lo[0]*x0_0[0] + q_lo[1]*x0_0[1] + q_lo[2]*x0_0[2] + q_lo[3]*x0_0[3]) * sc1 - sum_x0[0] * m1) + ((q_hi[0]*x1_0[0] + q_hi[1]*x1_0[1] + q_hi[2]*x1_0[2] + q_hi[3]*x1_0[3]) * sc2 - sum_x1[0] * m2);
            sumf2[1] += ((q_lo[0]*x0_1[0] + q_lo[1]*x0_1[1] + q_lo[2]*x0_1[2] + q_lo[3]*x0_1[3]) * sc1 - sum_x0[1] * m1) + ((q_hi[0]*x1_1[0] + q_hi[1]*x1_1[1] + q_hi[2]*x1_1[2] + q_hi[3]*x1_1[3]) * sc2 - sum_x1[1] * m2);
            sumf2[2] += ((q_lo[0]*x0_2[0] + q_lo[1]*x0_2[1] + q_lo[2]*x0_2[2] + q_lo[3]*x0_2[3]) * sc1 - sum_x0[2] * m1) + ((q_hi[0]*x1_2[0] + q_hi[1]*x1_2[1] + q_hi[2]*x1_2[2] + q_hi[3]*x1_2[3]) * sc2 - sum_x1[2] * m2);
            sumf2[3] += ((q_lo[0]*x0_3[0] + q_lo[1]*x0_3[1] + q_lo[2]*x0_3[2] + q_lo[3]*x0_3[3]) * sc1 - sum_x0[3] * m1) + ((q_hi[0]*x1_3[0] + q_hi[1]*x1_3[1] + q_hi[2]*x1_3[2] + q_hi[3]*x1_3[3]) * sc2 - sum_x1[3] * m2);
        }

        // Row 3
        if (first_row + 3 < rows) {
            device const block_q4_k& blk3 = r_w3[b];
            float sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk3.scales, sc1, m1, float(blk3.d), float(blk3.dmin));
            get_scale_min_k4(is + 1, blk3.scales, sc2, m2, float(blk3.d), float(blk3.dmin));
            uint32_t q3 = *(device const uint32_t*)(blk3.qs + q_off);
            float q_lo[4] = { float((q3 >>  0) & 0x0F), float((q3 >>  8) & 0x0F), float((q3 >> 16) & 0x0F), float((q3 >> 24) & 0x0F) };
            float q_hi[4] = { float((q3 >>  4) & 0x0F), float((q3 >> 12) & 0x0F), float((q3 >> 20) & 0x0F), float((q3 >> 28) & 0x0F) };

            sumf3[0] += ((q_lo[0]*x0_0[0] + q_lo[1]*x0_0[1] + q_lo[2]*x0_0[2] + q_lo[3]*x0_0[3]) * sc1 - sum_x0[0] * m1) + ((q_hi[0]*x1_0[0] + q_hi[1]*x1_0[1] + q_hi[2]*x1_0[2] + q_hi[3]*x1_0[3]) * sc2 - sum_x1[0] * m2);
            sumf3[1] += ((q_lo[0]*x0_1[0] + q_lo[1]*x0_1[1] + q_lo[2]*x0_1[2] + q_lo[3]*x0_1[3]) * sc1 - sum_x0[1] * m1) + ((q_hi[0]*x1_1[0] + q_hi[1]*x1_1[1] + q_hi[2]*x1_1[2] + q_hi[3]*x1_1[3]) * sc2 - sum_x1[1] * m2);
            sumf3[2] += ((q_lo[0]*x0_2[0] + q_lo[1]*x0_2[1] + q_lo[2]*x0_2[2] + q_lo[3]*x0_2[3]) * sc1 - sum_x0[2] * m1) + ((q_hi[0]*x1_2[0] + q_hi[1]*x1_2[1] + q_hi[2]*x1_2[2] + q_hi[3]*x1_2[3]) * sc2 - sum_x1[2] * m2);
            sumf3[3] += ((q_lo[0]*x0_3[0] + q_lo[1]*x0_3[1] + q_lo[2]*x0_3[2] + q_lo[3]*x0_3[3]) * sc1 - sum_x0[3] * m1) + ((q_hi[0]*x1_3[0] + q_hi[1]*x1_3[1] + q_hi[2]*x1_3[2] + q_hi[3]*x1_3[3]) * sc2 - sum_x1[3] * m2);
        }
    }

    #pragma unroll
    for (int t = 0; t < 4; t++) {
        sumf0[t] = simd_sum(sumf0[t]);
        sumf1[t] = simd_sum(sumf1[t]);
        sumf2[t] = simd_sum(sumf2[t]);
        sumf3[t] = simd_sum(sumf3[t]);
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            if (batch_idx0 + t < batch_size) {
                y[(batch_idx0 + t) * rows + first_row + 0] = sumf0[t];
                if (first_row + 1 < rows) y[(batch_idx0 + t) * rows + first_row + 1] = sumf1[t];
                if (first_row + 2 < rows) y[(batch_idx0 + t) * rows + first_row + 2] = sumf2[t];
                if (first_row + 3 < rows) y[(batch_idx0 + t) * rows + first_row + 3] = sumf3[t];
            }
        }
    }
}

// 128-thread TG (4 SG x 2 rows = 8 rows per TG) with float4 loads
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
    uint batch_idx0 = threadgroup_pos.x * 4;
    uint sg_id = tid / 32;
    uint lane_id = tid % 32;

    uint first_row = threadgroup_pos.y * 8 + sg_id * 2;
    if (batch_idx0 >= batch_size || first_row >= rows) return;

    device const float* cur_x0 = x + (batch_idx0 + 0) * cols;
    device const float* cur_x1 = (batch_idx0 + 1 < batch_size) ? (x + (batch_idx0 + 1) * cols) : cur_x0;
    device const float* cur_x2 = (batch_idx0 + 2 < batch_size) ? (x + (batch_idx0 + 2) * cols) : cur_x0;
    device const float* cur_x3 = (batch_idx0 + 3 < batch_size) ? (x + (batch_idx0 + 3) * cols) : cur_x0;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const uint nb = cols / 256;
    float sumf0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sumf1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const short s_tid = lane_id / 2;
    const short ix    = lane_id % 2;
    const short ip    = s_tid / 8;
    const short il    = s_tid % 8;
    const short l0    = 4 * il;
    const short is    = 8 * ip + l0 / 16;
    const short y_offset   = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    device const block_q6_k* r_w0 = w + (first_row + 0) * nb;
    device const block_q6_k* r_w1 = (first_row + 1 < rows) ? (w + (first_row + 1) * nb) : r_w0;

    for (uint i = ix; i < nb; i += 2) {
        uint off = i * 256 + y_offset;
        device const float* y_vec0 = cur_x0 + off;
        device const float* y_vec1 = cur_x1 + off;
        device const float* y_vec2 = cur_x2 + off;
        device const float* y_vec3 = cur_x3 + off;

        float4 y0_0 = *(device const float4*)(y_vec0 + 0);
        float4 y0_1 = *(device const float4*)(y_vec0 + 32);
        float4 y0_2 = *(device const float4*)(y_vec0 + 64);
        float4 y0_3 = *(device const float4*)(y_vec0 + 96);

        float4 y1_0 = *(device const float4*)(y_vec1 + 0);
        float4 y1_1 = *(device const float4*)(y_vec1 + 32);
        float4 y1_2 = *(device const float4*)(y_vec1 + 64);
        float4 y1_3 = *(device const float4*)(y_vec1 + 96);

        float4 y2_0 = *(device const float4*)(y_vec2 + 0);
        float4 y2_1 = *(device const float4*)(y_vec2 + 32);
        float4 y2_2 = *(device const float4*)(y_vec2 + 64);
        float4 y2_3 = *(device const float4*)(y_vec2 + 96);

        float4 y3_0 = *(device const float4*)(y_vec3 + 0);
        float4 y3_1 = *(device const float4*)(y_vec3 + 32);
        float4 y3_2 = *(device const float4*)(y_vec3 + 64);
        float4 y3_3 = *(device const float4*)(y_vec3 + 96);

        // Row 0
        {
            device const block_q6_k& b0 = r_w0[i];
            device const uint8_t* q1 = b0.ql + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = b0.qh + q_offset_h;
            device const int8_t*  sc = b0.scales + is;
            float d = float(b0.d);

            float4 sums0 = {0.f, 0.f, 0.f, 0.f};
            float4 sums1 = {0.f, 0.f, 0.f, 0.f};
            float4 sums2 = {0.f, 0.f, 0.f, 0.f};
            float4 sums3 = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (short l = 0; l < 4; ++l) {
                float w0 = float((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                float w1 = float((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                float w2 = float((int8_t)((q1[l] >> 4)  | ((qh[l] & kmask3) << 0)) - 32);
                float w3 = float((int8_t)((q2[l] >> 4)  | ((qh[l] & kmask4) >> 2)) - 32);

                sums0[0] += y0_0[l] * w0; sums0[1] += y0_1[l] * w1; sums0[2] += y0_2[l] * w2; sums0[3] += y0_3[l] * w3;
                sums1[0] += y1_0[l] * w0; sums1[1] += y1_1[l] * w1; sums1[2] += y1_2[l] * w2; sums1[3] += y1_3[l] * w3;
                sums2[0] += y2_0[l] * w0; sums2[1] += y2_1[l] * w1; sums2[2] += y2_2[l] * w2; sums2[3] += y2_3[l] * w3;
                sums3[0] += y3_0[l] * w0; sums3[1] += y3_1[l] * w1; sums3[2] += y3_2[l] * w2; sums3[3] += y3_3[l] * w3;
            }
            sumf0[0] += d * (sums0[0] * sc[0] + sums0[1] * sc[2] + sums0[2] * sc[4] + sums0[3] * sc[6]);
            sumf0[1] += d * (sums1[0] * sc[0] + sums1[1] * sc[2] + sums1[2] * sc[4] + sums1[3] * sc[6]);
            sumf0[2] += d * (sums2[0] * sc[0] + sums2[1] * sc[2] + sums2[2] * sc[4] + sums2[3] * sc[6]);
            sumf0[3] += d * (sums3[0] * sc[0] + sums3[1] * sc[2] + sums3[2] * sc[4] + sums3[3] * sc[6]);
        }

        // Row 1
        if (first_row + 1 < rows) {
            device const block_q6_k& b1 = r_w1[i];
            device const uint8_t* q1 = b1.ql + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = b1.qh + q_offset_h;
            device const int8_t*  sc = b1.scales + is;
            float d = float(b1.d);

            float4 sums0 = {0.f, 0.f, 0.f, 0.f};
            float4 sums1 = {0.f, 0.f, 0.f, 0.f};
            float4 sums2 = {0.f, 0.f, 0.f, 0.f};
            float4 sums3 = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (short l = 0; l < 4; ++l) {
                float w0 = float((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                float w1 = float((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                float w2 = float((int8_t)((q1[l] >> 4)  | ((qh[l] & kmask3) << 0)) - 32);
                float w3 = float((int8_t)((q2[l] >> 4)  | ((qh[l] & kmask4) >> 2)) - 32);

                sums0[0] += y0_0[l] * w0; sums0[1] += y0_1[l] * w1; sums0[2] += y0_2[l] * w2; sums0[3] += y0_3[l] * w3;
                sums1[0] += y1_0[l] * w0; sums1[1] += y1_1[l] * w1; sums1[2] += y1_2[l] * w2; sums1[3] += y1_3[l] * w3;
                sums2[0] += y2_0[l] * w0; sums2[1] += y2_1[l] * w1; sums2[2] += y2_2[l] * w2; sums2[3] += y2_3[l] * w3;
                sums3[0] += y3_0[l] * w0; sums3[1] += y3_1[l] * w1; sums3[2] += y3_2[l] * w2; sums3[3] += y3_3[l] * w3;
            }
            sumf1[0] += d * (sums0[0] * sc[0] + sums0[1] * sc[2] + sums0[2] * sc[4] + sums0[3] * sc[6]);
            sumf1[1] += d * (sums1[0] * sc[0] + sums1[1] * sc[2] + sums1[2] * sc[4] + sums1[3] * sc[6]);
            sumf1[2] += d * (sums2[0] * sc[0] + sums2[1] * sc[2] + sums2[2] * sc[4] + sums2[3] * sc[6]);
            sumf1[3] += d * (sums3[0] * sc[0] + sums3[1] * sc[2] + sums3[2] * sc[4] + sums3[3] * sc[6]);
        }
    }

    #pragma unroll
    for (int t = 0; t < 4; t++) {
        sumf0[t] = simd_sum(sumf0[t]);
        sumf1[t] = simd_sum(sumf1[t]);
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            if (batch_idx0 + t < batch_size) {
                y[(batch_idx0 + t) * rows + first_row + 0] = sumf0[t];
                if (first_row + 1 < rows) {
                    y[(batch_idx0 + t) * rows + first_row + 1] = sumf1[t];
                }
            }
        }
    }
}

// Fused Gate + Up + SwiGLU 128-thread TG (4 SG x 4 rows = 16 rows per TG)
kernel void gemm_fused_gate_up_swiglu_q4_k_batched(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_k* gate_w  [[buffer(2)]],
    device const block_q4_k* up_w    [[buffer(3)]],
    constant uint& batch_size        [[buffer(4)]],
    constant uint& rows              [[buffer(5)]],
    constant uint& cols              [[buffer(6)]],
    uint3 threadgroup_pos            [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint batch_idx0 = threadgroup_pos.x * 4;
    uint sg_id = tid / 32;
    uint lane_id = tid % 32;

    uint first_row = threadgroup_pos.y * 16 + sg_id * 4;
    if (batch_idx0 >= batch_size || first_row >= rows) return;

    device const float* cur_x0 = x + (batch_idx0 + 0) * cols;
    device const float* cur_x1 = (batch_idx0 + 1 < batch_size) ? (x + (batch_idx0 + 1) * cols) : cur_x0;
    device const float* cur_x2 = (batch_idx0 + 2 < batch_size) ? (x + (batch_idx0 + 2) * cols) : cur_x0;
    device const float* cur_x3 = (batch_idx0 + 3 < batch_size) ? (x + (batch_idx0 + 3) * cols) : cur_x0;
    const uint nb = cols / 256;

    float sum_g0[4] = {0.0f}, sum_u0[4] = {0.0f};
    float sum_g1[4] = {0.0f}, sum_u1[4] = {0.0f};
    float sum_g2[4] = {0.0f}, sum_u2[4] = {0.0f};
    float sum_g3[4] = {0.0f}, sum_u3[4] = {0.0f};

    uint c = lane_id / 8;
    uint l_sub = (lane_id % 8) * 4;
    int is = c * 2;
    int q_off = c * 32 + l_sub;
    int x_chunk_off = c * 64 + l_sub;

    device const block_q4_k* r_g0 = gate_w + (first_row + 0) * nb;
    device const block_q4_k* r_g1 = (first_row + 1 < rows) ? (gate_w + (first_row + 1) * nb) : r_g0;
    device const block_q4_k* r_g2 = (first_row + 2 < rows) ? (gate_w + (first_row + 2) * nb) : r_g0;
    device const block_q4_k* r_g3 = (first_row + 3 < rows) ? (gate_w + (first_row + 3) * nb) : r_g0;

    device const block_q4_k* r_u0 = up_w + (first_row + 0) * nb;
    device const block_q4_k* r_u1 = (first_row + 1 < rows) ? (up_w + (first_row + 1) * nb) : r_u0;
    device const block_q4_k* r_u2 = (first_row + 2 < rows) ? (up_w + (first_row + 2) * nb) : r_u0;
    device const block_q4_k* r_u3 = (first_row + 3 < rows) ? (up_w + (first_row + 3) * nb) : r_u0;

    for (uint b = 0; b < nb; b++) {
        uint x_off = b * 256 + x_chunk_off;
        float4 x0_0 = *(device const float4*)(cur_x0 + x_off);
        float4 x1_0 = *(device const float4*)(cur_x0 + x_off + 32);
        float4 x0_1 = *(device const float4*)(cur_x1 + x_off);
        float4 x1_1 = *(device const float4*)(cur_x1 + x_off + 32);
        float4 x0_2 = *(device const float4*)(cur_x2 + x_off);
        float4 x1_2 = *(device const float4*)(cur_x2 + x_off + 32);
        float4 x0_3 = *(device const float4*)(cur_x3 + x_off);
        float4 x1_3 = *(device const float4*)(cur_x3 + x_off + 32);

        float sum_x0[4] = {
            x0_0[0] + x0_0[1] + x0_0[2] + x0_0[3],
            x0_1[0] + x0_1[1] + x0_1[2] + x0_1[3],
            x0_2[0] + x0_2[1] + x0_2[2] + x0_2[3],
            x0_3[0] + x0_3[1] + x0_3[2] + x0_3[3]
        };
        float sum_x1[4] = {
            x1_0[0] + x1_0[1] + x1_0[2] + x1_0[3],
            x1_1[0] + x1_1[1] + x1_1[2] + x1_1[3],
            x1_2[0] + x1_2[1] + x1_2[2] + x1_2[3],
            x1_3[0] + x1_3[1] + x1_3[2] + x1_3[3]
        };

        // Row 0
        {
            device const block_q4_k& bg0 = r_g0[b];
            float scg1, mg1, scg2, mg2;
            get_scale_min_k4(is + 0, bg0.scales, scg1, mg1, float(bg0.d), float(bg0.dmin));
            get_scale_min_k4(is + 1, bg0.scales, scg2, mg2, float(bg0.d), float(bg0.dmin));
            uint32_t qg0 = *(device const uint32_t*)(bg0.qs + q_off);
            float qg_lo[4] = { float((qg0 >>  0) & 0x0F), float((qg0 >>  8) & 0x0F), float((qg0 >> 16) & 0x0F), float((qg0 >> 24) & 0x0F) };
            float qg_hi[4] = { float((qg0 >>  4) & 0x0F), float((qg0 >> 12) & 0x0F), float((qg0 >> 20) & 0x0F), float((qg0 >> 28) & 0x0F) };

            sum_g0[0] += ((qg_lo[0]*x0_0[0] + qg_lo[1]*x0_0[1] + qg_lo[2]*x0_0[2] + qg_lo[3]*x0_0[3]) * scg1 - sum_x0[0] * mg1) + ((qg_hi[0]*x1_0[0] + qg_hi[1]*x1_0[1] + qg_hi[2]*x1_0[2] + qg_hi[3]*x1_0[3]) * scg2 - sum_x1[0] * mg2);
            sum_g0[1] += ((qg_lo[0]*x0_1[0] + qg_lo[1]*x0_1[1] + qg_lo[2]*x0_1[2] + qg_lo[3]*x0_1[3]) * scg1 - sum_x0[1] * mg1) + ((qg_hi[0]*x1_1[0] + qg_hi[1]*x1_1[1] + qg_hi[2]*x1_1[2] + qg_hi[3]*x1_1[3]) * scg2 - sum_x1[1] * mg2);
            sum_g0[2] += ((qg_lo[0]*x0_2[0] + qg_lo[1]*x0_2[1] + qg_lo[2]*x0_2[2] + qg_lo[3]*x0_2[3]) * scg1 - sum_x0[2] * mg1) + ((qg_hi[0]*x1_2[0] + qg_hi[1]*x1_2[1] + qg_hi[2]*x1_2[2] + qg_hi[3]*x1_2[3]) * scg2 - sum_x1[2] * mg2);
            sum_g0[3] += ((qg_lo[0]*x0_3[0] + qg_lo[1]*x0_3[1] + qg_lo[2]*x0_3[2] + qg_lo[3]*x0_3[3]) * scg1 - sum_x0[3] * mg1) + ((qg_hi[0]*x1_3[0] + qg_hi[1]*x1_3[1] + qg_hi[2]*x1_3[2] + qg_hi[3]*x1_3[3]) * scg2 - sum_x1[3] * mg2);

            device const block_q4_k& bu0 = r_u0[b];
            float scu1, mu1, scu2, mu2;
            get_scale_min_k4(is + 0, bu0.scales, scu1, mu1, float(bu0.d), float(bu0.dmin));
            get_scale_min_k4(is + 1, bu0.scales, scu2, mu2, float(bu0.d), float(bu0.dmin));
            uint32_t qu0 = *(device const uint32_t*)(bu0.qs + q_off);
            float qu_lo[4] = { float((qu0 >>  0) & 0x0F), float((qu0 >>  8) & 0x0F), float((qu0 >> 16) & 0x0F), float((qu0 >> 24) & 0x0F) };
            float qu_hi[4] = { float((qu0 >>  4) & 0x0F), float((qu0 >> 12) & 0x0F), float((qu0 >> 20) & 0x0F), float((qu0 >> 28) & 0x0F) };

            sum_u0[0] += ((qu_lo[0]*x0_0[0] + qu_lo[1]*x0_0[1] + qu_lo[2]*x0_0[2] + qu_lo[3]*x0_0[3]) * scu1 - sum_x0[0] * mu1) + ((qu_hi[0]*x1_0[0] + qu_hi[1]*x1_0[1] + qu_hi[2]*x1_0[2] + qu_hi[3]*x1_0[3]) * scu2 - sum_x1[0] * mu2);
            sum_u0[1] += ((qu_lo[0]*x0_1[0] + qu_lo[1]*x0_1[1] + qu_lo[2]*x0_1[2] + qu_lo[3]*x0_1[3]) * scu1 - sum_x0[1] * mu1) + ((qu_hi[0]*x1_1[0] + qu_hi[1]*x1_1[1] + qu_hi[2]*x1_1[2] + qu_hi[3]*x1_1[3]) * scu2 - sum_x1[1] * mu2);
            sum_u0[2] += ((qu_lo[0]*x0_2[0] + qu_lo[1]*x0_2[1] + qu_lo[2]*x0_2[2] + qu_lo[3]*x0_2[3]) * scu1 - sum_x0[2] * mu1) + ((qu_hi[0]*x1_2[0] + qu_hi[1]*x1_2[1] + qu_hi[2]*x1_2[2] + qu_hi[3]*x1_2[3]) * scu2 - sum_x1[2] * mu2);
            sum_u0[3] += ((qu_lo[0]*x0_3[0] + qu_lo[1]*x0_3[1] + qu_lo[2]*x0_3[2] + qu_lo[3]*x0_3[3]) * scu1 - sum_x0[3] * mu1) + ((qu_hi[0]*x1_3[0] + qu_hi[1]*x1_3[1] + qu_hi[2]*x1_3[2] + qu_hi[3]*x1_3[3]) * scu2 - sum_x1[3] * mu2);
        }

        // Row 1
        if (first_row + 1 < rows) {
            device const block_q4_k& bg1 = r_g1[b];
            float scg1, mg1, scg2, mg2;
            get_scale_min_k4(is + 0, bg1.scales, scg1, mg1, float(bg1.d), float(bg1.dmin));
            get_scale_min_k4(is + 1, bg1.scales, scg2, mg2, float(bg1.d), float(bg1.dmin));
            uint32_t qg1 = *(device const uint32_t*)(bg1.qs + q_off);
            float qg_lo[4] = { float((qg1 >>  0) & 0x0F), float((qg1 >>  8) & 0x0F), float((qg1 >> 16) & 0x0F), float((qg1 >> 24) & 0x0F) };
            float qg_hi[4] = { float((qg1 >>  4) & 0x0F), float((qg1 >> 12) & 0x0F), float((qg1 >> 20) & 0x0F), float((qg1 >> 28) & 0x0F) };

            sum_g1[0] += ((qg_lo[0]*x0_0[0] + qg_lo[1]*x0_0[1] + qg_lo[2]*x0_0[2] + qg_lo[3]*x0_0[3]) * scg1 - sum_x0[0] * mg1) + ((qg_hi[0]*x1_0[0] + qg_hi[1]*x1_0[1] + qg_hi[2]*x1_0[2] + qg_hi[3]*x1_0[3]) * scg2 - sum_x1[0] * mg2);
            sum_g1[1] += ((qg_lo[0]*x0_1[0] + qg_lo[1]*x0_1[1] + qg_lo[2]*x0_1[2] + qg_lo[3]*x0_1[3]) * scg1 - sum_x0[1] * mg1) + ((qg_hi[0]*x1_1[0] + qg_hi[1]*x1_1[1] + qg_hi[2]*x1_1[2] + qg_hi[3]*x1_1[3]) * scg2 - sum_x1[1] * mg2);
            sum_g1[2] += ((qg_lo[0]*x0_2[0] + qg_lo[1]*x0_2[1] + qg_lo[2]*x0_2[2] + qg_lo[3]*x0_2[3]) * scg1 - sum_x0[2] * mg1) + ((qg_hi[0]*x1_2[0] + qg_hi[1]*x1_2[1] + qg_hi[2]*x1_2[2] + qg_hi[3]*x1_2[3]) * scg2 - sum_x1[2] * mg2);
            sum_g1[3] += ((qg_lo[0]*x0_3[0] + qg_lo[1]*x0_3[1] + qg_lo[2]*x0_3[2] + qg_lo[3]*x0_3[3]) * scg1 - sum_x0[3] * mg1) + ((qg_hi[0]*x1_3[0] + qg_hi[1]*x1_3[1] + qg_hi[2]*x1_3[2] + qg_hi[3]*x1_3[3]) * scg2 - sum_x1[3] * mg2);

            device const block_q4_k& bu1 = r_u1[b];
            float scu1, mu1, scu2, mu2;
            get_scale_min_k4(is + 0, bu1.scales, scu1, mu1, float(bu1.d), float(bu1.dmin));
            get_scale_min_k4(is + 1, bu1.scales, scu2, mu2, float(bu1.d), float(bu1.dmin));
            uint32_t qu1 = *(device const uint32_t*)(bu1.qs + q_off);
            float qu_lo[4] = { float((qu1 >>  0) & 0x0F), float((qu1 >>  8) & 0x0F), float((qu1 >> 16) & 0x0F), float((qu1 >> 24) & 0x0F) };
            float qu_hi[4] = { float((qu1 >>  4) & 0x0F), float((qu1 >> 12) & 0x0F), float((qu1 >> 20) & 0x0F), float((qu1 >> 28) & 0x0F) };

            sum_u1[0] += ((qu_lo[0]*x0_0[0] + qu_lo[1]*x0_0[1] + qu_lo[2]*x0_0[2] + qu_lo[3]*x0_0[3]) * scu1 - sum_x0[0] * mu1) + ((qu_hi[0]*x1_0[0] + qu_hi[1]*x1_0[1] + qu_hi[2]*x1_0[2] + qu_hi[3]*x1_0[3]) * scu2 - sum_x1[0] * mu2);
            sum_u1[1] += ((qu_lo[0]*x0_1[0] + qu_lo[1]*x0_1[1] + qu_lo[2]*x0_1[2] + qu_lo[3]*x0_1[3]) * scu1 - sum_x0[1] * mu1) + ((qu_hi[0]*x1_1[0] + qu_hi[1]*x1_1[1] + qu_hi[2]*x1_1[2] + qu_hi[3]*x1_1[3]) * scu2 - sum_x1[1] * mu2);
            sum_u1[2] += ((qu_lo[0]*x0_2[0] + qu_lo[1]*x0_2[1] + qu_lo[2]*x0_2[2] + qu_lo[3]*x0_2[3]) * scu1 - sum_x0[2] * mu1) + ((qu_hi[0]*x1_2[0] + qu_hi[1]*x1_2[1] + qu_hi[2]*x1_2[2] + qu_hi[3]*x1_2[3]) * scu2 - sum_x1[2] * mu2);
            sum_u1[3] += ((qu_lo[0]*x0_3[0] + qu_lo[1]*x0_3[1] + qu_lo[2]*x0_3[2] + qu_lo[3]*x0_3[3]) * scu1 - sum_x0[3] * mu1) + ((qu_hi[0]*x1_3[0] + qu_hi[1]*x1_3[1] + qu_hi[2]*x1_3[2] + qu_hi[3]*x1_3[3]) * scu2 - sum_x1[3] * mu2);
        }

        // Row 2
        if (first_row + 2 < rows) {
            device const block_q4_k& bg2 = r_g2[b];
            float scg1, mg1, scg2, mg2;
            get_scale_min_k4(is + 0, bg2.scales, scg1, mg1, float(bg2.d), float(bg2.dmin));
            get_scale_min_k4(is + 1, bg2.scales, scg2, mg2, float(bg2.d), float(bg2.dmin));
            uint32_t qg2 = *(device const uint32_t*)(bg2.qs + q_off);
            float qg_lo[4] = { float((qg2 >>  0) & 0x0F), float((qg2 >>  8) & 0x0F), float((qg2 >> 16) & 0x0F), float((qg2 >> 24) & 0x0F) };
            float qg_hi[4] = { float((qg2 >>  4) & 0x0F), float((qg2 >> 12) & 0x0F), float((qg2 >> 20) & 0x0F), float((qg2 >> 28) & 0x0F) };

            sum_g2[0] += ((qg_lo[0]*x0_0[0] + qg_lo[1]*x0_0[1] + qg_lo[2]*x0_0[2] + qg_lo[3]*x0_0[3]) * scg1 - sum_x0[0] * mg1) + ((qg_hi[0]*x1_0[0] + qg_hi[1]*x1_0[1] + qg_hi[2]*x1_0[2] + qg_hi[3]*x1_0[3]) * scg2 - sum_x1[0] * mg2);
            sum_g2[1] += ((qg_lo[0]*x0_1[0] + qg_lo[1]*x0_1[1] + qg_lo[2]*x0_1[2] + qg_lo[3]*x0_1[3]) * scg1 - sum_x0[1] * mg1) + ((qg_hi[0]*x1_1[0] + qg_hi[1]*x1_1[1] + qg_hi[2]*x1_1[2] + qg_hi[3]*x1_1[3]) * scg2 - sum_x1[1] * mg2);
            sum_g2[2] += ((qg_lo[0]*x0_2[0] + qg_lo[1]*x0_2[1] + qg_lo[2]*x0_2[2] + qg_lo[3]*x0_2[3]) * scg1 - sum_x0[2] * mg1) + ((qg_hi[0]*x1_2[0] + qg_hi[1]*x1_2[1] + qg_hi[2]*x1_2[2] + qg_hi[3]*x1_2[3]) * scg2 - sum_x1[2] * mg2);
            sum_g2[3] += ((qg_lo[0]*x0_3[0] + qg_lo[1]*x0_3[1] + qg_lo[2]*x0_3[2] + qg_lo[3]*x0_3[3]) * scg1 - sum_x0[3] * mg1) + ((qg_hi[0]*x1_3[0] + qg_hi[1]*x1_3[1] + qg_hi[2]*x1_3[2] + qg_hi[3]*x1_3[3]) * scg2 - sum_x1[3] * mg2);

            device const block_q4_k& bu2 = r_u2[b];
            float scu1, mu1, scu2, mu2;
            get_scale_min_k4(is + 0, bu2.scales, scu1, mu1, float(bu2.d), float(bu2.dmin));
            get_scale_min_k4(is + 1, bu2.scales, scu2, mu2, float(bu2.d), float(bu2.dmin));
            uint32_t qu2 = *(device const uint32_t*)(bu2.qs + q_off);
            float qu_lo[4] = { float((qu2 >>  0) & 0x0F), float((qu2 >>  8) & 0x0F), float((qu2 >> 16) & 0x0F), float((qu2 >> 24) & 0x0F) };
            float qu_hi[4] = { float((qu2 >>  4) & 0x0F), float((qu2 >> 12) & 0x0F), float((qu2 >> 20) & 0x0F), float((qu2 >> 28) & 0x0F) };

            sum_u2[0] += ((qu_lo[0]*x0_0[0] + qu_lo[1]*x0_0[1] + qu_lo[2]*x0_0[2] + qu_lo[3]*x0_0[3]) * scu1 - sum_x0[0] * mu1) + ((qu_hi[0]*x1_0[0] + qu_hi[1]*x1_0[1] + qu_hi[2]*x1_0[2] + qu_hi[3]*x1_0[3]) * scu2 - sum_x1[0] * mu2);
            sum_u2[1] += ((qu_lo[0]*x0_1[0] + qu_lo[1]*x0_1[1] + qu_lo[2]*x0_1[2] + qu_lo[3]*x0_1[3]) * scu1 - sum_x0[1] * mu1) + ((qu_hi[0]*x1_1[0] + qu_hi[1]*x1_1[1] + qu_hi[2]*x1_1[2] + qu_hi[3]*x1_1[3]) * scu2 - sum_x1[1] * mu2);
            sum_u2[2] += ((qu_lo[0]*x0_2[0] + qu_lo[1]*x0_2[1] + qu_lo[2]*x0_2[2] + qu_lo[3]*x0_2[3]) * scu1 - sum_x0[2] * mu1) + ((qu_hi[0]*x1_2[0] + qu_hi[1]*x1_2[1] + qu_hi[2]*x1_2[2] + qu_hi[3]*x1_2[3]) * scu2 - sum_x1[2] * mu2);
            sum_u2[3] += ((qu_lo[0]*x0_3[0] + qu_lo[1]*x0_3[1] + qu_lo[2]*x0_3[2] + qu_lo[3]*x0_3[3]) * scu1 - sum_x0[3] * mu1) + ((qu_hi[0]*x1_3[0] + qu_hi[1]*x1_3[1] + qu_hi[2]*x1_3[2] + qu_hi[3]*x1_3[3]) * scu2 - sum_x1[3] * mu2);
        }

        // Row 3
        if (first_row + 3 < rows) {
            device const block_q4_k& bg3 = r_g3[b];
            float scg1, mg1, scg2, mg2;
            get_scale_min_k4(is + 0, bg3.scales, scg1, mg1, float(bg3.d), float(bg3.dmin));
            get_scale_min_k4(is + 1, bg3.scales, scg2, mg2, float(bg3.d), float(bg3.dmin));
            uint32_t qg3 = *(device const uint32_t*)(bg3.qs + q_off);
            float qg_lo[4] = { float((qg3 >>  0) & 0x0F), float((qg3 >>  8) & 0x0F), float((qg3 >> 16) & 0x0F), float((qg3 >> 24) & 0x0F) };
            float qg_hi[4] = { float((qg3 >>  4) & 0x0F), float((qg3 >> 12) & 0x0F), float((qg3 >> 20) & 0x0F), float((qg3 >> 28) & 0x0F) };

            sum_g3[0] += ((qg_lo[0]*x0_0[0] + qg_lo[1]*x0_0[1] + qg_lo[2]*x0_0[2] + qg_lo[3]*x0_0[3]) * scg1 - sum_x0[0] * mg1) + ((qg_hi[0]*x1_0[0] + qg_hi[1]*x1_0[1] + qg_hi[2]*x1_0[2] + qg_hi[3]*x1_0[3]) * scg2 - sum_x1[0] * mg2);
            sum_g3[1] += ((qg_lo[0]*x0_1[0] + qg_lo[1]*x0_1[1] + qg_lo[2]*x0_1[2] + qg_lo[3]*x0_1[3]) * scg1 - sum_x0[1] * mg1) + ((qg_hi[0]*x1_1[0] + qg_hi[1]*x1_1[1] + qg_hi[2]*x1_1[2] + qg_hi[3]*x1_1[3]) * scg2 - sum_x1[1] * mg2);
            sum_g3[2] += ((qg_lo[0]*x0_2[0] + qg_lo[1]*x0_2[1] + qg_lo[2]*x0_2[2] + qg_lo[3]*x0_2[3]) * scg1 - sum_x0[2] * mg1) + ((qg_hi[0]*x1_2[0] + qg_hi[1]*x1_2[1] + qg_hi[2]*x1_2[2] + qg_hi[3]*x1_2[3]) * scg2 - sum_x1[2] * mg2);
            sum_g3[3] += ((qg_lo[0]*x0_3[0] + qg_lo[1]*x0_3[1] + qg_lo[2]*x0_3[2] + qg_lo[3]*x0_3[3]) * scg1 - sum_x0[3] * mg1) + ((qg_hi[0]*x1_3[0] + qg_hi[1]*x1_3[1] + qg_hi[2]*x1_3[2] + qg_hi[3]*x1_3[3]) * scg2 - sum_x1[3] * mg2);

            device const block_q4_k& bu3 = r_u3[b];
            float scu1, mu1, scu2, mu2;
            get_scale_min_k4(is + 0, bu3.scales, scu1, mu1, float(bu3.d), float(bu3.dmin));
            get_scale_min_k4(is + 1, bu3.scales, scu2, mu2, float(bu3.d), float(bu3.dmin));
            uint32_t qu3 = *(device const uint32_t*)(bu3.qs + q_off);
            float qu_lo[4] = { float((qu3 >>  0) & 0x0F), float((qu3 >>  8) & 0x0F), float((qu3 >> 16) & 0x0F), float((qu3 >> 24) & 0x0F) };
            float qu_hi[4] = { float((qu3 >>  4) & 0x0F), float((qu3 >> 12) & 0x0F), float((qu3 >> 20) & 0x0F), float((qu3 >> 28) & 0x0F) };

            sum_u3[0] += ((qu_lo[0]*x0_0[0] + qu_lo[1]*x0_0[1] + qu_lo[2]*x0_0[2] + qu_lo[3]*x0_0[3]) * scu1 - sum_x0[0] * mu1) + ((qu_hi[0]*x1_0[0] + qu_hi[1]*x1_0[1] + qu_hi[2]*x1_0[2] + qu_hi[3]*x1_0[3]) * scu2 - sum_x1[0] * mu2);
            sum_u3[1] += ((qu_lo[0]*x0_1[0] + qu_lo[1]*x0_1[1] + qu_lo[2]*x0_1[2] + qu_lo[3]*x0_1[3]) * scu1 - sum_x0[1] * mu1) + ((qu_hi[0]*x1_1[0] + qu_hi[1]*x1_1[1] + qu_hi[2]*x1_1[2] + qu_hi[3]*x1_1[3]) * scu2 - sum_x1[1] * mu2);
            sum_u3[2] += ((qu_lo[0]*x0_2[0] + qu_lo[1]*x0_2[1] + qu_lo[2]*x0_2[2] + qu_lo[3]*x0_2[3]) * scu1 - sum_x0[2] * mu1) + ((qu_hi[0]*x1_2[0] + qu_hi[1]*x1_2[1] + qu_hi[2]*x1_2[2] + qu_hi[3]*x1_2[3]) * scu2 - sum_x1[2] * mu2);
            sum_u3[3] += ((qu_lo[0]*x0_3[0] + qu_lo[1]*x0_3[1] + qu_lo[2]*x0_3[2] + qu_lo[3]*x0_3[3]) * scu1 - sum_x0[3] * mu1) + ((qu_hi[0]*x1_3[0] + qu_hi[1]*x1_3[1] + qu_hi[2]*x1_3[2] + qu_hi[3]*x1_3[3]) * scu2 - sum_x1[3] * mu2);
        }
    }

    #pragma unroll
    for (int t = 0; t < 4; t++) {
        sum_g0[t] = simd_sum(sum_g0[t]); sum_u0[t] = simd_sum(sum_u0[t]);
        sum_g1[t] = simd_sum(sum_g1[t]); sum_u1[t] = simd_sum(sum_u1[t]);
        sum_g2[t] = simd_sum(sum_g2[t]); sum_u2[t] = simd_sum(sum_u2[t]);
        sum_g3[t] = simd_sum(sum_g3[t]); sum_u3[t] = simd_sum(sum_u3[t]);
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            if (batch_idx0 + t < batch_size) {
                float s0 = (sum_g0[t] / (1.0f + exp(-sum_g0[t]))) * sum_u0[t];
                y[(batch_idx0 + t) * rows + first_row + 0] = s0;
                if (first_row + 1 < rows) {
                    float s1 = (sum_g1[t] / (1.0f + exp(-sum_g1[t]))) * sum_u1[t];
                    y[(batch_idx0 + t) * rows + first_row + 1] = s1;
                }
                if (first_row + 2 < rows) {
                    float s2 = (sum_g2[t] / (1.0f + exp(-sum_g2[t]))) * sum_u2[t];
                    y[(batch_idx0 + t) * rows + first_row + 2] = s2;
                }
                if (first_row + 3 < rows) {
                    float s3 = (sum_g3[t] / (1.0f + exp(-sum_g3[t]))) * sum_u3[t];
                    y[(batch_idx0 + t) * rows + first_row + 3] = s3;
                }
            }
        }
    }
}

// --- Fused Transformer Math Kernels ---

// Fused Residual Add (x += proj) + RMSNorm (out_norm = rmsnorm(x, weight, eps))
kernel void kernel_residual_rmsnorm(
    device float* x                  [[buffer(0)]],
    device const float* proj         [[buffer(1)]],
    device float* out_norm           [[buffer(2)]],
    device const float* weight       [[buffer(3)]],
    constant uint& dim               [[buffer(4)]],
    constant float& eps              [[buffer(5)]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += 32) {
        float v = x[i] + proj[i];
        x[i] = v;
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float scale = 1.0f / sqrt((sum_sq / float(dim)) + eps);

    for (uint i = tid; i < dim; i += 32) {
        out_norm[i] = x[i] * scale * weight[i];
    }
}

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

kernel void kernel_rmsnorm_batch(
    device float* out                [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const float* weight       [[buffer(2)]],
    constant uint& dim               [[buffer(3)]],
    constant float& eps              [[buffer(4)]],
    constant uint& batch_size        [[buffer(5)]],
    uint b                           [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    if (b >= batch_size) return;
    device const float* x_b = x + b * dim;
    device float* out_b = out + b * dim;

    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += 32) {
        float v = x_b[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float scale = 1.0f / sqrt((sum_sq / float(dim)) + eps);

    for (uint i = tid; i < dim; i += 32) {
        out_b[i] = x_b[i] * scale * weight[i];
    }
}

kernel void kernel_residual_rmsnorm_batch(
    device float* x                  [[buffer(0)]],
    device const float* proj         [[buffer(1)]],
    device float* out_norm           [[buffer(2)]],
    device const float* weight       [[buffer(3)]],
    constant uint& dim               [[buffer(4)]],
    constant float& eps              [[buffer(5)]],
    constant uint& batch_size        [[buffer(6)]],
    uint b                           [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    if (b >= batch_size) return;
    device float* x_b = x + b * dim;
    device const float* proj_b = proj + b * dim;
    device float* out_b = out_norm + b * dim;

    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += 32) {
        float v = x_b[i] + proj_b[i];
        x_b[i] = v;
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float scale = 1.0f / sqrt((sum_sq / float(dim)) + eps);

    for (uint i = tid; i < dim; i += 32) {
        out_b[i] = x_b[i] * scale * weight[i];
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
    thread float thread_accum[8] = {0.0f};

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

        uint step = 0;
        for (uint d = tid; d < head_dim; d += 32) {
            thread_accum[step] = thread_accum[step] * scale_prev + exp_val * v_t[d];
            step++;
        }
    }

    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    uint step = 0;
    for (uint d = tid; d < head_dim; d += 32) {
        out_h[d] = thread_accum[step] * inv_sum;
        step++;
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

// --- Fused Gate-Up SwiGLU Kernels (Unified Activation Streaming) ---

kernel void gemv_fused_gate_up_swiglu_q4_0(
    device float* out_gate_swiglu    [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_0* w_gate  [[buffer(2)]],
    device const block_q4_0* w_up    [[buffer(3)]],
    constant uint& hidden_dim        [[buffer(4)]],
    constant uint& dim               [[buffer(5)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= hidden_dim) return;

    threadgroup float tg_gate[4][8];
    threadgroup float tg_up[4][8];

    uint num_blocks = dim / 32;
    device const block_q4_0* gate_blocks[8];
    device const block_q4_0* up_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        gate_blocks[i] = (r < hidden_dim) ? (w_gate + r * num_blocks) : (w_gate + r0 * num_blocks);
        up_blocks[i]   = (r < hidden_dim) ? (w_up + r * num_blocks)   : (w_up + r0 * num_blocks);
    }

    float sums_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sums_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 128) {
        uint x_off = b * 32;
        float x_low[16], x_high[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            x_low[j]  = x[x_off + j];
            x_high[j] = x[x_off + j + 16];
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < hidden_dim) {
                // Gate
                {
                    device const block_q4_0& blk = gate_blocks[i][b];
                    float d = float(blk.d);
                    device const uint8_t* qs = blk.qs;
                    float b_sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint8_t val = qs[j];
                        b_sum += float(int(val & 0x0F) - 8) * x_low[j] + float(int((val >> 4) & 0x0F) - 8) * x_high[j];
                    }
                    sums_g[i] += b_sum * d;
                }
                // Up
                {
                    device const block_q4_0& blk = up_blocks[i][b];
                    float d = float(blk.d);
                    device const uint8_t* qs = blk.qs;
                    float b_sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint8_t val = qs[j];
                        b_sum += float(int(val & 0x0F) - 8) * x_low[j] + float(int((val >> 4) & 0x0F) - 8) * x_high[j];
                    }
                    sums_u[i] += b_sum * d;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums_g[i] = simd_sum(sums_g[i]);
        sums_u[i] = simd_sum(sums_u[i]);
    }

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_gate[simd_id][i] = sums_g[i];
            tg_up[simd_id][i]   = sums_u[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < hidden_dim) {
                float total_g = tg_gate[0][i] + tg_gate[1][i] + tg_gate[2][i] + tg_gate[3][i];
                float total_u = tg_up[0][i]   + tg_up[1][i]   + tg_up[2][i]   + tg_up[3][i];
                float silu = total_g / (1.0f + exp(-total_g));
                out_gate_swiglu[r0 + i] = silu * total_u;
            }
        }
    }
}

kernel void gemv_fused_gate_up_swiglu_q8_0(
    device float* out_gate_swiglu    [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q8_0* w_gate  [[buffer(2)]],
    device const block_q8_0* w_up    [[buffer(3)]],
    constant uint& hidden_dim        [[buffer(4)]],
    constant uint& dim               [[buffer(5)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= hidden_dim) return;

    threadgroup float tg_gate[4][8];
    threadgroup float tg_up[4][8];

    uint num_blocks = dim / 32;
    device const block_q8_0* gate_blocks[8];
    device const block_q8_0* up_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        gate_blocks[i] = (r < hidden_dim) ? (w_gate + r * num_blocks) : (w_gate + r0 * num_blocks);
        up_blocks[i]   = (r < hidden_dim) ? (w_up + r * num_blocks)   : (w_up + r0 * num_blocks);
    }

    float sums_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sums_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 128) {
        uint x_off = b * 32;
        float x_vals[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            x_vals[j] = x[x_off + j];
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < hidden_dim) {
                // Gate
                {
                    device const block_q8_0& blk = gate_blocks[i][b];
                    float d = float(blk.d);
                    device const int8_t* qs = blk.qs;
                    float b_sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        b_sum += float(qs[j]) * x_vals[j];
                    }
                    sums_g[i] += b_sum * d;
                }
                // Up
                {
                    device const block_q8_0& blk = up_blocks[i][b];
                    float d = float(blk.d);
                    device const int8_t* qs = blk.qs;
                    float b_sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        b_sum += float(qs[j]) * x_vals[j];
                    }
                    sums_u[i] += b_sum * d;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums_g[i] = simd_sum(sums_g[i]);
        sums_u[i] = simd_sum(sums_u[i]);
    }

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_gate[simd_id][i] = sums_g[i];
            tg_up[simd_id][i]   = sums_u[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < hidden_dim) {
                float total_g = tg_gate[0][i] + tg_gate[1][i] + tg_gate[2][i] + tg_gate[3][i];
                float total_u = tg_up[0][i]   + tg_up[1][i]   + tg_up[2][i]   + tg_up[3][i];
                float silu = total_g / (1.0f + exp(-total_g));
                out_gate_swiglu[r0 + i] = silu * total_u;
            }
        }
    }
}

// High-efficiency Fused Gate-Up + SwiGLU Q4_K Kernel (2 SIMDgroups, 4 rows/SIMDgroup = 8 rows/TG)
kernel void gemv_fused_gate_up_swiglu_q4_k(
    device float* out_gate_swiglu    [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q4_k* w_gate  [[buffer(2)]],
    device const block_q4_k* w_up    [[buffer(3)]],
    constant uint& hidden_dim        [[buffer(4)]],
    constant uint& dim               [[buffer(5)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    const short NSG = 2;
    const short nr0 = 4;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short sgitg = tid / 32;
    const short tiisg = tid % 32;

    const short ix = tiisg / 8;  // 0...3 (which block)
    const short it = tiisg % 8;  // 0...7 (which 32-element chunk)
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0...3

    const int nb = dim / 256;

    const int first_row = (tg_idx * NSG + sgitg) * nr0;
    if (first_row >= hidden_dim) return;

    device const block_q4_k * g_ptr = w_gate + first_row * nb;
    device const block_q4_k * u_ptr = w_up   + first_row * nb;
    device const float      * y     = x;

    float yl[16];
    float yh[16];

    float sum_g[4] = {0.f, 0.f, 0.f, 0.f};
    float sum_u[4] = {0.f, 0.f, 0.f, 0.f};

    device const float * y4 = y + ix * 256 + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        #pragma unroll
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc_g = (device const uint16_t *)g_ptr[ib].scales + iq;
        device const uint16_t * q1_g = (device const uint16_t *)g_ptr[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh_g = &g_ptr[ib].d;

        device const uint16_t * sc_u = (device const uint16_t *)u_ptr[ib].scales + iq;
        device const uint16_t * q1_u = (device const uint16_t *)u_ptr[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh_u = &u_ptr[ib].d;

        for (short row = 0; row < nr0; row++) {
            if (first_row + row >= hidden_dim) break;

            // --- Gate Row ---
            sc16[0] = sc_g[0] & kmask1;
            sc16[1] = sc_g[2] & kmask1;
            sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
            sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);

            device const uint16_t * q2_g = q1_g + 32;

            float4 acc1_g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2_g = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (short i = 0; i < 4; ++i) {
                acc1_g[0] += yl[2*i + 0] * (q1_g[i] & 0x000F);
                acc1_g[1] += yl[2*i + 1] * (q1_g[i] & 0x0F00);
                acc1_g[2] += yl[2*i + 8] * (q1_g[i] & 0x00F0);
                acc1_g[3] += yl[2*i + 9] * (q1_g[i] & 0xF000);
                acc2_g[0] += yh[2*i + 0] * (q2_g[i] & 0x000F);
                acc2_g[1] += yh[2*i + 1] * (q2_g[i] & 0x0F00);
                acc2_g[2] += yh[2*i + 8] * (q2_g[i] & 0x00F0);
                acc2_g[3] += yh[2*i + 9] * (q2_g[i] & 0xF000);
            }

            sum_g[row] += float(dh_g[0]) * ((acc1_g[0] + (1.f/256.f) * acc1_g[1]) * sc8[0] +
                                            (acc1_g[2] + (1.f/256.f) * acc1_g[3]) * sc8[1] * (1.f/16.f) +
                                            (acc2_g[0] + (1.f/256.f) * acc2_g[1]) * sc8[4] +
                                            (acc2_g[2] + (1.f/256.f) * acc2_g[3]) * sc8[5] * (1.f/16.f)) -
                          float(dh_g[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1_g += (nb * sizeof(block_q4_k)) / 2;
            sc_g += (nb * sizeof(block_q4_k)) / 2;
            dh_g += (nb * sizeof(block_q4_k)) / 2;

            // --- Up Row ---
            sc16[0] = sc_u[0] & kmask1;
            sc16[1] = sc_u[2] & kmask1;
            sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
            sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);

            device const uint16_t * q2_u = q1_u + 32;

            float4 acc1_u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2_u = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (short i = 0; i < 4; ++i) {
                acc1_u[0] += yl[2*i + 0] * (q1_u[i] & 0x000F);
                acc1_u[1] += yl[2*i + 1] * (q1_u[i] & 0x0F00);
                acc1_u[2] += yl[2*i + 8] * (q1_u[i] & 0x00F0);
                acc1_u[3] += yl[2*i + 9] * (q1_u[i] & 0xF000);
                acc2_u[0] += yh[2*i + 0] * (q2_u[i] & 0x000F);
                acc2_u[1] += yh[2*i + 1] * (q2_u[i] & 0x0F00);
                acc2_u[2] += yh[2*i + 8] * (q2_u[i] & 0x00F0);
                acc2_u[3] += yh[2*i + 9] * (q2_u[i] & 0xF000);
            }

            sum_u[row] += float(dh_u[0]) * ((acc1_u[0] + (1.f/256.f) * acc1_u[1]) * sc8[0] +
                                            (acc1_u[2] + (1.f/256.f) * acc1_u[3]) * sc8[1] * (1.f/16.f) +
                                            (acc2_u[0] + (1.f/256.f) * acc2_u[1]) * sc8[4] +
                                            (acc2_u[2] + (1.f/256.f) * acc2_u[3]) * sc8[5] * (1.f/16.f)) -
                          float(dh_u[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1_u += (nb * sizeof(block_q4_k)) / 2;
            sc_u += (nb * sizeof(block_q4_k)) / 2;
            dh_u += (nb * sizeof(block_q4_k)) / 2;
        }

        y4 += 4 * 256;
    }

    for (int row = 0; row < nr0; ++row) {
        if (first_row + row >= hidden_dim) break;
        float g = simd_sum(sum_g[row]);
        float u = simd_sum(sum_u[row]);
        if (tiisg == 0) {
            out_gate_swiglu[first_row + row] = (g / (1.0f + exp(-g))) * u;
        }
    }
}

kernel void gemv_fused_gate_up_swiglu_q6_k(
    device float* out_gate_swiglu    [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q6_k* w_gate  [[buffer(2)]],
    device const block_q6_k* w_up    [[buffer(3)]],
    constant uint& hidden_dim        [[buffer(4)]],
    constant uint& dim               [[buffer(5)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= hidden_dim) return;

    threadgroup float tg_gate[4][8];
    threadgroup float tg_up[4][8];

    uint num_blocks = dim / 256;
    device const block_q6_k* gate_blocks[8];
    device const block_q6_k* up_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        gate_blocks[i] = (r < hidden_dim) ? (w_gate + r * num_blocks) : (w_gate + r0 * num_blocks);
        up_blocks[i]   = (r < hidden_dim) ? (w_up + r * num_blocks)   : (w_up + r0 * num_blocks);
    }

    float sums_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sums_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 128) {
        uint x_off = b * 256;

        for (int n = 0; n < 2; n++) {
            device const float* x_sub = x + x_off + n * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                float x0 = x_sub[l + 0];
                float x1 = x_sub[l + 32];
                float x2 = x_sub[l + 64];
                float x3 = x_sub[l + 96];

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    if (r0 + i < hidden_dim) {
                        // Gate
                        {
                            device const block_q6_k& blk = gate_blocks[i][b];
                            device const uint8_t* ql_sub = blk.ql + n * 64;
                            device const uint8_t* qh_sub = blk.qh + n * 32;
                            device const int8_t*  sc_sub = blk.scales + n * 8;
                            float d = float(blk.d);

                            int q1 = int((ql_sub[l] & 0x0F) | (((qh_sub[l] >> 0) & 3) << 4)) - 32;
                            int q2 = int((ql_sub[l + 32] & 0x0F) | (((qh_sub[l] >> 2) & 3) << 4)) - 32;
                            int q3 = int(((ql_sub[l] >> 4) & 0x0F) | (((qh_sub[l] >> 4) & 3) << 4)) - 32;
                            int q4 = int(((ql_sub[l + 32] >> 4) & 0x0F) | (((qh_sub[l] >> 6) & 3) << 4)) - 32;

                            float s1 = d * float(sc_sub[is + 0]);
                            float s2 = d * float(sc_sub[is + 2]);
                            float s3 = d * float(sc_sub[is + 4]);
                            float s4 = d * float(sc_sub[is + 6]);

                            sums_g[i] += (float(q1) * s1) * x0
                                       + (float(q2) * s2) * x1
                                       + (float(q3) * s3) * x2
                                       + (float(q4) * s4) * x3;
                        }

                        // Up
                        {
                            device const block_q6_k& blk = up_blocks[i][b];
                            device const uint8_t* ql_sub = blk.ql + n * 64;
                            device const uint8_t* qh_sub = blk.qh + n * 32;
                            device const int8_t*  sc_sub = blk.scales + n * 8;
                            float d = float(blk.d);

                            int q1 = int((ql_sub[l] & 0x0F) | (((qh_sub[l] >> 0) & 3) << 4)) - 32;
                            int q2 = int((ql_sub[l + 32] & 0x0F) | (((qh_sub[l] >> 2) & 3) << 4)) - 32;
                            int q3 = int(((ql_sub[l] >> 4) & 0x0F) | (((qh_sub[l] >> 4) & 3) << 4)) - 32;
                            int q4 = int(((ql_sub[l + 32] >> 4) & 0x0F) | (((qh_sub[l] >> 6) & 3) << 4)) - 32;

                            float s1 = d * float(sc_sub[is + 0]);
                            float s2 = d * float(sc_sub[is + 2]);
                            float s3 = d * float(sc_sub[is + 4]);
                            float s4 = d * float(sc_sub[is + 6]);

                            sums_u[i] += (float(q1) * s1) * x0
                                       + (float(q2) * s2) * x1
                                       + (float(q3) * s3) * x2
                                       + (float(q4) * s4) * x3;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums_g[i] = simd_sum(sums_g[i]);
        sums_u[i] = simd_sum(sums_u[i]);
    }

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_gate[simd_id][i] = sums_g[i];
            tg_up[simd_id][i]   = sums_u[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < hidden_dim) {
                float total_g = tg_gate[0][i] + tg_gate[1][i] + tg_gate[2][i] + tg_gate[3][i];
                float total_u = tg_up[0][i]   + tg_up[1][i]   + tg_up[2][i]   + tg_up[3][i];
                float silu = total_g / (1.0f + exp(-total_g));
                out_gate_swiglu[r0 + i] = silu * total_u;
            }
        }
    }
}

// 128-Thread 8-Row SIMD Vectorized Q2_K GEMV
kernel void gemv_q2_k(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q2_k* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    threadgroup float tg_sums[4][8];

    uint blocks_per_row = cols / 256;
    device const block_q2_k* r_ptrs[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r_ptrs[i] = w + (r0 + i) * blocks_per_row;
    }

    float sums[8] = {0.0f};

    for (uint b = tid; b < blocks_per_row; b += 128) {
        uint x_base = b * 256;

        #pragma unroll
        for (int row_i = 0; row_i < 8; row_i++) {
            if (r0 + row_i >= rows) continue;
            device const block_q2_k& blk = r_ptrs[row_i][b];
            float d = (float)blk.d;
            float dmin = (float)blk.dmin;

            float blk_sum = 0.0f;
            for (int sb = 0; sb < 16; sb++) {
                float sc = (float)(blk.scales[sb] & 0x0F) * d;
                float m  = (float)(blk.scales[sb] >> 4) * dmin;
                for (int j = 0; j < 16; j++) {
                    int idx = sb * 16 + j;
                    int byte_idx = idx / 4;
                    int shift = (idx % 4) * 2;
                    float q = (float)((blk.qs[byte_idx] >> shift) & 3);
                    blk_sum += (q * sc - m) * x[x_base + idx];
                }
            }
            sums[row_i] += blk_sum;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_sums[simd_id][i] = sums[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) {
                y[r0 + i] = tg_sums[0][i] + tg_sums[1][i] + tg_sums[2][i] + tg_sums[3][i];
            }
        }
    }
}

// 128-Thread 8-Row SIMD Vectorized Q3_K GEMV
kernel void gemv_q3_k(
    device float* y                  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const block_q3_k* w       [[buffer(2)]],
    constant uint& rows              [[buffer(3)]],
    constant uint& cols              [[buffer(4)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_idx * 8;
    if (r0 >= rows) return;

    threadgroup float tg_sums[4][8];

    uint blocks_per_row = cols / 256;
    device const block_q3_k* r_ptrs[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r_ptrs[i] = w + (r0 + i) * blocks_per_row;
    }

    float sums[8] = {0.0f};

    for (uint b = tid; b < blocks_per_row; b += 128) {
        uint x_base = b * 256;

        #pragma unroll
        for (int row_i = 0; row_i < 8; row_i++) {
            if (r0 + row_i >= rows) continue;
            device const block_q3_k& blk = r_ptrs[row_i][b];
            float d = (float)blk.d;

            float blk_sum = 0.0f;
            for (int sb = 0; sb < 16; sb++) {
                float sc = (float)((int8_t)blk.scales[sb % 12]) * d;
                for (int j = 0; j < 16; j++) {
                    int idx = sb * 16 + j;
                    int byte_idx = idx / 4;
                    int shift = (idx % 4) * 2;
                    int low2 = (blk.qs[byte_idx] >> shift) & 3;

                    int h_byte = idx / 8;
                    int h_shift = idx % 8;
                    int high1 = (blk.hmask[h_byte] >> h_shift) & 1;

                    int q = low2 | (high1 << 2) - 4;
                    blk_sum += ((float)q * sc) * x[x_base + idx];
                }
            }
            sums[row_i] += blk_sum;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sums[i] = simd_sum(sums[i]);
    }

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tg_sums[simd_id][i] = sums[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (r0 + i < rows) {
                y[r0 + i] = tg_sums[0][i] + tg_sums[1][i] + tg_sums[2][i] + tg_sums[3][i];
            }
        }
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

kernel void kernel_sample_argmax(
    device const float* logits       [[buffer(0)]],
    device uint32_t* result_token    [[buffer(1)]],
    constant uint& vocab_size        [[buffer(2)]],
    uint t_idx                       [[thread_index_in_threadgroup]]
) {
    threadgroup float tg_max[1024];
    threadgroup uint  tg_idx_arr[1024];

    float local_max = -INFINITY;
    uint  local_idx = 0;

    for (uint i = t_idx; i < vocab_size; i += 1024) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    tg_max[t_idx] = local_max;
    tg_idx_arr[t_idx] = local_idx;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 512; s > 0; s >>= 1) {
        if (t_idx < s) {
            if (tg_max[t_idx + s] > tg_max[t_idx]) {
                tg_max[t_idx] = tg_max[t_idx + s];
                tg_idx_arr[t_idx] = tg_idx_arr[t_idx + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t_idx == 0) {
        result_token[0] = tg_idx_arr[0];
    }
}

// --- Qwen 3.5 Hybrid SSM & Attention Kernels ---

// 1D Depthwise Causal Convolution step with in-place SiLU activation and history shift
kernel void kernel_conv1d_step(
    device float* out                [[buffer(0)]],
    device const float* in           [[buffer(1)]],
    device float* state              [[buffer(2)]],
    device const float* conv_weight  [[buffer(3)]],
    constant uint& kernel_size       [[buffer(4)]],
    constant uint& channels          [[buffer(5)]],
    uint c                           [[thread_position_in_grid]]
) {
    if (c >= channels) return;
    uint hist_len = kernel_size - 1;
    float sum = 0.0f;
    for (uint k = 0; k < hist_len; k++) {
        sum += conv_weight[c * kernel_size + k] * state[k * channels + c];
    }
    sum += conv_weight[c * kernel_size + hist_len] * in[c];

    // SiLU activation: sum / (1.0f + exp(-sum))
    out[c] = sum / (1.0f + exp(-sum));

    // Shift state history
    for (uint k = 0; k < hist_len - 1; k++) {
        state[k * channels + c] = state[(k + 1) * channels + c];
    }
    if (hist_len > 0) {
        state[(hist_len - 1) * channels + c] = in[c];
    }
}

// Gated DeltaNet Recurrent SSM Step across 48 heads:
// Fused Q & K per-head L2-normalization (with 1/sqrt(d) scaling on Q),
// softplus dt & decay calculation, Delta Rule update with beta gate:
//   kv_mem = S * k
//   delta = (v - kv_mem) * sigmoid(beta)
//   S = decay * S + delta * k^T
//   y = S * q
// followed by head RMSNorm and SiLU output gating y *= silu(gate).
kernel void kernel_ssm_step(
    device float* ssm_out            [[buffer(0)]],
    device const float* conv_out     [[buffer(1)]],
    device const float* ssm_alpha    [[buffer(2)]],
    device const float* ssm_beta     [[buffer(3)]],
    device const float* dt_bias      [[buffer(4)]],
    device const float* ssm_a        [[buffer(5)]],
    device const float* ssm_norm_w   [[buffer(6)]],
    device const float* ssm_gate     [[buffer(7)]],
    device float* ssm_state          [[buffer(8)]],
    constant uint& ssm_inner         [[buffer(9)]],
    constant uint& ssm_state_size    [[buffer(10)]],
    constant uint& ssm_groups        [[buffer(11)]],
    constant uint& ssm_rank          [[buffer(12)]],
    constant float& eps              [[buffer(13)]],
    uint h                           [[threadgroup_position_in_grid]],
    uint i                           [[thread_index_in_threadgroup]]
) {
    if (h >= ssm_rank || i >= ssm_state_size) return;

    // Head parameters: Q and K heads are mapped modulo ssm_groups (matches llama.cpp kernel_gated_delta_net_impl)
    uint g = h % ssm_groups;

    // Base offsets into conv_out:
    // Q: [0 .. ssm_groups * ssm_state_size) (2048)
    // K: [ssm_groups * ssm_state_size .. 2 * ssm_groups * ssm_state_size) (2048)
    // V: [2 * ssm_groups * ssm_state_size .. 2 * ssm_groups * ssm_state_size + ssm_inner) (6144)
    uint q_base = 0;
    uint k_base = ssm_groups * ssm_state_size;
    uint v_base = 2 * ssm_groups * ssm_state_size;

    device const float* q_head = conv_out + q_base + g * ssm_state_size;
    device const float* k_head = conv_out + k_base + g * ssm_state_size;
    device const float* v_head = conv_out + v_base + h * ssm_state_size;

    float my_q = q_head[i];
    float my_k = k_head[i];
    float my_v = v_head[i];

    // Threadgroup storage for Q and K L2-normalization
    threadgroup float tg_k[128];
    threadgroup float tg_q[128];
    threadgroup float tg_q_sq[4];
    threadgroup float tg_k_sq[4];

    uint simd_id = i / 32;
    uint lane_id = i % 32;

    float q_sq_simd = simd_sum(my_q * my_q);
    float k_sq_simd = simd_sum(my_k * my_k);
    if (lane_id == 0) {
        tg_q_sq[simd_id] = q_sq_simd;
        tg_k_sq[simd_id] = k_sq_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_sq_total = tg_q_sq[0] + tg_q_sq[1] + tg_q_sq[2] + tg_q_sq[3];
    float k_sq_total = tg_k_sq[0] + tg_k_sq[1] + tg_k_sq[2] + tg_k_sq[3];

    // Q L2-norm with 1/sqrt(ssm_state_size) scaling; K L2-norm
    float inv_q_norm = rsqrt(q_sq_total + 1e-6f) * (1.0f / sqrt(float(ssm_state_size)));
    float inv_k_norm = rsqrt(k_sq_total + 1e-6f);

    tg_q[i] = my_q * inv_q_norm;
    tg_k[i] = my_k * inv_k_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay calculation: dt = softplus(alpha[h] + dt_bias[h])
    float alpha_val = ssm_alpha[h];
    if (dt_bias != nullptr) {
        alpha_val += dt_bias[h];
    }
    float dt = (alpha_val > 20.0f) ? alpha_val : log(1.0f + exp(alpha_val));
    float a_val = (ssm_a != nullptr) ? ssm_a[h] : 0.0f;
    float decay = exp(dt * a_val);

    // Beta: sigmoid(ssm_beta[h])
    float b_val = (ssm_beta != nullptr) ? ssm_beta[h] : 0.0f;
    float beta_val = 1.0f / (1.0f + exp(-b_val));

    // DeltaNet Recurrent Update: S[i, :] is row i
    device float* state_head = ssm_state + h * ssm_state_size * ssm_state_size;
    device float* s_row = state_head + i * ssm_state_size;

    // 1. Decay state first (matches upstream llama.cpp kernel_gated_delta_net_impl)
    for (uint j = 0; j < ssm_state_size; j++) {
        s_row[j] *= decay;
    }

    // 2. kv_mem_i = dot(S_decayed[i, :], k)
    float kv_mem = 0.0f;
    for (uint j = 0; j < ssm_state_size; j++) {
        kv_mem += s_row[j] * tg_k[j];
    }

    // 3. delta_i = (v_i - kv_mem_i) * beta
    float delta = (my_v - kv_mem) * beta_val;

    // 4. Update state: S[i, j] += delta * k[j]
    // 5. Output projection: y_i = dot(S[i, :], q)
    float y_val = 0.0f;
    for (uint j = 0; j < ssm_state_size; j++) {
        float s = s_row[j] + delta * tg_k[j];
        s_row[j] = s;
        y_val += s * tg_q[j];
    }

    // Threadgroup RMS reduction across 128 threads (4 SIMD groups of 32 threads)
    threadgroup float tg_y_sq[4];
    float y_sq_simd = simd_sum(y_val * y_val);
    if (lane_id == 0) {
        tg_y_sq[simd_id] = y_sq_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_y_sq = tg_y_sq[0] + tg_y_sq[1] + tg_y_sq[2] + tg_y_sq[3];
    float rms = rsqrt(total_y_sq / float(ssm_state_size) + eps);

    // Apply ssm_norm
    float normed_y = y_val * rms;
    if (ssm_norm_w != nullptr) {
        normed_y *= ssm_norm_w[i];
    }

    // Gate with SiLU(gate)
    device const float* gate_head = ssm_gate + h * ssm_state_size;
    float g_val = gate_head[i];
    float silu_g = g_val / (1.0f + exp(-g_val));

    device float* out_head = ssm_out + h * ssm_state_size;
    out_head[i] = normed_y * silu_g;
}

// Deinterleave Q and Gate from 12288 raw projection (24 heads x (256 Q + 256 Gate))
kernel void kernel_qwen35_split_q_gate(
    device float* q_out              [[buffer(0)]],
    device float* gate_out           [[buffer(1)]],
    device const float* q_gate_in    [[buffer(2)]],
    constant uint& total_elements    [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid < total_elements) {
        uint h = tid / 256;
        uint d = tid % 256;
        q_out[tid] = q_gate_in[h * 512 + d];
        gate_out[tid] = q_gate_in[h * 512 + 256 + d];
    }
}

// Post-attention output gating: attn_out[i] *= sigmoid(gate[i])
kernel void kernel_qwen35_attn_gate(
    device float* attn_out           [[buffer(0)]],
    device const float* gate         [[buffer(1)]],
    constant uint& total_elements    [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid < total_elements) {
        float g = gate[tid];
        float sig = 1.0f / (1.0f + exp(-g));
        attn_out[tid] *= sig;
    }
}

// Fuses per-head RMSNorm (256-dim) and partial RoPE (first 64-dim)
kernel void kernel_qwen35_head_norm_rope(
    device float* x                  [[buffer(0)]],
    device const float* norm_w       [[buffer(1)]],
    constant uint& num_heads         [[buffer(2)]],
    constant uint& head_dim          [[buffer(3)]],
    constant uint& rope_dim          [[buffer(4)]],
    constant uint& pos               [[buffer(5)]],
    constant float& theta            [[buffer(6)]],
    constant float& eps              [[buffer(7)]],
    uint h                           [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    if (h >= num_heads) return;

    device float* head_ptr = x + h * head_dim;

    // head_dim = 256. 256 threads per threadgroup (8 SIMD groups of 32 threads)
    float val = (tid < head_dim) ? head_ptr[tid] : 0.0f;
    float my_sq = val * val;
    float simd_sq = simd_sum(my_sq);

    threadgroup float tg_sq[8];
    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        tg_sq[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3] +
                     tg_sq[4] + tg_sq[5] + tg_sq[6] + tg_sq[7];
    float rms = rsqrt(total_sq / float(head_dim) + eps);

    if (tid < head_dim) {
        float w = (norm_w != nullptr) ? norm_w[tid] : 1.0f;
        val = val * rms * w;
        head_ptr[tid] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply RoPE on first rope_dim dimensions (rope_dim = 64, half_dim = 32)
    uint half_dim = rope_dim / 2;
    if (tid < half_dim) {
        float freq = 1.0f / pow(theta, float(2 * tid) / float(rope_dim));
        float angle = float(pos) * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        float v0 = head_ptr[tid];
        float v1 = head_ptr[tid + half_dim];
        head_ptr[tid]            = v0 * cos_val - v1 * sin_val;
        head_ptr[tid + half_dim] = v0 * sin_val + v1 * cos_val;
    }
}

// --- Batched Prefill Kernels for Qwen 3.5 Hybrid SSM & Attention ---

kernel void kernel_conv1d_batch(
    device float* out                [[buffer(0)]],
    device const float* in           [[buffer(1)]],
    device float* state              [[buffer(2)]],
    device const float* conv_weight  [[buffer(3)]],
    constant uint& kernel_size       [[buffer(4)]],
    constant uint& channels          [[buffer(5)]],
    constant uint& batch_size        [[buffer(6)]],
    uint c                           [[thread_position_in_grid]]
) {
    if (c >= channels) return;

    if (state == nullptr) {
        for (uint b = 0; b < batch_size; b++) {
            float u = in[b * channels + c];
            out[b * channels + c] = u / (1.0f + exp(-u));
        }
        return;
    }

    float s0 = state[0 * channels + c];
    float s1 = state[1 * channels + c];
    float s2 = state[2 * channels + c];

    float w0 = conv_weight[c * kernel_size + 0];
    float w1 = conv_weight[c * kernel_size + 1];
    float w2 = conv_weight[c * kernel_size + 2];
    float w3 = conv_weight[c * kernel_size + 3];

    for (uint b = 0; b < batch_size; b++) {
        float u = in[b * channels + c];
        float sum = w0 * s0 + w1 * s1 + w2 * s2 + w3 * u;
        out[b * channels + c] = sum / (1.0f + exp(-sum));
        s0 = s1;
        s1 = s2;
        s2 = u;
    }

    state[0 * channels + c] = s0;
    state[1 * channels + c] = s1;
    state[2 * channels + c] = s2;
}

kernel void kernel_ssm_batch(
    device float* ssm_out            [[buffer(0)]],
    device const float* conv_out     [[buffer(1)]],
    device const float* ssm_alpha    [[buffer(2)]],
    device const float* ssm_beta     [[buffer(3)]],
    device const float* dt_bias      [[buffer(4)]],
    device const float* ssm_a        [[buffer(5)]],
    device const float* ssm_norm_w   [[buffer(6)]],
    device const float* ssm_gate     [[buffer(7)]],
    device float* ssm_state          [[buffer(8)]],
    constant uint& ssm_inner         [[buffer(9)]],
    constant uint& ssm_state_size    [[buffer(10)]],
    constant uint& ssm_groups        [[buffer(11)]],
    constant uint& ssm_rank          [[buffer(12)]],
    constant float& eps              [[buffer(13)]],
    constant uint& batch_size        [[buffer(14)]],
    constant uint& ssm_channels      [[buffer(15)]],
    uint h                           [[threadgroup_position_in_grid]],
    uint i                           [[thread_index_in_threadgroup]]
) {
    if (h >= ssm_rank || i >= ssm_state_size) return;

    uint g = h % ssm_groups;
    uint q_base = 0;
    uint k_base = ssm_groups * ssm_state_size;
    uint v_base = 2 * ssm_groups * ssm_state_size;

    threadgroup float tg_k[128];
    threadgroup float tg_q[128];
    threadgroup float tg_q_sq[4];
    threadgroup float tg_k_sq[4];
    threadgroup float tg_y_sq[4];

    uint simd_id = i / 32;
    uint lane_id = i % 32;

    device float* state_head = ssm_state + h * ssm_state_size * ssm_state_size;
    device float* s_row = state_head + i * ssm_state_size;

    float dt_b = (dt_bias != nullptr) ? dt_bias[h] : 0.0f;
    float a_val = (ssm_a != nullptr) ? ssm_a[h] : 0.0f;
    float norm_w_val = (ssm_norm_w != nullptr) ? ssm_norm_w[i] : 1.0f;

    float s_local[128];
    #pragma unroll
    for (uint j = 0; j < ssm_state_size; j++) {
        s_local[j] = s_row[j];
    }

    for (uint b = 0; b < batch_size; b++) {
        device const float* cur_conv = conv_out + b * ssm_channels;
        device const float* q_head = cur_conv + q_base + g * ssm_state_size;
        device const float* k_head = cur_conv + k_base + g * ssm_state_size;
        device const float* v_head = cur_conv + v_base + h * ssm_state_size;

        float my_q = q_head[i];
        float my_k = k_head[i];
        float my_v = v_head[i];

        float q_sq_simd = simd_sum(my_q * my_q);
        float k_sq_simd = simd_sum(my_k * my_k);
        if (lane_id == 0) {
            tg_q_sq[simd_id] = q_sq_simd;
            tg_k_sq[simd_id] = k_sq_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float q_sq_total = tg_q_sq[0] + tg_q_sq[1] + tg_q_sq[2] + tg_q_sq[3];
        float k_sq_total = tg_k_sq[0] + tg_k_sq[1] + tg_k_sq[2] + tg_k_sq[3];

        float inv_q_norm = rsqrt(q_sq_total + 1e-6f) * (1.0f / sqrt(float(ssm_state_size)));
        float inv_k_norm = rsqrt(k_sq_total + 1e-6f);

        tg_q[i] = my_q * inv_q_norm;
        tg_k[i] = my_k * inv_k_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float alpha_val = ssm_alpha[b * ssm_rank + h] + dt_b;
        float dt = (alpha_val > 20.0f) ? alpha_val : log(1.0f + exp(alpha_val));
        float decay = exp(dt * a_val);

        float b_val = (ssm_beta != nullptr) ? ssm_beta[b * ssm_rank + h] : 0.0f;
        float beta_val = 1.0f / (1.0f + exp(-b_val));

        float kv_mem = 0.0f;
        #pragma unroll
        for (uint j = 0; j < ssm_state_size; j++) {
            s_local[j] *= decay;
            kv_mem += s_local[j] * tg_k[j];
        }

        float delta = (my_v - kv_mem) * beta_val;

        float y_val = 0.0f;
        #pragma unroll
        for (uint j = 0; j < ssm_state_size; j++) {
            s_local[j] += delta * tg_k[j];
            y_val += s_local[j] * tg_q[j];
        }

        float y_sq_simd = simd_sum(y_val * y_val);
        if (lane_id == 0) {
            tg_y_sq[simd_id] = y_sq_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_y_sq = tg_y_sq[0] + tg_y_sq[1] + tg_y_sq[2] + tg_y_sq[3];
        float rms = rsqrt(total_y_sq / float(ssm_state_size) + eps);

        float normed_y = y_val * rms * norm_w_val;

        device const float* cur_gate = ssm_gate + b * ssm_inner;
        float g_val = cur_gate[h * ssm_state_size + i];
        float silu_g = g_val / (1.0f + exp(-g_val));

        device float* cur_out = ssm_out + b * ssm_inner;
        cur_out[h * ssm_state_size + i] = normed_y * silu_g;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint j = 0; j < ssm_state_size; j++) {
        s_row[j] = s_local[j];
    }
}

kernel void kernel_attention_gqa_batch(
    device float* attn_out           [[buffer(0)]],
    device const float* q            [[buffer(1)]],
    device const float* k_cache      [[buffer(2)]],
    device const float* v_cache      [[buffer(3)]],
    constant uint& num_heads         [[buffer(4)]],
    constant uint& num_kv_heads      [[buffer(5)]],
    constant uint& head_dim          [[buffer(6)]],
    constant uint& start_pos         [[buffer(7)]],
    constant uint& max_seq           [[buffer(8)]],
    constant float& attn_scale       [[buffer(9)]],
    uint2 tg_pos                     [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint h = tg_pos.x;
    uint b = tg_pos.y;
    if (h >= num_heads) return;

    uint active_context = start_pos + b + 1;
    if (active_context > max_seq) active_context = max_seq;

    uint kv_mul = num_heads / num_kv_heads;
    uint kv_head = h / kv_mul;
    uint kv_dim = num_kv_heads * head_dim;

    device const float* q_h = q + (b * num_heads + h) * head_dim;
    device float* out_h = attn_out + (b * num_heads + h) * head_dim;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    thread float thread_accum[8] = {0.0f};

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

        uint step = 0;
        for (uint d = tid; d < head_dim; d += 32) {
            thread_accum[step] = thread_accum[step] * scale_prev + exp_val * v_t[d];
            step++;
        }
    }

    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    uint step = 0;
    for (uint d = tid; d < head_dim; d += 32) {
        out_h[d] = thread_accum[step] * inv_sum;
        step++;
    }
}

kernel void kernel_kv_write_batch(
    device float* k_cache            [[buffer(0)]],
    device float* v_cache            [[buffer(1)]],
    device const float* k            [[buffer(2)]],
    device const float* v            [[buffer(3)]],
    constant uint& start_pos         [[buffer(4)]],
    constant uint& max_seq           [[buffer(5)]],
    constant uint& kv_dim            [[buffer(6)]],
    constant uint& batch_size        [[buffer(7)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint total_elems = batch_size * kv_dim;
    if (tid < total_elems) {
        uint b = tid / kv_dim;
        uint d = tid % kv_dim;
        uint slot = (start_pos + b) % max_seq;
        k_cache[slot * kv_dim + d] = k[tid];
        v_cache[slot * kv_dim + d] = v[tid];
    }
}

kernel void kernel_qwen35_split_q_gate_batch(
    device float* q_out              [[buffer(0)]],
    device float* gate_out           [[buffer(1)]],
    device const float* q_gate_in    [[buffer(2)]],
    constant uint& num_tokens        [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint total_elems = num_tokens * 24 * 256;
    if (tid < total_elems) {
        uint tok = tid / (24 * 256);
        uint in_tok = tid % (24 * 256);
        uint h = in_tok / 256;
        uint d = in_tok % 256;
        q_out[tid] = q_gate_in[tok * 12288 + h * 512 + d];
        gate_out[tid] = q_gate_in[tok * 12288 + h * 512 + 256 + d];
    }
}

kernel void kernel_rope_norm_batch(
    device float* q                  [[buffer(0)]],
    device float* k                  [[buffer(1)]],
    device const float* q_norm_w     [[buffer(2)]],
    device const float* k_norm_w     [[buffer(3)]],
    constant uint& start_pos         [[buffer(4)]],
    constant uint& num_heads         [[buffer(5)]],
    constant uint& num_kv_heads      [[buffer(6)]],
    constant uint& head_dim          [[buffer(7)]],
    constant uint& rope_dim          [[buffer(8)]],
    constant float& theta            [[buffer(9)]],
    constant float& eps              [[buffer(10)]],
    constant uint& batch_size        [[buffer(11)]],
    uint2 tg_pos                     [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint b = tg_pos.y;
    uint h = tg_pos.x;
    if (b >= batch_size) return;
    uint pos = start_pos + b;

    // Head Q norm + RoPE
    if (h < num_heads) {
        device float* q_h = q + (b * num_heads + h) * head_dim;

        float sum_sq = 0.0f;
        for (uint d = tid; d < head_dim; d += 32) {
            float v = q_h[d];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        float scale = rsqrt(sum_sq / float(head_dim) + eps);

        for (uint d = tid; d < head_dim; d += 32) {
            q_h[d] = q_h[d] * scale * q_norm_w[d];
        }

        // RoPE on Q
        uint half_rope = rope_dim / 2;
        for (uint d = tid; d < half_rope; d += 32) {
            float freq = 1.0f / pow(theta, float(2 * d) / float(rope_dim));
            float val = float(pos) * freq;
            float cos_val = cos(val);
            float sin_val = sin(val);
            float v0 = q_h[d];
            float v1 = q_h[d + half_rope];
            q_h[d]              = v0 * cos_val - v1 * sin_val;
            q_h[d + half_rope]  = v0 * sin_val + v1 * cos_val;
        }
    }

    // Head K norm + RoPE
    if (h < num_kv_heads) {
        device float* k_h = k + (b * num_kv_heads + h) * head_dim;

        float sum_sq = 0.0f;
        for (uint d = tid; d < head_dim; d += 32) {
            float v = k_h[d];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        float scale = rsqrt(sum_sq / float(head_dim) + eps);

        for (uint d = tid; d < head_dim; d += 32) {
            k_h[d] = k_h[d] * scale * k_norm_w[d];
        }

        // RoPE on K
        uint half_rope = rope_dim / 2;
        for (uint d = tid; d < half_rope; d += 32) {
            float freq = 1.0f / pow(theta, float(2 * d) / float(rope_dim));
            float val = float(pos) * freq;
            float cos_val = cos(val);
            float sin_val = sin(val);
            float v0 = k_h[d];
            float v1 = k_h[d + half_rope];
            k_h[d]              = v0 * cos_val - v1 * sin_val;
            k_h[d + half_rope]  = v0 * sin_val + v1 * cos_val;
        }
    }
}

kernel void kernel_embed_lookup_q4_k(
    device float* out_x                     [[buffer(0)]],
    device const block_q4_k* w_embd         [[buffer(1)]],
    constant uint& token_id                 [[buffer(2)]],
    constant uint& dim                      [[buffer(3)]],
    uint tid                                [[thread_index_in_threadgroup]]
) {
    uint nb = dim / 256;
    device const block_q4_k* tok_blocks = w_embd + token_id * nb;

    for (uint sb = tid; sb < nb * 8; sb += 64) {
        uint b = sb / 8;
        uint s = sb % 8;
        device const block_q4_k& blk = tok_blocks[b];
        float d, m;
        get_scale_min_k4(s, blk.scales, d, m, float(blk.d), float(blk.dmin));

        uint pair = s / 2;
        bool is_high = (s % 2) != 0;
        device const uint8_t* qs_pair = blk.qs + pair * 32;

        if (!is_high) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                out_x[b * 256 + pair * 64 + i] = float(qs_pair[i] & 0x0F) * d - m;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                out_x[b * 256 + pair * 64 + 32 + i] = float(qs_pair[i] >> 4) * d - m;
            }
        }
    }
}

kernel void kernel_embed_lookup_q4_k_batch(
    device float* out_x                     [[buffer(0)]],
    device const block_q4_k* w_embd         [[buffer(1)]],
    device const uint32_t* token_ids        [[buffer(2)]],
    constant uint& dim                      [[buffer(3)]],
    constant uint& batch_size               [[buffer(4)]],
    uint b_idx                              [[threadgroup_position_in_grid]],
    uint tid                                [[thread_index_in_threadgroup]]
) {
    if (b_idx >= batch_size) return;
    uint token_id = token_ids[b_idx];
    uint nb = dim / 256;
    device const block_q4_k* tok_blocks = w_embd + token_id * nb;
    device float* out_cur = out_x + b_idx * dim;

    for (uint sb = tid; sb < nb * 8; sb += 64) {
        uint b = sb / 8;
        uint s = sb % 8;
        device const block_q4_k& blk = tok_blocks[b];
        float d, m;
        get_scale_min_k4(s, blk.scales, d, m, float(blk.d), float(blk.dmin));

        uint pair = s / 2;
        bool is_high = (s % 2) != 0;
        device const uint8_t* qs_pair = blk.qs + pair * 32;

        if (!is_high) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                out_cur[b * 256 + pair * 64 + i] = float(qs_pair[i] & 0x0F) * d - m;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                out_cur[b * 256 + pair * 64 + 32 + i] = float(qs_pair[i] >> 4) * d - m;
            }
        }
    }
}

kernel void kernel_qwen35_fused_split_norm_rope(
    device float* out_q              [[buffer(0)]],
    device float* out_gate           [[buffer(1)]],
    device const float* q_raw        [[buffer(2)]],
    device const float* norm_w       [[buffer(3)]],
    constant uint& num_heads         [[buffer(4)]],
    constant uint& head_dim          [[buffer(5)]],
    constant uint& rope_dim          [[buffer(6)]],
    constant uint& pos               [[buffer(7)]],
    constant float& theta            [[buffer(8)]],
    constant float& eps              [[buffer(9)]],
    uint h                           [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    if (h >= num_heads) return;

    device const float* head_raw = q_raw + h * 512;
    device float* head_q = out_q + h * head_dim;
    device float* head_g = out_gate + h * head_dim;

    if (tid < head_dim) {
        head_g[tid] = head_raw[head_dim + tid];
    }

    float val = (tid < head_dim) ? head_raw[tid] : 0.0f;
    float my_sq = val * val;
    float simd_sq = simd_sum(my_sq);

    threadgroup float tg_sq[8];
    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    if (lane_id == 0) {
        tg_sq[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3] +
                     tg_sq[4] + tg_sq[5] + tg_sq[6] + tg_sq[7];
    float rms = rsqrt(total_sq / float(head_dim) + eps);

    if (tid < head_dim) {
        float w = (norm_w != nullptr) ? norm_w[tid] : 1.0f;
        val = val * rms * w;
        head_q[tid] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_dim = rope_dim / 2;
    if (tid < half_dim) {
        float freq = 1.0f / pow(theta, float(2 * tid) / float(rope_dim));
        float angle = float(pos) * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);
        float v0 = head_q[tid];
        float v1 = head_q[tid + half_dim];
        head_q[tid]            = v0 * cos_val - v1 * sin_val;
        head_q[tid + half_dim] = v0 * sin_val + v1 * cos_val;
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
    id<MTLFunction> fn_gemm_fused = [library newFunctionWithName:@"gemm_fused_gate_up_swiglu_q4_k_batched"];

    id<MTLFunction> fn_rmsnorm  = [library newFunctionWithName:@"kernel_rmsnorm"];
    id<MTLFunction> fn_rope     = [library newFunctionWithName:@"kernel_rope"];
    id<MTLFunction> fn_attn     = [library newFunctionWithName:@"kernel_attention_gqa"];
    id<MTLFunction> fn_kv_write = [library newFunctionWithName:@"kernel_kv_write"];
    id<MTLFunction> fn_swiglu   = [library newFunctionWithName:@"kernel_swiglu"];
    id<MTLFunction> fn_residual = [library newFunctionWithName:@"kernel_add_residual"];
    id<MTLFunction> fn_res_norm = [library newFunctionWithName:@"kernel_residual_rmsnorm"];

    g_pipeline_f32      = [g_device newComputePipelineStateWithFunction:fn_f32 error:&error];
    g_pipeline_f16      = [g_device newComputePipelineStateWithFunction:fn_f16 error:&error];
    g_pipeline_q4_0     = [g_device newComputePipelineStateWithFunction:fn_q4_0 error:&error];
    g_pipeline_q8_0     = [g_device newComputePipelineStateWithFunction:fn_q8_0 error:&error];
    g_pipeline_q4_k     = [g_device newComputePipelineStateWithFunction:fn_q4_k error:&error];
    g_pipeline_q6_k     = [g_device newComputePipelineStateWithFunction:fn_q6_k error:&error];
    if (fn_gemm_fused) g_pipeline_gemm_fused_gate_up_q4_k = [g_device newComputePipelineStateWithFunction:fn_gemm_fused error:&error];
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
    if (fn_res_norm) {
        g_pipeline_residual_rmsnorm = [g_device newComputePipelineStateWithFunction:fn_res_norm error:&error];
    }

    id<MTLFunction> fn_conv1d_step = [library newFunctionWithName:@"kernel_conv1d_step"];
    id<MTLFunction> fn_ssm_step    = [library newFunctionWithName:@"kernel_ssm_step"];
    id<MTLFunction> fn_split_q_gate = [library newFunctionWithName:@"kernel_qwen35_split_q_gate"];
    id<MTLFunction> fn_attn_gate   = [library newFunctionWithName:@"kernel_qwen35_attn_gate"];
    id<MTLFunction> fn_norm_rope   = [library newFunctionWithName:@"kernel_qwen35_head_norm_rope"];
    id<MTLFunction> fn_embed_q4k   = [library newFunctionWithName:@"kernel_embed_lookup_q4_k"];
    id<MTLFunction> fn_fused_split = [library newFunctionWithName:@"kernel_qwen35_fused_split_norm_rope"];

    if (fn_conv1d_step) g_pipeline_conv1d_step         = [g_device newComputePipelineStateWithFunction:fn_conv1d_step error:&error];
    if (fn_ssm_step)    g_pipeline_ssm_step            = [g_device newComputePipelineStateWithFunction:fn_ssm_step error:&error];
    if (fn_split_q_gate) g_pipeline_qwen35_split_q_gate = [g_device newComputePipelineStateWithFunction:fn_split_q_gate error:&error];
    if (fn_attn_gate)    g_pipeline_qwen35_attn_gate     = [g_device newComputePipelineStateWithFunction:fn_attn_gate error:&error];
    if (fn_norm_rope)   g_pipeline_qwen35_norm_rope    = [g_device newComputePipelineStateWithFunction:fn_norm_rope error:&error];
    if (fn_embed_q4k)   g_pipeline_embed_lookup_q4_k   = [g_device newComputePipelineStateWithFunction:fn_embed_q4k error:&error];
    if (fn_fused_split) g_pipeline_qwen35_fused_split_norm_rope = [g_device newComputePipelineStateWithFunction:fn_fused_split error:&error];


    id<MTLFunction> fn_conv1d_b    = [library newFunctionWithName:@"kernel_conv1d_batch"];
    id<MTLFunction> fn_ssm_b       = [library newFunctionWithName:@"kernel_ssm_batch"];
    id<MTLFunction> fn_attn_b      = [library newFunctionWithName:@"kernel_attention_gqa_batch"];
    id<MTLFunction> fn_kv_w_b      = [library newFunctionWithName:@"kernel_kv_write_batch"];
    id<MTLFunction> fn_split_b     = [library newFunctionWithName:@"kernel_qwen35_split_q_gate_batch"];
    id<MTLFunction> fn_rope_b      = [library newFunctionWithName:@"kernel_rope_norm_batch"];

    if (fn_conv1d_b) g_pipeline_conv1d_batch        = [g_device newComputePipelineStateWithFunction:fn_conv1d_b error:&error];
    if (fn_ssm_b)    g_pipeline_ssm_batch           = [g_device newComputePipelineStateWithFunction:fn_ssm_b error:&error];
    if (fn_attn_b)   g_pipeline_attention_gqa_batch = [g_device newComputePipelineStateWithFunction:fn_attn_b error:&error];
    if (fn_kv_w_b)   g_pipeline_kv_write_batch      = [g_device newComputePipelineStateWithFunction:fn_kv_w_b error:&error];
    if (fn_split_b)  g_pipeline_split_q_gate_batch  = [g_device newComputePipelineStateWithFunction:fn_split_b error:&error];
    if (fn_rope_b)   g_pipeline_rope_norm_batch     = [g_device newComputePipelineStateWithFunction:fn_rope_b error:&error];

    id<MTLFunction> fn_norm_b     = [library newFunctionWithName:@"kernel_rmsnorm_batch"];
    id<MTLFunction> fn_res_norm_b = [library newFunctionWithName:@"kernel_residual_rmsnorm_batch"];
    id<MTLFunction> fn_embed_b    = [library newFunctionWithName:@"kernel_embed_lookup_q4_k_batch"];

    if (fn_norm_b)     g_pipeline_rmsnorm_batch          = [g_device newComputePipelineStateWithFunction:fn_norm_b error:&error];
    if (fn_res_norm_b) g_pipeline_residual_rmsnorm_batch = [g_device newComputePipelineStateWithFunction:fn_res_norm_b error:&error];
    if (fn_embed_b)    g_pipeline_embed_lookup_q4_k_batch = [g_device newComputePipelineStateWithFunction:fn_embed_b error:&error];

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
    g_buf_token     = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    if (!g_k_cache || !g_v_cache || !g_buf_x || !g_buf_logits || !g_buf_token) return -2;
    return 0;
}

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
) {
    if (!metal_is_available()) return -1;

    // Allocate standard transformer buffers with properly scaled dims
    uint32_t alloc_dim = dim;
    if (ssm_channels > alloc_dim) alloc_dim = ssm_channels;
    if (12288 > alloc_dim) alloc_dim = 12288;

    int ret = metal_alloc_buffers(alloc_dim, hidden_dim, kv_dim, vocab_size, num_layers, max_seq);
    if (ret != 0) return ret;

    // Allocate specialized SSM working buffers
    g_buf_ssm_gate         = [g_device newBufferWithLength:ssm_inner * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_ssm_qkv          = [g_device newBufferWithLength:ssm_channels * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_ssm_conv_out     = [g_device newBufferWithLength:ssm_channels * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_ssm_alpha        = [g_device newBufferWithLength:ssm_rank * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_ssm_beta         = [g_device newBufferWithLength:ssm_rank * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_ssm_out          = [g_device newBufferWithLength:ssm_inner * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_q_raw            = [g_device newBufferWithLength:12288 * sizeof(float) options:MTLResourceStorageModeShared];
    g_buf_qwen35_attn_gate = [g_device newBufferWithLength:6144 * sizeof(float) options:MTLResourceStorageModeShared];

    // Allocate persistent recurrent SSM states across all layers
    size_t conv_bytes = (size_t)num_layers * 3 * ssm_channels * sizeof(float);
    size_t ssm_bytes  = (size_t)num_layers * ssm_rank * ssm_state_size * ssm_state_size * sizeof(float);

    g_ssm_conv_state = [g_device newBufferWithLength:conv_bytes options:MTLResourceStorageModeShared];
    g_ssm_state      = [g_device newBufferWithLength:ssm_bytes  options:MTLResourceStorageModeShared];

    if (!g_buf_ssm_gate || !g_buf_ssm_qkv || !g_buf_ssm_conv_out || !g_buf_ssm_alpha || !g_buf_ssm_beta ||
        !g_buf_ssm_out || !g_buf_q_raw || !g_buf_qwen35_attn_gate || !g_ssm_conv_state || !g_ssm_state) {
        return -3;
    }

    memset(g_ssm_conv_state.contents, 0, conv_bytes);
    memset(g_ssm_state.contents, 0, ssm_bytes);

    return 0;
}

void metal_reset_ssm_state(void) {
    if (g_ssm_conv_state) {
        memset(g_ssm_conv_state.contents, 0, g_ssm_conv_state.length);
    }
    if (g_ssm_state) {
        memset(g_ssm_state.contents, 0, g_ssm_state.length);
    }
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

        MTLSize threadgroups, threadsPerGroup;
        if (pipeline == g_pipeline_q4_k) {
            threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
            threadsPerGroup = MTLSizeMake(64, 1, 1);
        } else if (pipeline == g_pipeline_q6_k) {
            threadgroups = MTLSizeMake((rows + 15) / 16, 1, 1);
            threadsPerGroup = MTLSizeMake(128, 1, 1);
        } else {
            threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
            threadsPerGroup = MTLSizeMake(128, 1, 1);
        }

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

        MTLSize threadgroups, threadsPerGroup;
        if (pipeline == g_pipeline_q4_k) {
            threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
            threadsPerGroup = MTLSizeMake(64, 1, 1);
        } else if (pipeline == g_pipeline_q6_k) {
            threadgroups = MTLSizeMake((rows + 15) / 16, 1, 1);
            threadsPerGroup = MTLSizeMake(128, 1, 1);
        } else {
            threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
            threadsPerGroup = MTLSizeMake(128, 1, 1);
        }

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

        uint32_t batch_dim = batch_size;
        uint32_t row_dim = rows;
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        if (pipeline == g_pipeline_gemm_q4_k) {
            batch_dim = (batch_size + 3) / 4;
            row_dim = (rows + 15) / 16;
            tpg = MTLSizeMake(128, 1, 1);
        } else if (pipeline == g_pipeline_gemm_q6_k) {
            batch_dim = (batch_size + 3) / 4;
            row_dim = (rows + 7) / 8;
            tpg = MTLSizeMake(128, 1, 1);
        }
        MTLSize tgs = MTLSizeMake(batch_dim, row_dim, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_gemm_buf(int quant_type, float* y, const float* x, metal_buffer_t w_buf, uint32_t batch_size, uint32_t rows, uint32_t cols) {
    if (!metal_is_available() || !w_buf) return -1;
    id<MTLComputePipelineState> pipeline = nil;
    switch (quant_type) {
        case 2: pipeline = g_pipeline_gemm_q4_0; break;
        case 8: pipeline = g_pipeline_gemm_q8_0; break;
        case 12: pipeline = g_pipeline_gemm_q4_k; break;
        case 14: pipeline = g_pipeline_gemm_q6_k; break;
        default: return -2;
    }

    @autoreleasepool {
        id<MTLBuffer> buf_y = [g_device newBufferWithBytesNoCopy:y length:batch_size * rows * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_x = [g_device newBufferWithBytesNoCopy:(void*)x length:batch_size * cols * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_w = (__bridge id<MTLBuffer>)w_buf;

        if (!buf_y || !buf_x || !buf_w) return -3;

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

        uint32_t batch_dim = batch_size;
        uint32_t row_dim = rows;
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        if (pipeline == g_pipeline_gemm_q4_k) {
            batch_dim = (batch_size + 3) / 4;
            row_dim = (rows + 15) / 16;
            tpg = MTLSizeMake(128, 1, 1);
        } else if (pipeline == g_pipeline_gemm_q6_k) {
            batch_dim = (batch_size + 3) / 4;
            row_dim = (rows + 7) / 8;
            tpg = MTLSizeMake(128, 1, 1);
        }
        MTLSize tgs = MTLSizeMake(batch_dim, row_dim, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_gemm_fused_gate_up_buf(int quant_type, float* y, const float* x, metal_buffer_t gate_buf, metal_buffer_t up_buf, uint32_t batch_size, uint32_t rows, uint32_t cols) {
    if (!metal_is_available() || !gate_buf || !up_buf) return -1;
    if (quant_type != 12 || !g_pipeline_gemm_fused_gate_up_q4_k) return -2;

    @autoreleasepool {
        id<MTLBuffer> buf_y = [g_device newBufferWithBytesNoCopy:y length:batch_size * rows * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_x = [g_device newBufferWithBytesNoCopy:(void*)x length:batch_size * cols * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_gate = (__bridge id<MTLBuffer>)gate_buf;
        id<MTLBuffer> buf_up   = (__bridge id<MTLBuffer>)up_buf;

        if (!buf_y || !buf_x || !buf_gate || !buf_up) return -3;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_gemm_fused_gate_up_q4_k];
        [enc setBuffer:buf_y offset:0 atIndex:0];
        [enc setBuffer:buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_gate offset:0 atIndex:2];
        [enc setBuffer:buf_up offset:0 atIndex:3];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&rows length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&cols length:sizeof(uint32_t) atIndex:6];

        uint32_t batch_dim = (batch_size + 3) / 4;
        uint32_t row_dim = (rows + 15) / 16;
        MTLSize tgs = MTLSizeMake(batch_dim, row_dim, 1);
        MTLSize tpg = MTLSizeMake(128, 1, 1);
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

int metal_rmsnorm_batch(float* out, const float* x, const float* weight, uint32_t dim, float eps, uint32_t batch_size) {
    if (!metal_is_available() || !g_pipeline_rmsnorm_batch || batch_size == 0) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_out = [g_device newBufferWithBytesNoCopy:out length:batch_size * dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_x   = [g_device newBufferWithBytesNoCopy:(void*)x length:batch_size * dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_w   = [g_device newBufferWithBytesNoCopy:(void*)weight length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_out || !buf_x || !buf_w) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_rmsnorm_batch];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_w offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:5];

        MTLSize tgs = MTLSizeMake(batch_size, 1, 1);
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

int metal_residual_rmsnorm_batch(float* x, const float* proj, float* out_norm, const float* weight, uint32_t dim, float eps, uint32_t batch_size) {
    if (!metal_is_available() || !g_pipeline_residual_rmsnorm_batch || batch_size == 0) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_x    = [g_device newBufferWithBytesNoCopy:x length:batch_size * dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_proj = [g_device newBufferWithBytesNoCopy:(void*)proj length:batch_size * dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_out  = [g_device newBufferWithBytesNoCopy:out_norm length:batch_size * dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_w    = [g_device newBufferWithBytesNoCopy:(void*)weight length:dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_x || !buf_proj || !buf_out || !buf_w) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_residual_rmsnorm_batch];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_proj offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        [enc setBuffer:buf_w offset:0 atIndex:3];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float) atIndex:5];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:6];

        MTLSize tgs = MTLSizeMake(batch_size, 1, 1);
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

int metal_embed_lookup_q4_k_batch(float* out_x, metal_buffer_t w_embd, const uint32_t* token_ids, uint32_t dim, uint32_t batch_size) {
    if (!metal_is_available() || !g_pipeline_embed_lookup_q4_k_batch || !w_embd || batch_size == 0) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_out = [g_device newBufferWithBytesNoCopy:out_x length:batch_size * dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_tokens = [g_device newBufferWithBytesNoCopy:(void*)token_ids length:batch_size * sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_embd = (__bridge id<MTLBuffer>)w_embd;

        if (!buf_out || !buf_tokens || !buf_embd) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_embed_lookup_q4_k_batch];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_embd offset:0 atIndex:1];
        [enc setBuffer:buf_tokens offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];

        MTLSize tgs = MTLSizeMake(batch_size, 1, 1);
        MTLSize tpg = MTLSizeMake(64, 1, 1);
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
        case 3:
        case 8: return g_pipeline_q8_0;
        case 12: return g_pipeline_q4_k;
        case 14: return g_pipeline_q6_k;
        default: return nil;
    }
}

static inline id<MTLComputePipelineState> get_fused_gate_up_pipeline(int quant_type) {
    switch (quant_type) {
        case 2: return g_pipeline_fused_gate_up_q4_0;
        case 3:
        case 8: return g_pipeline_fused_gate_up_q8_0;
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

    MTLSize threadgroups, threadsPerGroup;
    if (pipeline == g_pipeline_q4_k) {
        threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
        threadsPerGroup = MTLSizeMake(64, 1, 1);
    } else if (pipeline == g_pipeline_q6_k) {
        threadgroups = MTLSizeMake((rows + 15) / 16, 1, 1);
        threadsPerGroup = MTLSizeMake(128, 1, 1);
    } else {
        threadgroups = MTLSizeMake((rows + 7) / 8, 1, 1);
        threadsPerGroup = MTLSizeMake(128, 1, 1);
    }
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
}

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
) {
    if (!metal_is_available() || !initial_x || (!out_logits && !out_token) || !layers) return -1;
    if (!g_buf_x || !g_buf_logits || !output_norm_buf || !output_weight_buf) return -2;

    @autoreleasepool {
        memcpy(g_buf_x.contents, initial_x, dim * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

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
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 2. Q, K, V Projections
            encode_gemv_buf(enc, get_pipeline(lw->wq_type), g_buf_q, g_buf_xb, (__bridge id<MTLBuffer>)lw->wq, dim, dim);
            encode_gemv_buf(enc, get_pipeline(lw->wk_type), g_buf_k, g_buf_xb, (__bridge id<MTLBuffer>)lw->wk, kv_dim, dim);
            encode_gemv_buf(enc, get_pipeline(lw->wv_type), g_buf_v, g_buf_xb, (__bridge id<MTLBuffer>)lw->wv, kv_dim, dim);

            // Add Q, K, V biases if present (e.g. Qwen 2 / 2.5)
            if (lw->bq) {
                [enc setComputePipelineState:g_pipeline_residual];
                [enc setBuffer:g_buf_q offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->bq offset:0 atIndex:1];
                [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            if (lw->bk) {
                [enc setComputePipelineState:g_pipeline_residual];
                [enc setBuffer:g_buf_k offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->bk offset:0 atIndex:1];
                [enc setBytes:(void*)&kv_dim length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            if (lw->bv) {
                [enc setComputePipelineState:g_pipeline_residual];
                [enc setBuffer:g_buf_v offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->bv offset:0 atIndex:1];
                [enc setBytes:(void*)&kv_dim length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

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

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

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

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 5. FlashAttention (GQA)
            size_t layer_kv_offset = (size_t)l * max_seq * kv_dim * sizeof(float);
            [enc setComputePipelineState:g_pipeline_attn];
            [enc setBuffer:g_buf_attn_out offset:0 atIndex:0];
            [enc setBuffer:g_buf_q offset:0 atIndex:1];
            [enc setBuffer:g_k_cache offset:layer_kv_offset atIndex:2];
            [enc setBuffer:g_v_cache offset:layer_kv_offset atIndex:3];
            [enc setBytes:(void*)&num_heads length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:(void*)&num_kv_heads length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:(void*)&head_dim length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:(void*)&active_context length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:(void*)&attn_scale length:sizeof(float) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 6. WO Projection
            encode_gemv_buf(enc, get_pipeline(lw->wo_type), g_buf_attn_proj, g_buf_attn_out, (__bridge id<MTLBuffer>)lw->wo, dim, dim);

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 7. AddResidual (x += attn_proj)
            [enc setComputePipelineState:g_pipeline_residual];
            [enc setBuffer:g_buf_x offset:0 atIndex:0];
            [enc setBuffer:g_buf_attn_proj offset:0 atIndex:1];
            [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 8. FFN RMSNorm (x -> xb)
            [enc setComputePipelineState:g_pipeline_rmsnorm];
            [enc setBuffer:g_buf_xb offset:0 atIndex:0];
            [enc setBuffer:g_buf_x offset:0 atIndex:1];
            [enc setBuffer:buf_ffn_norm offset:0 atIndex:2];
            [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:(void*)&norm_eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

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
                MTLSize tgs, tpg;
                if (lw->ffn_gate_type == 12) {
                    tgs = MTLSizeMake((hidden_dim + 7) / 8, 1, 1);
                    tpg = MTLSizeMake(64, 1, 1);
                } else {
                    tgs = MTLSizeMake((hidden_dim + 7) / 8, 1, 1);
                    tpg = MTLSizeMake(128, 1, 1);
                }
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

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 11. FFN Down Projection
            encode_gemv_buf(enc, get_pipeline(lw->ffn_down_type), g_buf_down, g_buf_gate, (__bridge id<MTLBuffer>)lw->ffn_down, dim, hidden_dim);

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 12. AddResidual (x += ffn_down)
            [enc setComputePipelineState:g_pipeline_residual];
            [enc setBuffer:g_buf_x offset:0 atIndex:0];
            [enc setBuffer:g_buf_down offset:0 atIndex:1];
            [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Final RMSNorm on x -> xb
        id<MTLBuffer> buf_out_norm = (__bridge id<MTLBuffer>)output_norm_buf;
        [enc setComputePipelineState:g_pipeline_rmsnorm];
        [enc setBuffer:g_buf_xb offset:0 atIndex:0];
        [enc setBuffer:g_buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_out_norm offset:0 atIndex:2];
        [enc setBytes:(void*)&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:(void*)&norm_eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Final Output Logits Projection
        encode_gemv_buf(enc, get_pipeline(output_weight_type), g_buf_logits, g_buf_xb, (__bridge id<MTLBuffer>)output_weight_buf, vocab_size, dim);

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        if (out_token && g_pipeline_sample_argmax && g_buf_token) {
            [enc setComputePipelineState:g_pipeline_sample_argmax];
            [enc setBuffer:g_buf_logits offset:0 atIndex:0];
            [enc setBuffer:g_buf_token offset:0 atIndex:1];
            [enc setBytes:&vocab_size length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            printf("Metal command buffer failed! status=%lu, error=%s\n",
                   (unsigned long)[cmd status],
                   [cmd error] ? [[[cmd error] localizedDescription] UTF8String] : "none");
            fflush(stdout);
            return -3;
        }

        if (g_buf_logits && out_logits) {
            memcpy(out_logits, g_buf_logits.contents, vocab_size * sizeof(float));
        }
        if (out_token && g_buf_token) {
            *out_token = *(uint32_t*)g_buf_token.contents;
        }
    }
    return 0;
}

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
) {
    if (!metal_is_available() || (!initial_x && (!token_embd_buf || token_id < 0)) || (!out_logits && !out_token) || !layers) return -1;
    if (!g_buf_x || !g_buf_logits || !output_norm_buf || !output_weight_buf || !g_ssm_state) return -2;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        if (token_embd_buf && token_id >= 0 && g_pipeline_embed_lookup_q4_k) {
            uint32_t u_tok = (uint32_t)token_id;
            [enc setComputePipelineState:g_pipeline_embed_lookup_q4_k];
            [enc setBuffer:g_buf_x offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)token_embd_buf offset:0 atIndex:1];
            [enc setBytes:&u_tok length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else if (initial_x) {
            memcpy(g_buf_x.contents, initial_x, dim * sizeof(float));
        }

        id<MTLBuffer> buf_out_norm = (__bridge id<MTLBuffer>)output_norm_buf;

        // Initial RMSNorm on x -> xb for Layer 0
        id<MTLBuffer> first_attn_norm = (__bridge id<MTLBuffer>)layers[0].attn_norm;
        [enc setComputePipelineState:g_pipeline_rmsnorm];
        [enc setBuffer:g_buf_xb offset:0 atIndex:0];
        [enc setBuffer:g_buf_x offset:0 atIndex:1];
        [enc setBuffer:first_attn_norm offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&norm_eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        for (uint32_t l = 0; l < num_layers; l++) {
            const metal_qwen35_layer_weights_t* lw = &layers[l];
            id<MTLBuffer> buf_ffn_norm  = (__bridge id<MTLBuffer>)lw->ffn_norm;

            if (lw->is_ssm) {
                // SSM Branch
                // 2. Gate, QKV, Alpha, Beta GEMVs execute concurrently!
                encode_gemv_buf(enc, get_pipeline(lw->ssm_gate_type), g_buf_ssm_gate, g_buf_xb, (__bridge id<MTLBuffer>)lw->ssm_gate, ssm_inner, dim);
                encode_gemv_buf(enc, get_pipeline(lw->ssm_qkv_type), g_buf_ssm_qkv, g_buf_xb, (__bridge id<MTLBuffer>)lw->ssm_qkv, ssm_channels, dim);
                encode_gemv_buf(enc, get_pipeline(lw->ssm_alpha_type), g_buf_ssm_alpha, g_buf_xb, (__bridge id<MTLBuffer>)lw->ssm_alpha, ssm_rank, dim);
                encode_gemv_buf(enc, get_pipeline(lw->ssm_beta_type), g_buf_ssm_beta, g_buf_xb, (__bridge id<MTLBuffer>)lw->ssm_beta, ssm_rank, dim);

                // conv1d needs ssm_qkv
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // 3. 1D Causal Convolution
                uint32_t conv_kernel = 4;
                size_t conv_offset = (size_t)l * 3 * ssm_channels * sizeof(float);
                [enc setComputePipelineState:g_pipeline_conv1d_step];
                [enc setBuffer:g_buf_ssm_conv_out offset:0 atIndex:0];
                [enc setBuffer:g_buf_ssm_qkv offset:0 atIndex:1];
                [enc setBuffer:g_ssm_conv_state offset:conv_offset atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ssm_conv1d offset:0 atIndex:3];
                [enc setBytes:&conv_kernel length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&ssm_channels length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((ssm_channels + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                // ssm_step needs conv_out, alpha, beta, gate
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // 5. DeltaNet Recurrent SSM Step
                size_t ssm_offset = (size_t)l * ssm_rank * ssm_state_size * ssm_state_size * sizeof(float);
                [enc setComputePipelineState:g_pipeline_ssm_step];
                [enc setBuffer:g_buf_ssm_out offset:0 atIndex:0];
                [enc setBuffer:g_buf_ssm_conv_out offset:0 atIndex:1];
                [enc setBuffer:g_buf_ssm_alpha offset:0 atIndex:2];
                [enc setBuffer:g_buf_ssm_beta offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ssm_dt_bias offset:0 atIndex:4];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ssm_a offset:0 atIndex:5];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ssm_norm offset:0 atIndex:6];
                [enc setBuffer:g_buf_ssm_gate offset:0 atIndex:7];
                [enc setBuffer:g_ssm_state offset:ssm_offset atIndex:8];
                [enc setBytes:&ssm_inner length:sizeof(uint32_t) atIndex:9];
                [enc setBytes:&ssm_state_size length:sizeof(uint32_t) atIndex:10];
                [enc setBytes:&ssm_groups length:sizeof(uint32_t) atIndex:11];
                [enc setBytes:&ssm_rank length:sizeof(uint32_t) atIndex:12];
                [enc setBytes:&norm_eps length:sizeof(float) atIndex:13];
                [enc dispatchThreadgroups:MTLSizeMake(ssm_rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(ssm_state_size, 1, 1)];

                // ssm_out needs ssm_out buffer
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // 6. SSM Out GEMV
                encode_gemv_buf(enc, get_pipeline(lw->ssm_out_type), g_buf_attn_proj, g_buf_ssm_out, (__bridge id<MTLBuffer>)lw->ssm_out, dim, ssm_inner);
            } else {
                // Attention Branch
                uint32_t qwen35_attn_dim = num_heads * head_dim;
                // 2. Q (12288), K (1024), V (1024) GEMVs execute concurrently!
                encode_gemv_buf(enc, get_pipeline(lw->wq_type), g_buf_q_raw, g_buf_xb, (__bridge id<MTLBuffer>)lw->wq, 12288, dim);
                encode_gemv_buf(enc, get_pipeline(lw->wk_type), g_buf_k, g_buf_xb, (__bridge id<MTLBuffer>)lw->wk, kv_dim, dim);
                encode_gemv_buf(enc, get_pipeline(lw->wv_type), g_buf_v, g_buf_xb, (__bridge id<MTLBuffer>)lw->wv, kv_dim, dim);

                // split_norm_rope needs q_raw, norm_rope needs k
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                if (g_pipeline_qwen35_fused_split_norm_rope) {
                    // 3. Fused Split Q/Gate + RMSNorm + RoPE on Q
                    [enc setComputePipelineState:g_pipeline_qwen35_fused_split_norm_rope];
                    [enc setBuffer:g_buf_q offset:0 atIndex:0];
                    [enc setBuffer:g_buf_qwen35_attn_gate offset:0 atIndex:1];
                    [enc setBuffer:g_buf_q_raw offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)lw->q_norm offset:0 atIndex:3];
                    [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&rope_dim length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&pos length:sizeof(uint32_t) atIndex:7];
                    [enc setBytes:&rope_theta length:sizeof(float) atIndex:8];
                    [enc setBytes:&norm_eps length:sizeof(float) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake(num_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

                    // Concurrent RMSNorm & RoPE on K
                    [enc setComputePipelineState:g_pipeline_qwen35_norm_rope];
                    [enc setBuffer:g_buf_k offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)lw->k_norm offset:0 atIndex:1];
                    [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:2];
                    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&rope_dim length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&pos length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&rope_theta length:sizeof(float) atIndex:6];
                    [enc setBytes:&norm_eps length:sizeof(float) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
                } else {
                    // Fallback unfused
                    [enc setComputePipelineState:g_pipeline_qwen35_split_q_gate];
                    [enc setBuffer:g_buf_q offset:0 atIndex:0];
                    [enc setBuffer:g_buf_qwen35_attn_gate offset:0 atIndex:1];
                    [enc setBuffer:g_buf_q_raw offset:0 atIndex:2];
                    [enc setBytes:&qwen35_attn_dim length:sizeof(uint32_t) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((qwen35_attn_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:g_pipeline_qwen35_norm_rope];
                    [enc setBuffer:g_buf_q offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)lw->q_norm offset:0 atIndex:1];
                    [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:2];
                    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&rope_dim length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&pos length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&rope_theta length:sizeof(float) atIndex:6];
                    [enc setBytes:&norm_eps length:sizeof(float) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(num_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

                    [enc setComputePipelineState:g_pipeline_qwen35_norm_rope];
                    [enc setBuffer:g_buf_k offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)lw->k_norm offset:0 atIndex:1];
                    [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:2];
                    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&rope_dim length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&pos length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&rope_theta length:sizeof(float) atIndex:6];
                    [enc setBytes:&norm_eps length:sizeof(float) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
                }

                // kv_write and FlashAttention need k, v, q
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // 5. KVWrite
                [enc setComputePipelineState:g_pipeline_kv_write];
                [enc setBuffer:g_k_cache offset:0 atIndex:0];
                [enc setBuffer:g_v_cache offset:0 atIndex:1];
                [enc setBuffer:g_buf_k offset:0 atIndex:2];
                [enc setBuffer:g_buf_v offset:0 atIndex:3];
                [enc setBytes:&l length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&slot length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&max_seq length:sizeof(uint32_t) atIndex:6];
                [enc setBytes:&kv_dim length:sizeof(uint32_t) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                // FlashAttention needs k_cache written
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // 6. FlashAttention (GQA)
                size_t layer_kv_offset = (size_t)l * max_seq * kv_dim * sizeof(float);
                [enc setComputePipelineState:g_pipeline_attn];
                [enc setBuffer:g_buf_attn_out offset:0 atIndex:0];
                [enc setBuffer:g_buf_q offset:0 atIndex:1];
                [enc setBuffer:g_k_cache offset:layer_kv_offset atIndex:2];
                [enc setBuffer:g_v_cache offset:layer_kv_offset atIndex:3];
                [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];
                [enc setBytes:&active_context length:sizeof(uint32_t) atIndex:7];
                [enc setBytes:&attn_scale length:sizeof(float) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(num_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                // attn_gate needs attn_out
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // 7. Post-attention output gating: attn_out *= sigmoid(gate)
                [enc setComputePipelineState:g_pipeline_qwen35_attn_gate];
                [enc setBuffer:g_buf_attn_out offset:0 atIndex:0];
                [enc setBuffer:g_buf_qwen35_attn_gate offset:0 atIndex:1];
                [enc setBytes:&qwen35_attn_dim length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((qwen35_attn_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                // wo needs gated attn_out
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // 8. Output GEMV
                encode_gemv_buf(enc, get_pipeline(lw->wo_type), g_buf_attn_proj, g_buf_attn_out, (__bridge id<MTLBuffer>)lw->wo, dim, ssm_inner);
            }

            // residual_rmsnorm needs attn_proj
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Residual add (x += attn_proj) + FFN RMSNorm (x -> xb)
            [enc setComputePipelineState:g_pipeline_residual_rmsnorm];
            [enc setBuffer:g_buf_x offset:0 atIndex:0];
            [enc setBuffer:g_buf_attn_proj offset:0 atIndex:1];
            [enc setBuffer:g_buf_xb offset:0 atIndex:2];
            [enc setBuffer:buf_ffn_norm offset:0 atIndex:3];
            [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&norm_eps length:sizeof(float) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // FFN needs new xb
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // FFN Gate-Up + SwiGLU
            id<MTLComputePipelineState> fused_gate_up = get_fused_gate_up_pipeline(lw->ffn_gate_type);
            if (fused_gate_up && lw->ffn_gate_type == lw->ffn_up_type) {
                [enc setComputePipelineState:fused_gate_up];
                [enc setBuffer:g_buf_gate offset:0 atIndex:0];
                [enc setBuffer:g_buf_xb offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ffn_gate offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)lw->ffn_up offset:0 atIndex:3];
                [enc setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&dim length:sizeof(uint32_t) atIndex:5];
                MTLSize tgs, tpg;
                if (lw->ffn_gate_type == 12) {
                    tgs = MTLSizeMake((hidden_dim + 7) / 8, 1, 1);
                    tpg = MTLSizeMake(64, 1, 1);
                } else {
                    tgs = MTLSizeMake((hidden_dim + 7) / 8, 1, 1);
                    tpg = MTLSizeMake(128, 1, 1);
                }
                [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];
            } else {
                encode_gemv_buf(enc, get_pipeline(lw->ffn_gate_type), g_buf_gate, g_buf_xb, (__bridge id<MTLBuffer>)lw->ffn_gate, hidden_dim, dim);
                encode_gemv_buf(enc, get_pipeline(lw->ffn_up_type), g_buf_up, g_buf_xb, (__bridge id<MTLBuffer>)lw->ffn_up, hidden_dim, dim);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:g_pipeline_swiglu];
                [enc setBuffer:g_buf_gate offset:0 atIndex:0];
                [enc setBuffer:g_buf_up offset:0 atIndex:1];
                [enc setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((hidden_dim + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            // ffn_down needs g_buf_gate
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // FFN Down GEMV
            encode_gemv_buf(enc, get_pipeline(lw->ffn_down_type), g_buf_down, g_buf_gate, (__bridge id<MTLBuffer>)lw->ffn_down, dim, hidden_dim);

            // residual_rmsnorm needs g_buf_down
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Residual add (x += ffn_down) + Next Norm (x -> xb)
            id<MTLBuffer> next_norm = (l + 1 < num_layers) ? (__bridge id<MTLBuffer>)layers[l + 1].attn_norm : buf_out_norm;
            [enc setComputePipelineState:g_pipeline_residual_rmsnorm];
            [enc setBuffer:g_buf_x offset:0 atIndex:0];
            [enc setBuffer:g_buf_down offset:0 atIndex:1];
            [enc setBuffer:g_buf_xb offset:0 atIndex:2];
            [enc setBuffer:next_norm offset:0 atIndex:3];
            [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&norm_eps length:sizeof(float) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // Next layer (or output GEMV) needs new xb
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // Final Output Logits GEMV
        encode_gemv_buf(enc, get_pipeline(output_weight_type), g_buf_logits, g_buf_xb, (__bridge id<MTLBuffer>)output_weight_buf, vocab_size, dim);

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        if (out_token && g_pipeline_sample_argmax && g_buf_token) {
            [enc setComputePipelineState:g_pipeline_sample_argmax];
            [enc setBuffer:g_buf_logits offset:0 atIndex:0];
            [enc setBuffer:g_buf_token offset:0 atIndex:1];
            [enc setBytes:&vocab_size length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            printf("Metal Qwen35 command buffer failed! status=%lu, error=%s\n",
                   (unsigned long)[cmd status],
                   [cmd error] ? [[[cmd error] localizedDescription] UTF8String] : "none");
            fflush(stdout);
            return -3;
        }

        if (g_buf_logits && out_logits) {
            memcpy(out_logits, g_buf_logits.contents, vocab_size * sizeof(float));
        }
        if (out_token && g_buf_token) {
            *out_token = *(uint32_t*)g_buf_token.contents;
        }
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
        size_t layer_kv_offset = (size_t)layer_idx * max_seq * kv_dim * sizeof(float);
        [enc setComputePipelineState:g_pipeline_attn];
        [enc setBuffer:buf_attn_out offset:0 atIndex:0];
        [enc setBuffer:buf_q offset:0 atIndex:1];
        [enc setBuffer:g_k_cache offset:layer_kv_offset atIndex:2];
        [enc setBuffer:g_v_cache offset:layer_kv_offset atIndex:3];
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

int metal_conv1d_batch(float* out, const float* in, float* state, const float* conv_weight,
                       uint32_t kernel_size, uint32_t channels, uint32_t batch_size) {
    if (!metal_is_available() || !g_pipeline_conv1d_batch) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_out = [g_device newBufferWithBytesNoCopy:out length:batch_size * channels * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_in  = [g_device newBufferWithBytesNoCopy:(void*)in length:batch_size * channels * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_st  = state ? [g_device newBufferWithBytesNoCopy:state length:(kernel_size - 1) * channels * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil] : nil;
        id<MTLBuffer> buf_w   = [g_device newBufferWithBytesNoCopy:(void*)conv_weight length:channels * kernel_size * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_out || !buf_in || !buf_w) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_conv1d_batch];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_in offset:0 atIndex:1];
        [enc setBuffer:buf_st offset:0 atIndex:2];
        [enc setBuffer:buf_w offset:0 atIndex:3];
        [enc setBytes:&kernel_size length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&channels length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:6];

        MTLSize tgs = MTLSizeMake((channels + 63) / 64, 1, 1);
        MTLSize tpg = MTLSizeMake(64, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_ssm_recurrence_batch(
    float* ssm_out, const float* conv_out, const float* ssm_alpha, const float* ssm_beta,
    const float* dt_bias, const float* ssm_a, const float* ssm_norm_w, const float* ssm_gate,
    float* ssm_state, uint32_t ssm_inner, uint32_t ssm_state_size, uint32_t ssm_groups,
    uint32_t ssm_rank, float eps, uint32_t batch_size, uint32_t ssm_channels) {
    if (!metal_is_available() || !g_pipeline_ssm_batch) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_out   = [g_device newBufferWithBytesNoCopy:ssm_out length:batch_size * ssm_inner * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_conv  = [g_device newBufferWithBytesNoCopy:(void*)conv_out length:batch_size * ssm_channels * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_alpha = [g_device newBufferWithBytesNoCopy:(void*)ssm_alpha length:batch_size * ssm_rank * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_beta  = ssm_beta ? [g_device newBufferWithBytesNoCopy:(void*)ssm_beta length:batch_size * ssm_rank * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil] : nil;
        id<MTLBuffer> buf_dt    = dt_bias ? [g_device newBufferWithBytesNoCopy:(void*)dt_bias length:ssm_rank * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil] : nil;
        id<MTLBuffer> buf_a     = ssm_a ? [g_device newBufferWithBytesNoCopy:(void*)ssm_a length:ssm_rank * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil] : nil;
        id<MTLBuffer> buf_norm  = ssm_norm_w ? [g_device newBufferWithBytesNoCopy:(void*)ssm_norm_w length:ssm_state_size * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil] : nil;
        id<MTLBuffer> buf_gate  = [g_device newBufferWithBytesNoCopy:(void*)ssm_gate length:batch_size * ssm_inner * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_st    = [g_device newBufferWithBytesNoCopy:ssm_state length:ssm_rank * ssm_state_size * ssm_state_size * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_out || !buf_conv || !buf_alpha || !buf_gate || !buf_st) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_ssm_batch];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_conv offset:0 atIndex:1];
        [enc setBuffer:buf_alpha offset:0 atIndex:2];
        [enc setBuffer:buf_beta offset:0 atIndex:3];
        [enc setBuffer:buf_dt offset:0 atIndex:4];
        [enc setBuffer:buf_a offset:0 atIndex:5];
        [enc setBuffer:buf_norm offset:0 atIndex:6];
        [enc setBuffer:buf_gate offset:0 atIndex:7];
        [enc setBuffer:buf_st offset:0 atIndex:8];
        [enc setBytes:&ssm_inner length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&ssm_state_size length:sizeof(uint32_t) atIndex:10];
        [enc setBytes:&ssm_groups length:sizeof(uint32_t) atIndex:11];
        [enc setBytes:&ssm_rank length:sizeof(uint32_t) atIndex:12];
        [enc setBytes:&eps length:sizeof(float) atIndex:13];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:14];
        [enc setBytes:&ssm_channels length:sizeof(uint32_t) atIndex:15];

        MTLSize tgs = MTLSizeMake(ssm_rank, 1, 1);
        MTLSize tpg = MTLSizeMake(ssm_state_size, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_attention_gqa_batch(
    float* attn_out, const float* q, const float* k_cache, const float* v_cache,
    uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t start_pos, uint32_t max_seq, float attn_scale, uint32_t batch_size) {
    if (!metal_is_available() || !g_pipeline_attention_gqa_batch) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_out = [g_device newBufferWithBytesNoCopy:attn_out length:batch_size * num_heads * head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_q   = [g_device newBufferWithBytesNoCopy:(void*)q length:batch_size * num_heads * head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_kc  = [g_device newBufferWithBytesNoCopy:(void*)k_cache length:max_seq * num_kv_heads * head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_vc  = [g_device newBufferWithBytesNoCopy:(void*)v_cache length:max_seq * num_kv_heads * head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_out || !buf_q || !buf_kc || !buf_vc) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_attention_gqa_batch];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_q offset:0 atIndex:1];
        [enc setBuffer:buf_kc offset:0 atIndex:2];
        [enc setBuffer:buf_vc offset:0 atIndex:3];
        [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&start_pos length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&max_seq length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&attn_scale length:sizeof(float) atIndex:9];

        MTLSize tgs = MTLSizeMake(num_heads, batch_size, 1);
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

int metal_kv_write_batch(
    float* k_cache, float* v_cache, const float* k, const float* v,
    uint32_t start_pos, uint32_t max_seq, uint32_t kv_dim, uint32_t batch_size) {
    if (!metal_is_available() || !g_pipeline_kv_write_batch) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_kc = [g_device newBufferWithBytesNoCopy:k_cache length:max_seq * kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_vc = [g_device newBufferWithBytesNoCopy:v_cache length:max_seq * kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_k  = [g_device newBufferWithBytesNoCopy:(void*)k length:batch_size * kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_v  = [g_device newBufferWithBytesNoCopy:(void*)v length:batch_size * kv_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_kc || !buf_vc || !buf_k || !buf_v) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_kv_write_batch];
        [enc setBuffer:buf_kc offset:0 atIndex:0];
        [enc setBuffer:buf_vc offset:0 atIndex:1];
        [enc setBuffer:buf_k offset:0 atIndex:2];
        [enc setBuffer:buf_v offset:0 atIndex:3];
        [enc setBytes:&start_pos length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&max_seq length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&kv_dim length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:7];

        uint32_t total_elems = batch_size * kv_dim;
        MTLSize tgs = MTLSizeMake((total_elems + 31) / 32, 1, 1);
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

int metal_qwen35_split_q_gate_batch(
    float* q_out, float* gate_out, const float* q_gate_in, uint32_t num_tokens) {
    if (!metal_is_available() || !g_pipeline_split_q_gate_batch) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_q    = [g_device newBufferWithBytesNoCopy:q_out length:num_tokens * 6144 * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_gate = [g_device newBufferWithBytesNoCopy:gate_out length:num_tokens * 6144 * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_in   = [g_device newBufferWithBytesNoCopy:(void*)q_gate_in length:num_tokens * 12288 * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_q || !buf_gate || !buf_in) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_split_q_gate_batch];
        [enc setBuffer:buf_q offset:0 atIndex:0];
        [enc setBuffer:buf_gate offset:0 atIndex:1];
        [enc setBuffer:buf_in offset:0 atIndex:2];
        [enc setBytes:&num_tokens length:sizeof(uint32_t) atIndex:3];

        uint32_t total_elems = num_tokens * 6144;
        MTLSize tgs = MTLSizeMake((total_elems + 255) / 256, 1, 1);
        MTLSize tpg = MTLSizeMake(256, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

int metal_rope_norm_batch(
    float* q, float* k, const float* q_norm_w, const float* k_norm_w,
    uint32_t start_pos, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t rope_dim, float theta, float eps, uint32_t batch_size) {
    if (!metal_is_available() || !g_pipeline_rope_norm_batch) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_q  = [g_device newBufferWithBytesNoCopy:q length:batch_size * num_heads * head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_k  = [g_device newBufferWithBytesNoCopy:k length:batch_size * num_kv_heads * head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_qw = [g_device newBufferWithBytesNoCopy:(void*)q_norm_w length:head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_kw = [g_device newBufferWithBytesNoCopy:(void*)k_norm_w length:head_dim * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_q || !buf_k || !buf_qw || !buf_kw) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_rope_norm_batch];
        [enc setBuffer:buf_q offset:0 atIndex:0];
        [enc setBuffer:buf_k offset:0 atIndex:1];
        [enc setBuffer:buf_qw offset:0 atIndex:2];
        [enc setBuffer:buf_kw offset:0 atIndex:3];
        [enc setBytes:&start_pos length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&rope_dim length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&theta length:sizeof(float) atIndex:9];
        [enc setBytes:&eps length:sizeof(float) atIndex:10];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:11];

        uint32_t max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
        MTLSize tgs = MTLSizeMake(max_heads, batch_size, 1);
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

int metal_qwen35_attn_gate(float* attn_out, const float* gate, uint32_t total_elements) {
    if (!metal_is_available() || !g_pipeline_qwen35_attn_gate) return -1;

    @autoreleasepool {
        id<MTLBuffer> buf_out  = [g_device newBufferWithBytesNoCopy:attn_out length:total_elements * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_gate = [g_device newBufferWithBytesNoCopy:(void*)gate length:total_elements * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

        if (!buf_out || !buf_gate) return -2;

        bool is_batched = (g_batch_encoder != nil);
        id<MTLCommandBuffer> cmd = is_batched ? g_batch_cmd : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = is_batched ? g_batch_encoder : [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline_qwen35_attn_gate];
        [enc setBuffer:buf_out offset:0 atIndex:0];
        [enc setBuffer:buf_gate offset:0 atIndex:1];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:2];

        MTLSize tgs = MTLSizeMake((total_elements + 255) / 256, 1, 1);
        MTLSize tpg = MTLSizeMake(256, 1, 1);
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];

        if (!is_batched) {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 0;
}

