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

// 128-Thread 8-Row Vectorized Q4_K GEMV
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

    threadgroup float tg_sums[4][8];

    uint num_blocks = cols / 256;
    device const block_q4_k* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 128) {
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

// 128-Thread 8-Row Vectorized Q6_K GEMV
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

    threadgroup float tg_sums[4][8];

    uint num_blocks = cols / 256;
    device const block_q6_k* r_blocks[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint r = r0 + i;
        r_blocks[i] = (r < rows) ? (w + r * num_blocks) : (w + r0 * num_blocks);
    }

    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = tid; b < num_blocks; b += 128) {
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
    uint r0 = tg_idx * 8;
    if (r0 >= hidden_dim) return;

    threadgroup float tg_gate[4][8];
    threadgroup float tg_up[4][8];

    uint num_blocks = dim / 256;
    device const block_q4_k* gate_blocks[8];
    device const block_q4_k* up_blocks[8];
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
                if (r0 + i < hidden_dim) {
                    // Gate block
                    {
                        device const block_q4_k& blk = gate_blocks[i][b];
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
                            sums_g[i] += (float(byte_val & 0x0F) * sc - m) * x_low[j] + (float((byte_val >> 4) & 0x0F) * sc - m) * x_high[j];
                        }
                    }

                    // Up block
                    {
                        device const block_q4_k& blk = up_blocks[i][b];
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
                            sums_u[i] += (float(byte_val & 0x0F) * sc - m) * x_low[j] + (float((byte_val >> 4) & 0x0F) * sc - m) * x_high[j];
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

        for (int sb = 0; sb < 16; sb++) {
            int sb_x_off = x_off + sb * 16;
            float x_vals[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                x_vals[j] = x[sb_x_off + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (r0 + i < hidden_dim) {
                    // Gate
                    {
                        device const block_q6_k& blk = gate_blocks[i][b];
                        float sc = float(blk.scales[sb]) * float(blk.d);
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            int idx = sb * 16 + j;
                            uint8_t l = blk.ql[idx / 2];
                            int q_val = (idx % 2 == 0) ? int(l & 0x0F) : int((l >> 4) & 0x0F);
                            uint8_t h = (blk.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                            q_val = (q_val | (int(h) << 4)) - 32;
                            sums_g[i] += (float(q_val) * sc) * x_vals[j];
                        }
                    }

                    // Up
                    {
                        device const block_q6_k& blk = up_blocks[i][b];
                        float sc = float(blk.scales[sb]) * float(blk.d);
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            int idx = sb * 16 + j;
                            uint8_t l = blk.ql[idx / 2];
                            int q_val = (idx % 2 == 0) ? int(l & 0x0F) : int((l >> 4) & 0x0F);
                            uint8_t h = (blk.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                            q_val = (q_val | (int(h) << 4)) - 32;
                            sums_u[i] += (float(q_val) * sc) * x_vals[j];
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
    uint tid                         [[thread_position_in_grid]],
    uint t_idx                       [[thread_index_in_threadgroup]],
    uint tg_idx                      [[threadgroup_position_in_grid]]
) {
    threadgroup float tg_max[128];
    threadgroup uint  tg_idx_arr[128];

    float local_max = -INFINITY;
    uint  local_idx = 0;

    for (uint i = tid; i < vocab_size; i += 1024) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    tg_max[t_idx] = local_max;
    tg_idx_arr[t_idx] = local_idx;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 64; s > 0; s >>= 1) {
        if (t_idx < s) {
            if (tg_max[t_idx + s] > tg_max[t_idx]) {
                tg_max[t_idx] = tg_max[t_idx + s];
                tg_idx_arr[t_idx] = tg_idx_arr[t_idx + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t_idx == 0) {
        result_token[tg_idx] = tg_idx_arr[0];
    }
}
