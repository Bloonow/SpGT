#pragma once
#include <cuda.h>
#include "../utils/buffer.cu"

namespace position {
__global__ void add_pos2d_rc_256x2_1x2(
    const float *pos, float *head, const int M, const int n_head
) {
    // 线程块与线程的标识
    const int brid = blockIdx.y;
    const int bcid = blockIdx.x;
    const int tid = threadIdx.x;
    if (brid * 256 + tid < M) {
        float2 pos_reg = *reinterpret_cast<const float2*>(pos + blockIdx.z * 2 * M + brid * 2 * 256 + tid * 2);
        *reinterpret_cast<float*>(
            head + blockIdx.z * (2 + 32) * n_head * M + bcid * (2 + 32) * M + brid * 256 + tid + 0 * M
        ) = pos_reg.x;
        *reinterpret_cast<float*>(
            head + blockIdx.z * (2 + 32) * n_head * M + bcid * (2 + 32) * M + brid * 256 + tid + 1 * M
        ) = pos_reg.y;
    }
}
} // namespace position

namespace lnorm {
__device__ __forceinline__
void store_invsigma_smem_cc_128x128_8x8(
    const float invsigma_reg[2][4], float *smem_buf, float *invsigma,
    const int &M, const int &N, const int &muS, 
    const int &brid, const int &bcid, const int &tid, const int &lid,
    const int &wrid, const int &wcid, const int &lcid
) {
    // 每个 mh32 注意力头对应一列 invsigma 值
    float *invsigma_block = invsigma + (blockIdx.z * muS + bcid * 4 * M + brid * 128);
    // 一个 128x128 的线程块中有 4 个 mh32 注意力头
    float *invsigma_smem = reinterpret_cast<float*>(smem_buf);

    // 写回 invsigma 的子区域，使用 128x4 共享内存搬运 128x4 数据，共需 1 次，每次每个线程写回 2x1 数据
    float2 inv_trans;
    inv_trans.x = invsigma_reg[lcid / 2][(lcid % 2) * 2 + 0];
    inv_trans.y = invsigma_reg[lcid / 2][(lcid % 2) * 2 + 1];

    // 将数据写入到共享内存，以防其他线程仍在读写共享内存，进行同步
    __syncthreads();
    *reinterpret_cast<float2*>(invsigma_smem + wcid * 128 + wrid * 64 + lid * 2) = inv_trans;

    // 将数据从共享内存转移到全局内存，使用 128x2 排列的线程搬运 128x4 (x2) 共享内存，共需 2 次，每次每个线程写回 1 (x2) 个数据
    __syncthreads();
    #pragma unroll
    for (int gmem_column = 0; gmem_column < 4; gmem_column += 2) {
        if ((brid * 128 + tid % 128 < M) && (bcid * 4 + gmem_column + tid / 128 < (N / 32))) {
            *reinterpret_cast<float*>(
                invsigma_block + (gmem_column + tid / 128) * M + (tid % 128)
            ) = *reinterpret_cast<float*>(invsigma_smem + gmem_column * 128 + tid);
        }
    }
}

__device__ __forceinline__
void lnorm_rrrc_128x128_8x8(
    float Creg[2][2][4][4], float *smem_buf,
    const float *lnw, const float *lnb, float *invsigma, const float &norm_eps,
    const int &M, const int &N, const int &muS, 
    const int &brid, const int &bcid, const int &tid, const int &lid,
    const int &wcols, const int &wrid, const int &wcid, const int &lcid
) {
    // 将层归一化参数 lnw, lnb 读入共享内存
    float *lnw_smem = reinterpret_cast<float*>(smem_buf);
    float *lnb_smem = reinterpret_cast<float*>(smem_buf + 128);
    // 为防止仍有线程仍在读写共享内存，进行同步
    __syncthreads();
    if (tid < 128) {
        lnw_smem[tid] = *reinterpret_cast<const float*>(lnw + bcid * 128 + tid);
        lnb_smem[tid] = *reinterpret_cast<const float*>(lnb + bcid * 128 + tid);
    }

    // 计算每行元素的 mu 与 invsigma 值
    float mu_reg[2][4] = {}, inv_reg[2][4] = {};
    #pragma unroll
    for (int cpi = 0; cpi < 2; ++cpi) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            // 沿着 cpj 展开，是线程所负责的该行的 8 个元素
            float mu_val = (Creg[cpi][0][row][0] + Creg[cpi][0][row][1] + Creg[cpi][0][row][2] + Creg[cpi][0][row][3] +
                            Creg[cpi][1][row][0] + Creg[cpi][1][row][1] + Creg[cpi][1][row][2] + Creg[cpi][1][row][3]) * 0.125F;
            float sigma2_val = ((Creg[cpi][0][row][0] - mu_val) * (Creg[cpi][0][row][0] - mu_val) + 
                                (Creg[cpi][0][row][1] - mu_val) * (Creg[cpi][0][row][1] - mu_val) + 
                                (Creg[cpi][0][row][2] - mu_val) * (Creg[cpi][0][row][2] - mu_val) + 
                                (Creg[cpi][0][row][3] - mu_val) * (Creg[cpi][0][row][3] - mu_val) +
                                (Creg[cpi][1][row][0] - mu_val) * (Creg[cpi][1][row][0] - mu_val) + 
                                (Creg[cpi][1][row][1] - mu_val) * (Creg[cpi][1][row][1] - mu_val) + 
                                (Creg[cpi][1][row][2] - mu_val) * (Creg[cpi][1][row][2] - mu_val) + 
                                (Creg[cpi][1][row][3] - mu_val) * (Creg[cpi][1][row][3] - mu_val));
            float count_val = 8.f;
            // 此时每个线程已获得自己所处理 row 行数据的局部的 TILE 个元素的 mu 和 sigma2，接下来每行 4 个线程间进行归约
            for (int up_delta = 16; up_delta >= 1; up_delta /= 16) {
                // 对于一个线程束 Warp 中的线程排列有假设
                float mu_tmp     = __shfl_down_sync(0xffffffff, mu_val,     up_delta, warpSize);
                float sigma2_tmp = __shfl_down_sync(0xffffffff, sigma2_val, up_delta, warpSize);
                float count_tmp  = __shfl_down_sync(0xffffffff, count_val,  up_delta, warpSize);
                // welford
                float delta = mu_tmp - mu_val;
                float total = count_tmp + count_val;
                float inv_total = 1.f / total;
                // if (total > 0) {
                //     // 此处肯定是 total > 0
                count_val = count_val * inv_total;
                count_tmp = count_tmp * inv_total;
                mu_val = count_val * mu_val + count_tmp * mu_tmp;
                sigma2_val = sigma2_val + sigma2_tmp + delta * delta * count_val * count_tmp * total;
                // } else {
                //     mu_val = 0.F;
                //     sigma2_val = 0.F;
                // }
                count_val = total;
            }
            // 此时，每个线程束每行 4 个线程的最左侧线程获得了该行的最终结果，现将之广播给该行的其他线程
            mu_val     = __shfl_sync(0xffffffff, mu_val,     (tid % 16) / 2 * 2, warpSize);
            sigma2_val = __shfl_sync(0xffffffff, sigma2_val, (tid % 16) / 2 * 2, warpSize);
            // 每个线程获得该行最终的 mu 和 sigma2 值
            mu_reg[cpi][row] = mu_val;
            inv_reg[cpi][row] = rsqrt(sigma2_val * (1.F / 32) + norm_eps);
        }
    }

    // 应用 lnw, lnb 层归一化的参数
    __syncthreads();
    #pragma unroll
    for (int cpj = 0; cpj < 2; ++cpj) {
        float4 lnw_reg = *reinterpret_cast<float4*>(lnw_smem + wcid * wcols * 8 + lcid * 4 + cpj * wcols * 4);
        float4 lnb_reg = *reinterpret_cast<float4*>(lnb_smem + wcid * wcols * 8 + lcid * 4 + cpj * wcols * 4);
        #pragma unroll
        for (int cpi = 0; cpi < 2; ++cpi) {
            #pragma unroll
            for (int row = 0; row < 4; ++row) {
                Creg[cpi][cpj][row][0] = (Creg[cpi][cpj][row][0] - mu_reg[cpi][row]) * inv_reg[cpi][row] * lnw_reg.x + lnb_reg.x;
                Creg[cpi][cpj][row][1] = (Creg[cpi][cpj][row][1] - mu_reg[cpi][row]) * inv_reg[cpi][row] * lnw_reg.y + lnb_reg.y;
                Creg[cpi][cpj][row][2] = (Creg[cpi][cpj][row][2] - mu_reg[cpi][row]) * inv_reg[cpi][row] * lnw_reg.z + lnb_reg.z;
                Creg[cpi][cpj][row][3] = (Creg[cpi][cpj][row][3] - mu_reg[cpi][row]) * inv_reg[cpi][row] * lnw_reg.w + lnb_reg.w;
            }
        }
    }
    
    // 将每行元素的 invsigma 值写回到全局内存，使用共享内存搬运
    store_invsigma_smem_cc_128x128_8x8(inv_reg, smem_buf, invsigma, M, N, muS, brid, bcid, tid, lid, wrid, wcid, lcid);
}

__global__ void lnorm_grad_input_ccc_128x32_1x32(
    const float *grad_ln, const float *hat_ln, const float *invsigma, const float *lnw, const float *lnb, float *grad_intput,
    const int M, const int hatS, const int muS, const int woPosS, const int d_pos
) {
    // 注意，原梯度 grad_ln 中包含位置编码的空间，而所求梯度 grad_input 中去掉位置编码的空间
    const int brid = blockIdx.y;
    const int bcid = blockIdx.x;
    const int tid = threadIdx.x;

    float stats_x1 = 0.F, stats_x2 = 0.F;
    float grad_ln_val = 0.F, hat_val = 0.F, invsigma_val = 0.F;
    float lnw_val = 1.F, lnb_val = 0.F;

    // 共享内存，至少 32 (x2) 个 float 的空间，用于存放层归一化的 lnw, lnb 参数
    float *lnw_smem = buffer::SharedMemory<float, 64>().pointer();
    float *lnb_smem = reinterpret_cast<float*>(lnw_smem + 32);
    if (tid < 32) {
        *reinterpret_cast<float*>(lnw_smem + tid) = *reinterpret_cast<const float*>(lnw + bcid * 32 + tid);
        *reinterpret_cast<float*>(lnb_smem + tid) = *reinterpret_cast<const float*>(lnb + bcid * 32 + tid);
    }
    if (brid * 128 + tid < M) {
        invsigma_val = *reinterpret_cast<const float*>(invsigma + blockIdx.z * muS + bcid * M + brid * 128 + tid);
    }
    __syncthreads();

    // 计算该线程负责的 hat 的含有位置编码的偏移，以及 input 不含有位置编码的偏移
    const int hat_offset = blockIdx.z * hatS + bcid * (d_pos + 32) * M + d_pos * M + brid * 128 + tid;
    const int input_offset = blockIdx.z * woPosS + bcid * 32 * M + brid * 128 + tid;
    
    // 每个线程负责处理该行的 1x32 个数据
    #pragma unroll
    for (int column = 0; column < 32; ++column) {
        lnw_val = lnw_smem[column];
        lnb_val = lnb_smem[column];
        if (brid * 128 + tid < M) {
            grad_ln_val = *reinterpret_cast<const float*>(grad_ln + hat_offset + column * M);
            // 此处并不是纯粹的 hat = (x - mu) * invsigma
            // 而是层归一化之后的结果 hat_ln = (x - mu) * invsigma * lnw + lnb
            hat_val = *reinterpret_cast<const float*>(hat_ln + hat_offset + column * M);
            hat_val = (hat_val - lnb_val) * (1.F / lnw_val);
        }
        stats_x1 += grad_ln_val * lnw_val;
        stats_x2 += grad_ln_val * lnw_val * hat_val;
    }
    // 计算得到该行的 stats_x1 与 stats_x2 值
    float term = (1.F / 32) * invsigma_val;
    #pragma unroll
    for (int column = 0; column < 32; ++column) {
        lnw_val = lnw_smem[column];
        lnb_val = lnb_smem[column];
        if (brid * 128 + tid < M) {
            grad_ln_val = *reinterpret_cast<const float*>(grad_ln + hat_offset + column * M);
            hat_val = *reinterpret_cast<const float*>(hat_ln + hat_offset + column * M);
            hat_val = (hat_val - lnb_val) * (1.F / lnw_val);
        }
        float grad_input_val = 32.F * lnw_val * grad_ln_val;
        grad_input_val -= hat_val * stats_x2;
        grad_input_val -= stats_x1;
        grad_input_val *= term;
        if (brid * 128 + tid < M) {
            *reinterpret_cast<float*>(grad_intput + input_offset + column * M) = grad_input_val;
        }
    }
}

__global__ void lnorm_grad_weight_bias_ccrr_loop256x1_1x1(
    const float *grad_ln, const float *hat_ln, const float *lnw, const float *lnb, float *grad_lnw, float *grad_lnb,
    const int M, const int hatS, const int d_pos, const int batchCount
) {
    // 一个线程块 256 个线程，负责一列全部 loop256x1 数据，最后做线程块内归约
    const int bcid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;

    // 从全局内存中加载该列数据的 lnw 与 lnb，以用于从 hat_ln_val 计算真正的 hat_val 值
    float lnw_val = lnw[bcid];
    float lnb_val = lnb[bcid];
    
    float grad_ln_val = 0.F, hat_val = 0.F;
    float grad_lnw_val = 0.F, grad_lnb_val = 0.F;

    // 共享内存至少 8 (x2) 个 float 的空间
    float *grad_lnw_smem = buffer::SharedMemory<float, 16>().pointer();
    float *grad_lnb_smem = reinterpret_cast<float*>(grad_lnw_smem + 8);

    // 每个线程块负责一列的全部数据
    for (int batch_index = 0; batch_index < batchCount; ++batch_index) {
        // 计算该线程负责的 hat 的含有位置编码的偏移
        int hat_offset = batch_index * hatS + bcid * M + (bcid / 32 + 1) * d_pos * M;
        #pragma unroll
        for (int row = tid; row < M; row += 256) {
            grad_ln_val = *reinterpret_cast<const float*>(grad_ln + hat_offset + row);
            hat_val = *reinterpret_cast<const float*>(hat_ln + hat_offset + row);
            hat_val = (hat_val - lnb_val) * (1.F / lnw_val);
            grad_lnw_val += grad_ln_val * hat_val;
            grad_lnb_val += grad_ln_val;
        }

    }
    // 一个线程束 Warp 内将 32 个局部值进行归约
    for (int lane_delta = 16; lane_delta >= 1; lane_delta /= 2) {
        grad_lnw_val += __shfl_down_sync(0xffffffff, grad_lnw_val, lane_delta, warpSize);
        grad_lnb_val += __shfl_down_sync(0xffffffff, grad_lnb_val, lane_delta, warpSize);
    }
    // 一个线程块的所有线程束 Warp 之间进行归约
    if (lid == 0) {
        grad_lnw_smem[wid] = grad_lnw_val;
        grad_lnb_smem[wid] = grad_lnb_val;
    }
    __syncthreads();
    // 由线程块内的 tid == 0 线程将该列的最终值写回
    if (tid == 0) {
        grad_lnw_val = 0.F;
        grad_lnb_val = 0.F;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            grad_lnw_val += grad_lnw_smem[i];
            grad_lnb_val += grad_lnb_smem[i];
        }
        grad_lnw[bcid] = grad_lnw_val;
        grad_lnb[bcid] = grad_lnb_val;
    }
}
} // namespace lnorm

namespace sgemm_extra_128x128_8x8 {
struct TileIndex {
    int brid, bcid, tid, wid, lid;
    int wrows, wcols, wrid, wcid, lrid, lcid;
    int M, N, K, aS, bS, cS;
    __device__ TileIndex(const int M, const int N, const int K, const int aS, const int bS, const int cS) {
        // 线程块与线程的标识
        brid = blockIdx.y; bcid = blockIdx.x; tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        wrid = wid / 4; wcid = wid % 4;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
        // 矩阵形状与跨步
        this->M = M; this->N = N; this->K = K;
        this->aS = aS; this->bS = bS; this->cS = cS;
    }
};

__device__ __forceinline__
void compute_tile_rrr(
    float Creg[2][2][4][4], float *Asmem, float *Bsmem,
    const int &wrows, const int &wcols, const int &wrid, const int &wcid, const int &lrid, const int &lcid
) {
    float4 Areg[2], Breg[2];
    // 每个线程计算 C 的子域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (int kid = 0; kid < 8; ++kid) {
        Areg[0] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 0 * wrows * 4 + lrid * 4 + kid * 132);
        Areg[1] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 1 * wrows * 4 + lrid * 4 + kid * 132);
        Breg[0] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 0 * wcols * 4 + lcid * 4 + kid * 128);
        Breg[1] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 1 * wcols * 4 + lcid * 4 + kid * 128);
        #pragma unroll
        for (int cpi = 0; cpi < 2; ++cpi) {
            #pragma unroll
            for (int cpj = 0; cpj < 2; ++cpj) {
                Creg[cpi][cpj][0][0] += Areg[cpi].x * Breg[cpj].x;
                Creg[cpi][cpj][0][1] += Areg[cpi].x * Breg[cpj].y;
                Creg[cpi][cpj][0][2] += Areg[cpi].x * Breg[cpj].z;
                Creg[cpi][cpj][0][3] += Areg[cpi].x * Breg[cpj].w;
                Creg[cpi][cpj][1][0] += Areg[cpi].y * Breg[cpj].x;
                Creg[cpi][cpj][1][1] += Areg[cpi].y * Breg[cpj].y;
                Creg[cpi][cpj][1][2] += Areg[cpi].y * Breg[cpj].z;
                Creg[cpi][cpj][1][3] += Areg[cpi].y * Breg[cpj].w;
                Creg[cpi][cpj][2][0] += Areg[cpi].z * Breg[cpj].x;
                Creg[cpi][cpj][2][1] += Areg[cpi].z * Breg[cpj].y;
                Creg[cpi][cpj][2][2] += Areg[cpi].z * Breg[cpj].z;
                Creg[cpi][cpj][2][3] += Areg[cpi].z * Breg[cpj].w;
                Creg[cpi][cpj][3][0] += Areg[cpi].w * Breg[cpj].x;
                Creg[cpi][cpj][3][1] += Areg[cpi].w * Breg[cpj].y;
                Creg[cpi][cpj][3][2] += Areg[cpi].w * Breg[cpj].z;
                Creg[cpi][cpj][3][3] += Areg[cpi].w * Breg[cpj].w;
            }
        }
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[2][2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha,
    const int &M, const int &N, const int &K, const int &aS, const int &bS, const int &cS,
    const int &brid, const int &bcid, const int &tid, const int &wid, const int &lid,
    const int &wrows, const int &wcols, const int &wrid, const int &wcid, const int &lrid, const int &lcid
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 1024 * 4);
    
    // [NEXT] A_tid + eid * K + kth * 8
    // [NEXT] B_tid + eid * 32  + kth * 8 * N
    const float *A_tid = A + (blockIdx.z * aS + brid * 128 * K) + (tid / 8 * 4 * K + tid % 8);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 128) + (wid * N + lid);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (brid * 128 + tid / 8 * 4 + eid < M) A_valid |= (1u << eid);
        if (bcid * 128 + lid + eid * 32 < N)    B_valid |= (1u << eid);
    }

    int kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (tid % 8 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * K);
        }
        if ((B_valid & (1u << eid)) && (wid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * 32);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 128 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + tid % 8 * 132 + tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
    Bsmem[wid * 128 + lid + 0 * 32] = Btrans[0];
    Bsmem[wid * 128 + lid + 1 * 32] = Btrans[1];
    Bsmem[wid * 128 + lid + 2 * 32] = Btrans[2];
    Bsmem[wid * 128 + lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_tid += kstart;
    B_tid += kstart * N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * 32);
            }
        }
        // 计算 C 的子区域
        compute_tile_rrr(Creg, Asmem, Bsmem, wrows, wcols, wrid, wcid, lrid, lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 2048;
        Bsmem += (2 * (kth & 1) - 1) * 1024;
        *reinterpret_cast<float4*>(Asmem + tid % 8 * 132 + tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
        Bsmem[wid * 128 + lid + 0 * 32] = Btrans[0];
        Bsmem[wid * 128 + lid + 1 * 32] = Btrans[1];
        Bsmem[wid * 128 + lid + 2 * 32] = Btrans[2];
        Bsmem[wid * 128 + lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_tid += 8;
        B_tid += 8 * N;
    }
    // 计算 C 的子区域
    compute_tile_rrr(Creg, Asmem, Bsmem, wrows, wcols, wrid, wcid, lrid, lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int cpi = 0; cpi < 2; ++cpi) {
        #pragma unroll
        for (int cpj = 0; cpj < 2; ++cpj) {
            #pragma unroll
            for (int row = 0; row < 4; ++row) {
                Creg[cpi][cpj][row][0] *= alpha;
                Creg[cpi][cpj][row][1] *= alpha;
                Creg[cpi][cpj][row][2] *= alpha;
                Creg[cpi][cpj][row][3] *= alpha;
            }
        }
    }
}

__device__ __forceinline__
void add_bias_rr(
    float Creg[2][2][4][4], float *smem_buf, const float *bias,
    const int &bcid, const int &tid,  const int &wcols, const int &wcid, const int &lcid
) {
    // 将偏差读入共享内存，为防止仍有线程仍在读写共享内存，进行同步
    __syncthreads();
    if (tid < 128) {
        smem_buf[tid] = *reinterpret_cast<const float*>(bias + bcid * 128 + tid);
    }
    __syncthreads();
    #pragma unroll
    for (int cpj = 0; cpj < 2; ++cpj) {
        float4 bias_reg = *reinterpret_cast<float4*>(smem_buf + wcid * wcols * 8 + lcid * 4 + cpj * wcols * 4);
        #pragma unroll
        for (int cpi = 0; cpi < 2; ++cpi) {
            #pragma unroll
            for (int row = 0; row < 4; ++row) {
                Creg[cpi][cpj][row][0] += bias_reg.x;
                Creg[cpi][cpj][row][1] += bias_reg.y;
                Creg[cpi][cpj][row][2] += bias_reg.z;
                Creg[cpi][cpj][row][3] += bias_reg.w;
            }
        }
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[2][2][4][4], float *smem_buf, float *C,
    const int &M, const int &N, const int &cS, const int &d_pos,
    const int &brid, const int &bcid, const int &tid,
    const int &wrows, const int &wcols, const int &wrid, const int &wcid, const int &lrid, const int &lcid
) {
    // 注意 d_pos > 0 时存在预留 position 位置编码的情况
    float4 trans1, trans2;
    // 写回矩阵 C 的子区域，使用 128x32 共享内存搬运 128x128 数据，共需 4 次，每次每个线程写回 8x2 数据 Creg[0:1][0:1][0:3][column]
    // 对应 [cpi, cpj] = [0:1, 0:1] 时为 4x1 形状的数据
    float *C_block = C + (blockIdx.z * cS + bcid * (128 + 4 * d_pos) * M + brid * 128);
    for (int column = 0; column < 4; ++column) {
        __syncthreads();
        // 将数据写入到共享内存，存在bank冲突，待改进
        #pragma unroll
        for (int cpj = 0; cpj < 2; ++cpj) {
            trans1.x = Creg[0][cpj][0][column]; trans1.y = Creg[0][cpj][1][column]; trans1.z = Creg[0][cpj][2][column]; trans1.w = Creg[0][cpj][3][column];
            trans2.x = Creg[1][cpj][0][column]; trans2.y = Creg[1][cpj][1][column]; trans2.z = Creg[1][cpj][2][column]; trans2.w = Creg[1][cpj][3][column];
            *reinterpret_cast<float4*>(
                smem_buf + (wcid * wcols * 2 * 128 + wrid * wrows * 8) + (cpj * wcols * 128 + 0 * wrows * 4) + (lcid * 128 + lrid * 4)
            ) = trans1;
            *reinterpret_cast<float4*>(
                smem_buf + (wcid * wcols * 2 * 128 + wrid * wrows * 8) + (cpj * wcols * 128 + 1 * wrows * 4) + (lcid * 128 + lrid * 4)
            ) = trans2;
        }
        __syncthreads();
        // 将数据从共享内存转移到全局内存
        // 使用 128x2 排列的线程搬运 128x32 共享内存，共需 16 次，每次每个线程写回 1 个数据
        #pragma unroll
        for (int gmem_column = column; gmem_column < 128; gmem_column += 8) {
            if ((brid * 128 + tid % 128 < M) && (bcid * 128 + gmem_column + tid / 128 * 4 < N)) {
                *reinterpret_cast<float*>(
                    C_block + (gmem_column + tid / 128 * 4) * M + (tid % 128) + (gmem_column / 32 + 1) * d_pos * M
                ) = *reinterpret_cast<float*>(smem_buf + gmem_column / 4 * 128 + tid);
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int d_pos
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 6>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_rrr(
        Creg, smem_buf, A, B, alpha,
        T.M, T.N, T.K, T.aS, T.bS, T.cS,
        T.brid, T.bcid, T.tid, T.wid, T.lid,
        T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C,
        T.M, T.N, T.cS, d_pos,
        T.brid, T.bcid, T.tid,
        T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_bias_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha, const float *bias,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int d_pos
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 6>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_rrr(
        Creg, smem_buf, A, B, alpha,
        T.M, T.N, T.K, T.aS, T.bS, T.cS,
        T.brid, T.bcid, T.tid, T.wid, T.lid,
        T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    add_bias_rr(Creg, smem_buf, bias, T.bcid, T.tid, T.wcols, T.wcid, T.lcid);
    store_result_smem_rc(
        Creg, smem_buf, C,
        T.M, T.N, T.cS, d_pos,
        T.brid, T.bcid, T.tid,
        T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_bias_lnorm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha, const float *bias,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const float *lnw, const float *lnb, float *invsigma, const float norm_eps, const int muS,
    const int d_pos
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 6>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_rrr(
        Creg, smem_buf, A, B, alpha,
        T.M, T.N, T.K, T.aS, T.bS, T.cS,
        T.brid, T.bcid, T.tid, T.wid, T.lid,
        T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    add_bias_rr(Creg, smem_buf, bias, T.bcid, T.tid, T.wcols, T.wcid, T.lcid);
    lnorm::lnorm_rrrc_128x128_8x8(
        Creg, smem_buf,
        lnw, lnb, invsigma, norm_eps,
        T.M, T.N, muS,
        T.brid, T.bcid, T.tid, T.lid,
        T.wcols, T.wrid, T.wcid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C,
        T.M, T.N, T.cS, d_pos,
        T.brid, T.bcid, T.tid,
        T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}
} // namespace sgemm_extra_128x128_8x8

namespace mh32 {
__host__ void sgemm_rrc_cuda(
    const float *A, const float *B, float *C, const float alpha, const float *bias,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const float *lnw, const float *lnb, float *invsigma, const float norm_eps, const int muS,
    const float *pos, const int d_pos,
    const int batchCount
) {
    const dim3 block_size(256, 1, 1);
    const dim3 grid_size((N + 127) / 128, (M + 127) / 128, batchCount);
    if (lnw && lnb && invsigma && bias) {
        sgemm_extra_128x128_8x8::sgemm_bias_lnorm_rrc_kernel<<<grid_size, block_size>>>(
            A, B, C, alpha, bias, M, N, K, aS, bS, cS, lnw, lnb, invsigma, norm_eps, muS, d_pos
        );
    } else if (bias) {
        sgemm_extra_128x128_8x8::sgemm_bias_rrc_kernel<<<grid_size, block_size>>>(
            A, B, C, alpha, bias, M, N, K, aS, bS, cS, d_pos
        );
    } else {
        sgemm_extra_128x128_8x8::sgemm_rrc_kernel<<<grid_size, block_size>>>(
            A, B, C, alpha, M, N, K, aS, bS, cS, d_pos
        );
    }
    if (pos && d_pos > 0) {
        const int n_head = N / 32;
        const dim3 block_size_2(256, 1, 1);
        const dim3 grid_size_2(n_head, (M + 255) / 256, batchCount);
        position::add_pos2d_rc_256x2_1x2<<<grid_size_2, block_size_2>>>(pos, C, M, n_head);
    }
}

__host__ void lnorm_grad_ccc_cuda(
    const float *grad_ln, const float *hat_ln, const float *invsigma, const float *lnw, const float *lnb,
    float *grad_input, float *grad_lnw, float *grad_lnb,
    const int M, const int N, const int hatS, const int muS, const int woPosS, const int d_pos,
    const int batchCount
) {
    const int n_head = (N + 31) / 32;
    const dim3 block_size(128, 1, 1);
    const dim3 grid_size(n_head, (M + 127) / 128, batchCount);
    lnorm::lnorm_grad_input_ccc_128x32_1x32<<<grid_size, block_size>>>(
        grad_ln, hat_ln, invsigma, lnw, lnb, grad_input, M, hatS, muS, woPosS, d_pos
    );
    const dim3 block_size_2(256, 1, 1);
    const dim3 grid_size_2(N, 1, 1);
    lnorm::lnorm_grad_weight_bias_ccrr_loop256x1_1x1<<<grid_size_2, block_size_2>>>(
        grad_ln, hat_ln, lnw, lnb, grad_lnw, grad_lnb, M, hatS, d_pos, batchCount
    );
}

} // namespace mh32 
