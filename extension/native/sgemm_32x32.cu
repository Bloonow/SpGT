#pragma once
#include <cuda.h>
#include "../utils/ptx.cu"
#include "../utils/buffer.cu"
#include "../utils/gemm_utils.cu"

namespace sgemm_32x32_4x4 {

/* [WHEN] K <= 48 */
struct TileIndex {
    uint32_t brid, bcid, tid, wid, lid;
    uint32_t wrows, wcols, wrid, wcid, lrid, lcid;
    __device__ TileIndex() {
        // 线程块与线程的标识
        brid = blockIdx.y; bcid = blockIdx.x;
        tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        wrid = wid / 2; wcid = wid % 2;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
    }
};

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[4][4], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    // 使用 32x32 共享内存搬运 32x32 数据（需 1 次）
    // [NEXT] C_smem_st + r * 32 * sizeof(float)
    uint32_t C_smem_st = ptx::smem_addr(smem_buf + (wrid * wrows * 4 * 32 + wcid * wcols * 4) + lrid * 4 * 32 + lcid * 4);
    float *C_block = C + (blockIdx.z * cS + brid * 32 * N + bcid * 32);
    // 将所有线程的全部数据写入到共享内存
    __syncthreads();
    ptx::st_smem(Creg[0][0], Creg[0][1], Creg[0][2], Creg[0][3], C_smem_st + 0 * 32 * sizeof(float));
    ptx::st_smem(Creg[1][0], Creg[1][1], Creg[1][2], Creg[1][3], C_smem_st + 1 * 32 * sizeof(float));
    ptx::st_smem(Creg[2][0], Creg[2][1], Creg[2][2], Creg[2][3], C_smem_st + 2 * 32 * sizeof(float));
    ptx::st_smem(Creg[3][0], Creg[3][1], Creg[3][2], Creg[3][3], C_smem_st + 3 * 32 * sizeof(float));
    __syncthreads();
    // 使用 2x32 排列的线程搬运 32x32 共享内存（需 16 次），每次每线程写回 1 个数据
    #pragma unroll
    for (uint32_t gmem_row = 0; gmem_row < 32; gmem_row += 2) {
        ptx::st_gmem(
            *reinterpret_cast<float*>(smem_buf + gmem_row * 32 + tid),
            C_block + (gmem_row + tid / 32) * N + (tid % 32),
            (brid * 32 + gmem_row + tid / 32 < M) && (bcid * 32 + tid % 32 < N)
        );
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[4][4], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    // 使用 32x32 共享内存搬运 32x32 数据（需 1 次）
    // [NEXT] C_smem_st + c * 32 * sizeof(float)
    uint32_t C_smem_st = ptx::smem_addr(smem_buf + (wcid * wcols * 4 * 32 + wrid * wrows * 4) + lcid * 4 * 32 + lrid * 4);
    float *C_block = C + (blockIdx.z * cS + bcid * 32 * M + brid * 32);
    // 将所有线程的全部数据写入到共享内存
    __syncthreads();
    ptx::st_smem(Creg[0][0], Creg[1][0], Creg[2][0], Creg[3][0], C_smem_st + 0 * 32 * sizeof(float));
    ptx::st_smem(Creg[0][1], Creg[1][1], Creg[2][1], Creg[3][1], C_smem_st + 1 * 32 * sizeof(float));
    ptx::st_smem(Creg[0][2], Creg[1][2], Creg[2][2], Creg[3][2], C_smem_st + 2 * 32 * sizeof(float));
    ptx::st_smem(Creg[0][3], Creg[1][3], Creg[2][3], Creg[3][3], C_smem_st + 3 * 32 * sizeof(float));
    __syncthreads();
    // 使用 32x2 排列的线程搬运 32x32 共享内存（需 16 次），每次每线程写回 1 个数据
    #pragma unroll
    for (uint32_t gmem_column = 0; gmem_column < 32; gmem_column += 2) {
        ptx::st_gmem(
            *reinterpret_cast<float*>(smem_buf + gmem_column * 32 + tid),
            C_block + (gmem_column + tid / 32) * M + (tid %32),
            (brid * 32 + tid % 32 < M) && (bcid * 32 + gmem_column + tid / 32 < N)
        );
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + (256 + 32) * 2);
    // [NEXT] A_smem_st
    // [NEXT] B_smem_st + eid * 32 * sizeof(float)
    uint32_t A_smem_st = ptx::smem_addr(A_smem + tid % 8 * 36 + tid / 8 * 4);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + wid * 4 * 32 + lid);
    // [NEXT] A_smem_ld + kid * ldA * sizeof(float)
    // [NEXT] B_smem_ld + kid * ldB * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 4 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 4 + lcid * 4);
    // [NEXT] A_tid + eid * K + kth * 8
    // [NEXT] B_tid + eid * N + kth * 8 * N
    const float *A_tid = A + (blockIdx.z * aS + brid * 32 * K) + (tid / 8 * 4 * K + tid % 8);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 32) + (wid * 4 * N + lid);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 32 + tid / 8 * 4 + eid < M) << eid;
        B_valid |= (uint32_t)(bcid * 32 + lid < N)               << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[4] = {0.f}, Breg[4] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * K, (A_valid & (1u << eid)) && (tid % 8 < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * N, (B_valid & (1u << eid)) && (wid * 4 + eid < kstart));
    }
    // 将预取数据写入到共享内存
    ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
    }
    __syncthreads();
    // 切换缓冲区
    A_smem_st += (256 + 32) * sizeof(float);
    B_smem_st += 256 * sizeof(float);
    // 数据指针向后移动 k 个数据
    A_tid += kstart;
    B_tid += kstart * N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * K, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * N, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 36 * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 32 * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 4; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 4; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
        }
        __syncthreads();
        // 切换缓冲区
        A_smem_st += (1 - 2 * (kth & 1)) * (256 + 32) * sizeof(float);
        B_smem_st += (1 - 2 * (kth & 1)) * 256 * sizeof(float);
        A_smem_ld += (2 * (kth & 1) - 1) * (256 + 32) * sizeof(float);
        B_smem_ld += (2 * (kth & 1) - 1) * 256 * sizeof(float);
        // 数据指针向后移动 k 个数据
        A_tid += 8;
        B_tid += 8 * N;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 36 * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 32 * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 4; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 4; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 4; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 4; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_rcr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + (256 + 32) * 2);
    // [NEXT] A_smem_st
    // [NEXT] B_smem_st
    uint32_t A_smem_st = ptx::smem_addr(A_smem + tid % 8 * 36 + tid / 8 * 4);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + tid % 8 * 36 + tid / 8 * 4);
    // [NEXT] A_smem_ld + kid * ldA * sizeof(float)
    // [NEXT] B_smem_ld + kid * ldB * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 4 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 4 + lcid * 4);
    // [NEXT] A_tid + eid * K + kth * 8
    // [NEXT] B_tid + eid * K + kth * 8
    const float *A_tid = A + (blockIdx.z * aS + brid * 32 * K) + (tid / 8 * 4 * K + tid % 8);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 32 * K) + (tid / 8 * 4 * K + tid % 8);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 32 + tid / 8 * 4 + eid < M) << eid;
        B_valid |= (uint32_t)(bcid * 32 + tid / 8 * 4 + eid < N) << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[4] = {0.f}, Breg[4] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * K, (A_valid & (1u << eid)) && (tid % 8 < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * K, (B_valid & (1u << eid)) && (tid % 8 < kstart));
    }
    // 将预取数据写入到共享内存
    ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
    ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
    __syncthreads();
    // 切换缓冲区
    A_smem_st += (256 + 32) * sizeof(float);
    B_smem_st += (256 + 32) * sizeof(float);
    // 数据指针向后移动 k 个数据
    A_tid += kstart;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * K, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * K, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 36 * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 36 * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 4; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 4; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
        ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
        __syncthreads();
        // 切换缓冲区
        A_smem_st += (1 - 2 * (kth & 1)) * (256 + 32) * sizeof(float);
        B_smem_st += (1 - 2 * (kth & 1)) * (256 + 32) * sizeof(float);
        A_smem_ld += (2 * (kth & 1) - 1) * (256 + 32) * sizeof(float);
        B_smem_ld += (2 * (kth & 1) - 1) * (256 + 32) * sizeof(float);
        // 数据指针向后移动 k 个数据
        A_tid += 8;
        B_tid += 8;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 36 * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 36 * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 4; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 4; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 4; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 4; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_crr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 256 * 2);
    // [NEXT] A_smem_st + eid * 32 * sizeof(float)
    // [NEXT] B_smem_st + eid * 32 * sizeof(float)
    uint32_t A_smem_st = ptx::smem_addr(A_smem + wid * 4 * 32 + lid);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + wid * 4 * 32 + lid);
    // [NEXT] A_smem_ld + kid * ldA * sizeof(float)
    // [NEXT] B_smem_ld + kid * ldB * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 4 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 4 + lcid * 4);
    // [NEXT] A_tid + eid * M + kth * 8 * M
    // [NEXT] B_tid + eid * N + kth * 8 * N
    const float *A_tid = A + (blockIdx.z * aS + brid * 32) + (wid * 4 * M + lid);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 32) + (wid * 4 * N + lid);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 32 + lid < M) << eid;
        B_valid |= (uint32_t)(bcid * 32 + lid < N) << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[4] = {0.f}, Breg[4] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * M, (A_valid & (1u << eid)) && (wid * 4 + eid < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * N, (B_valid & (1u << eid)) && (wid * 4 + eid < kstart));
    }
    // 将预取数据写入到共享内存
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
        ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
    }
    __syncthreads();
    // 切换缓冲区
    A_smem_st += 256 * sizeof(float);
    B_smem_st += 256 * sizeof(float);
    // 数据指针向后移动 k 个数据
    A_tid += kstart * M;
    B_tid += kstart * N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * M, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * N, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 32 * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 32 * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 4; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 4; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
            ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
        }
        __syncthreads();
        // 切换缓冲区
        A_smem_st += (1 - 2 * (kth & 1)) * 256 * sizeof(float);
        B_smem_st += (1 - 2 * (kth & 1)) * 256 * sizeof(float);
        A_smem_ld += (2 * (kth & 1) - 1) * 256 * sizeof(float);
        B_smem_ld += (2 * (kth & 1) - 1) * 256 * sizeof(float);
        // 数据指针向后移动 k 个数据
        A_tid += 8 * M;
        B_tid += 8 * N;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 32 * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 32 * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 4; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 4; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 4; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 4; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_ccr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 256 * 2);
    // [NEXT] A_smem_st + eid * 32 * sizeof(float)
    // [NEXT] B_smem_st
    uint32_t A_smem_st = ptx::smem_addr(A_smem + wid * 4 * 32 + lid);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + tid % 8 * 36 + tid / 8 * 4);
    // [NEXT] A_smem_ld + kid * ldA * sizeof(float)
    // [NEXT] B_smem_ld + kid * ldB * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 4 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 4 + lcid * 4);
    // [NEXT] A_tid + eid * M + kth * 8 * M
    // [NEXT] B_tid + eid * K + kth * 8
    const float *A_tid = A + (blockIdx.z * aS + brid * 32) + (wid * 4 * M + lid);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 32 * K) + (tid / 8 * 4 * K + tid % 8);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 32 + lid < M)               << eid;
        B_valid |= (uint32_t)(bcid * 32 + tid / 8 * 4 + eid < N) << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[4] = {0.f}, Breg[4] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * M, (A_valid & (1u << eid)) && (wid * 4 + eid < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * K, (B_valid & (1u << eid)) && (tid % 8 < kstart));
    }
    // 将预取数据写入到共享内存
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
    }
    ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
    __syncthreads();
    // 切换缓冲区
    A_smem_st += 256 * sizeof(float);
    B_smem_st += (256 + 32) * sizeof(float);
    // 数据指针向后移动 k 个数据
    A_tid += kstart * M;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * M, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * K, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 32 * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 36 * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 4; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 4; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
        }
        ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
        __syncthreads();
        // 切换缓冲区
        A_smem_st += (1 - 2 * (kth & 1)) * 256 * sizeof(float);
        B_smem_st += (1 - 2 * (kth & 1)) * (256 + 32) * sizeof(float);
        A_smem_ld += (2 * (kth & 1) - 1) * 256 * sizeof(float);
        B_smem_ld += (2 * (kth & 1) - 1) * (256 + 32) * sizeof(float);
        // 数据指针向后移动 k 个数据
        A_tid += 8 * M;
        B_tid += 8;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + kid * 32 * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + kid * 36 * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 4; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 4; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 4; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 4; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__global__ void sgemm_rrr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_rrr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_rrr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__global__ void sgemm_rcr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, (512 + 64) * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_rcr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__global__ void sgemm_rcc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, (512 + 64) * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_rcr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__global__ void sgemm_crr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 512 * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_crr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__global__ void sgemm_crc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 512 * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_crr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__global__ void sgemm_ccr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_ccr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__global__ void sgemm_ccc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T;
    float Creg[4][4] = {0.f};
    compute_block_ccr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__host__ void sgemm_cuda(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS,
    const GEMM_Order order, const uint32_t batchCount
) {
    const dim3 block_size(64, 1, 1);
    const dim3 grid_size((N + 31) / 32, (M + 31) / 32, batchCount);
    switch (order) {
    case GEMM_Order::RRR:
        sgemm_rrr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::RRC:
        sgemm_rrc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::RCR:
        sgemm_rcr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::RCC:
        sgemm_rcc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CRR:
        sgemm_crr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CRC:
        sgemm_crc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CCR:
        sgemm_ccr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CCC:
        sgemm_ccc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    default: break;
    }
}

} // namespace sgemm_32x32_4x4
