#ifndef __MATMUL_CU__
#define __MATMUL_CU__

#include <cuda.h>
#include "utils.cu"

namespace SpGT {
typedef enum {
    RRR = 0,
    RRC = 1,
    RCR = 2,
    RCC = 3,
    CRR = 4,
    CRC = 5,
    CCR = 6,
    CCC = 7
} Gemm_Order;

namespace sgemm_32x32_4x4 {
/* [WHEN] K <= 48 */
struct TileIndex {
    int brid, bcid, tid, wid, lid;
    int wrows, wcols, wrid, wcid, lrid, lcid;
    int M, N, K, aS, bS, cS;
    __device__ TileIndex(const int M, const int N, const int K, const int aS, const int bS, const int cS) {
        // 线程块与线程的标识
        brid = blockIdx.y; bcid = blockIdx.x; tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        wrid = wid / 2; wcid = wid % 2;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
        // 矩阵形状与跨步
        this->M = M; this->N = N; this->K = K;
        this->aS = aS; this->bS = bS; this->cS = cS;
    }
};

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[4][4], float *smem_buf, float *C,
    const int &M, const int &N, const int &cS,
    const int &brid, const int &bcid, const int &tid,
    const int &wrows, const int &wcols, const int &wrid, const int &wcid, const int &lrid, const int &lcid
) {
    float4 trans;
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = C + (blockIdx.z * cS + brid * 32 * N + bcid * 32);
    // 将所有线程的全部数据写入到共享内存
    __syncthreads();
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        trans.x = Creg[row][0]; trans.y = Creg[row][1]; trans.z = Creg[row][2]; trans.w = Creg[row][3];
        *reinterpret_cast<float4*>(
            smem_buf + (wrid * wrows * 4 * 32 + wcid * wcols * 4) + lrid * 4 * 32 + lcid * 4 + row * 32
        ) = trans;
    }
    __syncthreads();
    // 将数据从共享内存转移到全局内存
    // 使用 2x32 排列的线程搬运 32x32 共享内存，共需 16 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (int gmem_row = 0; gmem_row < 32; gmem_row += 2) {
        if ((brid * 32 + gmem_row + tid / 32 < M) && (bcid * 32 + tid % 32 < N)) {
            *reinterpret_cast<float*>(
                C_block + (gmem_row + tid / 32) * N + (tid % 32)
            ) = *reinterpret_cast<float*>(smem_buf + gmem_row * 32 + tid);
        }
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[4][4], float *smem_buf, float *C,
    const int &M, const int &N, const int &cS,
    const int &brid, const int &bcid, const int &tid,
    const int &wrows, const int &wcols, const int &wrid, const int &wcid, const int &lrid, const int &lcid
) {
    float4 trans;
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = C + (blockIdx.z * cS + bcid * 32 * M + brid * 32);
    // 将所有线程的全部数据写入到共享内存
    __syncthreads();
    #pragma unroll
    for (int column = 0; column < 4; ++column) {
        trans.x = Creg[0][column]; trans.y = Creg[1][column]; trans.z = Creg[2][column]; trans.w = Creg[3][column];
        *reinterpret_cast<float4*>(
            smem_buf + (wcid * wcols * 4 * 32 + wrid * wrows * 4) + lcid * 4 * 32 + lrid * 4 + column * 32
        ) = trans;
    }
    __syncthreads();
    // 将数据从共享内存转移到全局内存
    // 使用 32x2 排列的线程搬运 32x32 共享内存，共需 16 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (int gmem_column = 0; gmem_column < 32; gmem_column += 2) {
        if ((brid * 32 + tid % 32 < M) && (bcid * 32 + gmem_column + tid / 32 < N)) {
            *reinterpret_cast<float*>(
                C_block + (gmem_column + tid / 32) * M + (tid %32)
            ) = *reinterpret_cast<float*>(smem_buf + gmem_column * 32 + tid);
        }
    }
}

__device__ __forceinline__
void compute_tile_crr(
    float Creg[4][4], float *Asmem, float *Bsmem, const int &ldA, const int &ldB,
    const int &wrows, const int &wcols, const int &wrid, const int &wcid, const int &lrid, const int &lcid
) {
    float4 Areg, Breg;
    // 每个线程计算 C 的子域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (int kid = 0; kid < 8; ++kid) {
        Areg = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 4 + lrid * 4 + kid * ldA);
        Breg = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 4 + lcid * 4 + kid * ldB);
        Creg[0][0] += Areg.x * Breg.x;
        Creg[0][1] += Areg.x * Breg.y;
        Creg[0][2] += Areg.x * Breg.z;
        Creg[0][3] += Areg.x * Breg.w;
        Creg[1][0] += Areg.y * Breg.x;
        Creg[1][1] += Areg.y * Breg.y;
        Creg[1][2] += Areg.y * Breg.z;
        Creg[1][3] += Areg.y * Breg.w;
        Creg[2][0] += Areg.z * Breg.x;
        Creg[2][1] += Areg.z * Breg.y;
        Creg[2][2] += Areg.z * Breg.z;
        Creg[2][3] += Areg.z * Breg.w;
        Creg[3][0] += Areg.w * Breg.x;
        Creg[3][1] += Areg.w * Breg.y;
        Creg[3][2] += Areg.w * Breg.z;
        Creg[3][3] += Areg.w * Breg.w;
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + (256 + 32) * 2);
    
    // [NEXT] A_tid + eid * T.K + kth * 8
    // [NEXT] B_tid + eid * T.N + kth * 8 * T.N
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32) + (T.wid * 4 * T.N + T.lid);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.tid / 8 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid < T.N)               B_valid |= (1u << eid);
    }

    int kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
    Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_tid += kstart;
    B_tid += kstart * T.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (256 + 32);
        Bsmem += (2 * (kth & 1) - 1) * 256;
        *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
        Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_tid += 8;
        B_tid += 8 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__device__ __forceinline__
void compute_block_rcr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + (256 + 32) * 2);
    
    // [NEXT] A_tid + eid * T.K + kth * 8
    // [NEXT] B_tid + eid * T.K + kth * 8
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.tid / 8 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.tid / 8 * 4 + eid < T.N) B_valid |= (1u << eid);
    }

    int kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
        }
        if ((B_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
    *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_tid += kstart;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (256 + 32);
        Bsmem += (2 * (kth & 1) - 1) * (256 + 32);
        *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
        *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_tid += 8;
        B_tid += 8;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__device__ __forceinline__
void compute_block_crr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 256 * 2);
    
    // [NEXT] A_tid + eid * T.M + kth * 8 * T.M
    // [NEXT] B_tid + eid * T.N + kth * 8 * T.N
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32) + (T.wid * 4 * T.M + T.lid);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32) + (T.wid * 4 * T.N + T.lid);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid < T.N) B_valid |= (1u << eid);
    }

    int kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
    Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
    Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
    Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
    Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_tid += kstart * T.M;
    B_tid += kstart * T.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 256;
        Bsmem += (2 * (kth & 1) - 1) * 256;
        Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
        Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
        Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
        Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
        Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_tid += 8 * T.M;
        B_tid += 8 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__device__ __forceinline__
void compute_block_ccr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 256 * 2);
    
    // [NEXT] A_tid + eid * T.M + kth * 8 * T.M
    // [NEXT] B_tid + eid * T.K + kth * 8
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32) + (T.wid * 4 * T.M + T.lid);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid < T.M)               A_valid |= (1u << eid);
        if (T.bcid * 32 + T.tid / 8 * 4 + eid < T.N) B_valid |= (1u << eid);
    }

    int kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
        }
        if ((B_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
    Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
    Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
    Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_tid += kstart * T.M;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 256;
        Bsmem += (2 * (kth & 1) - 1) * (256 + 32);
        Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
        Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
        Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
        Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
        *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_tid += 8 * T.M;
        B_tid += 8;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__global__ void sgemm_rrr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_rcr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, (512 + 64) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_rcc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, (512 + 64) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_crr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, 512 * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_crc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, 512 * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_ccr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_ccc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__host__ void sgemm_cuda(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const Gemm_Order order, const int batchCount
) {
    const dim3 block_size(64, 1, 1);
    const dim3 grid_size((N + 31) / 32, (M + 31) / 32, batchCount);
    switch (order) {
    case Gemm_Order::RRR:
        sgemm_rrr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case Gemm_Order::RRC:
        sgemm_rrc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case Gemm_Order::RCR:
        sgemm_rcr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case Gemm_Order::RCC:
        sgemm_rcc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case Gemm_Order::CRR:
        sgemm_crr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case Gemm_Order::CRC:
        sgemm_crc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case Gemm_Order::CCR:
        sgemm_ccr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case Gemm_Order::CCC:
        sgemm_ccc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    default: break;
    }
}
} // namespace sgemm_32x32_4x4

namespace sgemm_32x32_4x8_SplitK {
/**
 * [WHEN]   K > 256
 * [LIMIT]  split_num <= 32
 * [BUFFER] batchCount * split_num * M * N * sizeof(float)
*/
struct TileIndexSplitK {
    int brid, bcid, tid, wid, lid;
    int wrows, wcols, lrid, lcid;
    int M, N, K, aS, bS, cS;
    // int slice_len, slice_idx;
    int split_len, split_idx, split_num, split_start, split_end;
    __device__ TileIndexSplitK(
        const int M, const int N, const int K, const int aS, const int bS, const int cS, const int regular_split_len
    ) {
        // 线程块与线程的标识，线程块使用 blockIdx.x 维度覆盖矩阵的 M,N 维度
        brid = blockIdx.x / ((N + 31) / 32);
        bcid = blockIdx.x % ((N + 31) / 32);
        tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
        // 矩阵形状与跨步
        this->M = M; this->N = N; this->K = K;
        this->aS = aS; this->bS = bS; this->cS = cS;
        // 沿着维度轴 K 切片的标识
        // slice_len = 4; slice_idx = wid;
        // 沿着维度轴 K 划分的标识
        split_idx = blockIdx.y; split_num = gridDim.y;
        split_len = regular_split_len;
        split_start = split_idx * split_len;
        split_end   = (split_idx + 1) * split_len;
        if (split_end > K) {
            split_end = K;
            split_len = split_end - split_start;
        }
    }
};

__device__ __forceinline__
void reduce_over_warp(
    float *smem, const int num_warp, const int num_datum, const int wid, const int lid
) {
    /* 在一个 Block 内的所有 Warp 之上进行归约，假设 num_datum 为 4 的整数倍 */
    for (int offset = num_warp / 2; offset >= 1; offset /= 2) {
        if (wid < offset) {
            #pragma unroll
            for (int i = lid * 4; i < num_datum; i += warpSize * 4) {
                float4 my = *reinterpret_cast<float4*>(smem + wid * num_datum + i);
                float4 other = *reinterpret_cast<float4*>(smem + (wid + offset) * num_datum + i);
                my.x += other.x; my.y += other.y; my.z += other.z; my.w += other.w;
                *reinterpret_cast<float4*>(smem + wid * num_datum + i) = my;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[2][4][4], float *smem_buf, float *SplitC, const int &split_num, const int &split_idx,
    const int &M, const int &N, const int &cS,
    const int &brid, const int &bcid, const int &tid, const int &wid, const int &lid,
    const int &wcols, const int &lrid, const int &lcid
) {
    // 存在 slice_num 份矩阵 C 子区域的部分结果，需先使用共享内存对其进行归约   
    float *Csmem = reinterpret_cast<float*>(smem_buf + 1024 * wid);
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = SplitC + (blockIdx.z * split_num * cS) + (split_idx * cS) + (brid * 32 * N + bcid * 32);

    float4 trans1, trans2;
    __syncthreads();
    // 首先，所有线程先将部分结果数据写入到共享内存，每个线程负责写回 Creg[2][4][4] 的数据
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        trans1.x = Creg[0][row][0]; trans1.y = Creg[0][row][1]; trans1.z = Creg[0][row][2]; trans1.w = Creg[0][row][3];
        trans2.x = Creg[1][row][0]; trans2.y = Creg[1][row][1]; trans2.z = Creg[1][row][2]; trans2.w = Creg[1][row][3];
        *reinterpret_cast<float4*>(
            Csmem + (0 * wcols * 4 + lcid * 4) + (lrid * 4 * 32 + row * 32)
        ) = trans1;
        *reinterpret_cast<float4*>(
            Csmem + (1 * wcols * 4 + lcid * 4) + (lrid * 4 * 32 + row * 32)
        ) = trans2;
    }
    __syncthreads();
    // 在 slice_num 个线程束之上进行归约，函数结束时存在显式 __syncthreads() 同步
    reduce_over_warp(smem_buf, 4, 1024, wid, lid);
    // 将数据从共享内存转移到全局内存
    // 使用 4x32 排列的线程搬运 32x32 共享内存，共需 8 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (int gmem_row = 0; gmem_row < 32; gmem_row += 4) {
        if ((brid * 32 + gmem_row + tid / 32 < M) && (bcid * 32 + tid % 32 < N)) {
            *reinterpret_cast<float*>(
                C_block + (gmem_row + tid / 32) * N + (tid % 32)
            ) = *reinterpret_cast<float*>(smem_buf + gmem_row * 32 + tid);
        }
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[2][4][4], float *smem_buf, float *SplitC, const int &split_num, const int &split_idx,
    const int &M, const int &N, const int &cS,
    const int &brid, const int &bcid, const int &tid, const int &wid, const int &lid,
    const int &wcols, const int &lrid, const int &lcid
) {
    // 存在 slice_num 份矩阵 C 子区域的部分结果，需先使用共享内存对其进行归约   
    float *Csmem = reinterpret_cast<float*>(smem_buf + 1024 * wid);
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = SplitC + (blockIdx.z * split_num * cS) + (split_idx * cS) + (bcid * 32 * M + brid * 32);

    float4 trans1, trans2;
    __syncthreads();
    // 首先，所有线程先将部分结果数据写入到共享内存，每个线程负责写回 Creg[2][4][4] 的数据
    #pragma unroll
    for (int column = 0; column < 4; ++column) {
        trans1.x = Creg[0][0][column]; trans1.y = Creg[0][1][column]; trans1.z = Creg[0][2][column]; trans1.w = Creg[0][3][column];
        trans2.x = Creg[1][0][column]; trans2.y = Creg[1][1][column]; trans2.z = Creg[1][2][column]; trans2.w = Creg[1][3][column];
        *reinterpret_cast<float4*>(
            Csmem + (0 * wcols * 4 * 32 + lcid * 4 * 32 + column * 32) + (lrid * 4)
        ) = trans1;
        *reinterpret_cast<float4*>(
            Csmem + (1 * wcols * 4 * 32 + lcid * 4 * 32 + column * 32) + (lrid * 4)
        ) = trans2;
    }
    __syncthreads();
    // 在 slice_num 个线程束之上进行归约，函数结束时存在显式 __syncthreads() 同步
    reduce_over_warp(smem_buf, 4, 1024, wid, lid);
    // 将数据从共享内存转移到全局内存
    // 使用 4x32 排列的线程搬运 32x32 共享内存，共需 8 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (int gmem_column = 0; gmem_column < 32; gmem_column += 4) {

        if ((brid * 32 + tid % 32 < M) && (bcid * 32 + gmem_column + tid / 32 < N)) {
            *reinterpret_cast<float*>(
                C_block + (gmem_column + tid / 32) * M + (tid % 32)
            ) = *reinterpret_cast<float*>(smem_buf + gmem_column * 32 + tid);
        }
    }
}

__device__ __forceinline__
void compute_tile_crr(
    float Creg[2][4][4], float *Asmem, float *Bsmem, const int &ldA, const int &ldB,
    const int &wcols, const int &lrid, const int &lcid
) {
    float4 Areg, Breg[2];
    // 每个线程计算 C 的子域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (int kid = 0; kid < 4; ++kid) {
        Areg = *reinterpret_cast<float4*>(Asmem + lrid * 4 + kid * ldA);
        Breg[0] = *reinterpret_cast<float4*>(Bsmem + 0 * wcols * 4 + lcid * 4 + kid * ldB);
        Breg[1] = *reinterpret_cast<float4*>(Bsmem + 1 * wcols * 4 + lcid * 4 + kid * ldB);
        #pragma unroll
        for (int cpj = 0; cpj < 2; ++cpj) {
            Creg[cpj][0][0] += Areg.x * Breg[cpj].x;
            Creg[cpj][0][1] += Areg.x * Breg[cpj].y;
            Creg[cpj][0][2] += Areg.x * Breg[cpj].z;
            Creg[cpj][0][3] += Areg.x * Breg[cpj].w;
            Creg[cpj][1][0] += Areg.y * Breg[cpj].x;
            Creg[cpj][1][1] += Areg.y * Breg[cpj].y;
            Creg[cpj][1][2] += Areg.y * Breg[cpj].z;
            Creg[cpj][1][3] += Areg.y * Breg[cpj].w;
            Creg[cpj][2][0] += Areg.z * Breg[cpj].x;
            Creg[cpj][2][1] += Areg.z * Breg[cpj].y;
            Creg[cpj][2][2] += Areg.z * Breg[cpj].z;
            Creg[cpj][2][3] += Areg.z * Breg[cpj].w;
            Creg[cpj][3][0] += Areg.w * Breg[cpj].x;
            Creg[cpj][3][1] += Areg.w * Breg[cpj].y;
            Creg[cpj][3][2] += Areg.w * Breg[cpj].z;
            Creg[cpj][3][3] += Areg.w * Breg[cpj].w;
        }
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndexSplitK &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid + (128 + 32) * 2);

    // [NEXT] A_lid + eid * T.K + kth * 16       + slice_idx * 4
    // [NEXT] B_lid + eid * T.N + kth * 16 * T.N + slice_idx * 4 * T.N
    const float *A_lid = A + (blockIdx.z * T.aS + T.brid * 32 * T.K) + T.split_start + (T.lid / 4 * 4 * T.K + T.lid % 4);
    const float *B_lid = B + (blockIdx.z * T.bS + T.bcid * 32) + T.split_start * T.N + T.lid;
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid / 4 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid < T.N)               B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    int kstart = T.split_len - ((T.split_len + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + T.lid % 4 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 + eid * T.K);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 * T.N + eid * T.N);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
    Bsmem[T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_lid += kstart;
    B_lid += kstart * T.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.split_len + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 + eid * T.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 * T.N + eid * T.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, T.wcols, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (128 + 32);
        Bsmem += (2 * (kth & 1) - 1) * 128;
        *reinterpret_cast<float4*>(Asmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
        Bsmem[T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_lid += 16;
        B_lid += 16 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, T.wcols, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_rcr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndexSplitK &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid + (128 + 32) * 2);

    // [NEXT] A_lid + eid * T.K + kth * 16 + slice_idx * 4
    // [NEXT] B_lid + eid * T.K + kth * 16 + slice_idx * 4
    const float *A_lid = A + (blockIdx.z * T.aS + T.brid * 32 * T.K) + T.split_start + (T.lid / 4 * 4 * T.K + T.lid % 4);
    const float *B_lid = B + (blockIdx.z * T.bS + T.bcid * 32 * T.K) + T.split_start + (T.lid / 4 * 4 * T.K + T.lid % 4);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid / 4 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid / 4 * 4 + eid < T.N) B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    int kstart = T.split_len - ((T.split_len + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + T.lid % 4 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 + eid * T.K);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + T.lid % 4 < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 + eid * T.K);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
    *reinterpret_cast<float4*>(Bsmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_lid += kstart;
    B_lid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.split_len + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 + eid * T.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 + eid * T.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, T.wcols, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (128 + 32);
        Bsmem += (2 * (kth & 1) - 1) * (128 + 32);
        *reinterpret_cast<float4*>(Asmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
        *reinterpret_cast<float4*>(Bsmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_lid += 16;
        B_lid += 16;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, T.wcols, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_crr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndexSplitK &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid + 128 * 2);

    // [NEXT] A_lid + eid * T.M + kth * 16 * T.M + slice_idx * 4 * T.M
    // [NEXT] B_lid + eid * T.N + kth * 16 * T.N + slice_idx * 4 * T.N
    const float *A_lid = A + (blockIdx.z * T.aS + T.brid * 32) + T.split_start * T.M + T.lid;
    const float *B_lid = B + (blockIdx.z * T.bS + T.bcid * 32) + T.split_start * T.N + T.lid;
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid < T.N) B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    int kstart = T.split_len - ((T.split_len + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 * T.M + eid * T.M);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 * T.N + eid * T.N);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[T.lid + 0 * 32] = Atrans[0];
    Asmem[T.lid + 1 * 32] = Atrans[1];
    Asmem[T.lid + 2 * 32] = Atrans[2];
    Asmem[T.lid + 3 * 32] = Atrans[3];
    Bsmem[T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_lid += kstart * T.M;
    B_lid += kstart * T.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.split_len + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 * T.M + eid * T.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 * T.N + eid * T.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, T.wcols, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 128;
        Bsmem += (2 * (kth & 1) - 1) * 128;
        Asmem[T.lid + 0 * 32] = Atrans[0];
        Asmem[T.lid + 1 * 32] = Atrans[1];
        Asmem[T.lid + 2 * 32] = Atrans[2];
        Asmem[T.lid + 3 * 32] = Atrans[3];
        Bsmem[T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_lid += 16 * T.M;
        B_lid += 16 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, T.wcols, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_ccr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndexSplitK &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 1024 * T.wid + 128 * 2);

    // [NEXT] A_lid + eid * T.M + kth * 16 * T.M + slice_idx * 4 * T.M
    // [NEXT] B_lid + eid * T.K + kth * 16 + slice_idx * 4
    const float *A_lid = A + (blockIdx.z * T.aS + T.brid * 32) + T.split_start * T.M + T.lid;
    const float *B_lid = B + (blockIdx.z * T.bS + T.bcid * 32 * T.K) + T.split_start + (T.lid / 4 * 4 * T.K + T.lid % 4);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    unsigned int A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid < T.M)               A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid / 4 * 4 + eid < T.N) B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    int kstart = T.split_len - ((T.split_len + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (int eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 * T.M + eid * T.M);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + T.lid % 4 < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 + eid * T.K);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[T.lid + 0 * 32] = Atrans[0];
    Asmem[T.lid + 1 * 32] = Atrans[1];
    Asmem[T.lid + 2 * 32] = Atrans[2];
    Asmem[T.lid + 3 * 32] = Atrans[3];
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Bsmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_lid += kstart * T.M;
    B_lid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (int kth = 1; kth < (T.split_len + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_lid + T.wid * 4 * T.M + eid * T.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_lid + T.wid * 4 + eid * T.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, T.wcols, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 128;
        Bsmem += (2 * (kth & 1) - 1) * (128 + 32);
        Asmem[T.lid + 0 * 32] = Atrans[0];
        Asmem[T.lid + 1 * 32] = Atrans[1];
        Asmem[T.lid + 2 * 32] = Atrans[2];
        Asmem[T.lid + 3 * 32] = Atrans[3];
        // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
        *reinterpret_cast<float4*>(Bsmem + T.lid % 4 * 36 + T.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_lid += 16 * T.M;
        B_lid += 16;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, T.wcols, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (int cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__global__ void sgemm_rrr_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void sgemm_rcr_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void sgemm_rcc_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void sgemm_crr_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void sgemm_crc_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void sgemm_ccr_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void sgemm_ccc_kernel(
    const float *A, const float *B, float *SplitC, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const int split_len
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndexSplitK T(M, N, K, aS, bS, cS, split_len);
    float Creg[2][4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(
        Creg, smem_buf, SplitC, T.split_num, T.split_idx, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wid, T.lid, T.wcols, T.lrid, T.lcid
    );
}

__global__ void reduce_kernel(
    const float *split_gmem, float *dest_gmem, const int strided, const int split_num
) {
    /* 将 split_gmem 内存当中的 split_num 个长度为 strided 的数据对象进行归约，存至 dest_gmem 中 */
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const float *split_batch = split_gmem + blockIdx.z * split_num * strided;
    float *dest_batch = dest_gmem + blockIdx.z * strided;
    float reg = 0.F;
    if (tid < strided) {
        // 循环展开导致一个线程同时跨取多个 split 的数据，从而导致非合并访存，更慢
        // #pragma unroll
        for (int split_idx = 0; split_idx < split_num; ++split_idx) {
            reg += split_batch[split_idx * strided + tid];
        }
        dest_batch[tid] = reg;
    }
}

__host__ void sgemm_cuda(
    const float *A, const float *B, float *C, const float alpha,
    const int M, const int N, const int K, const int aS, const int bS, const int cS,
    const Gemm_Order order, const int batchCount
) {
    int split_len = 64;  // 每个划分的长度
    int split_num = 24;  // 最大划分的个数
    const int split_threshold = 4096;  // 允许最大划分数增长的阈值
    split_num = split_num * static_cast<int>((K + split_threshold - 1) / split_threshold);
    if (K <= split_len * split_num) {
        split_num = (K + split_len - 1) / split_len;  // 调小 split_num
    } else {
        split_len = (K + split_num - 1) / split_num;  // 调大 split_len
    }
    // 执行运算所需的缓冲区
    const size_t bytes = batchCount * split_num * M * N * sizeof(float);
    float *SplitC = reinterpret_cast<float*>(GlobalBuffer::I().pointer(bytes));

    const dim3 block_size(128, 1, 1);
    const dim3 grid_size(((N + 31) / 32) * ((M + 31) / 32), split_num, batchCount);
    switch (order) {
    case Gemm_Order::RRR:
        sgemm_rrr_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    case Gemm_Order::RRC:
        sgemm_rrc_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    case Gemm_Order::RCR:
        sgemm_rcr_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    case Gemm_Order::RCC:
        sgemm_rcc_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    case Gemm_Order::CRR:
        sgemm_crr_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    case Gemm_Order::CRC:
        sgemm_crc_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    case Gemm_Order::CCR:
        sgemm_ccr_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    case Gemm_Order::CCC:
        sgemm_ccc_kernel<<<grid_size, block_size>>>(A, B, SplitC, alpha, M, N, K, aS, bS, cS, split_len); break;
    default: break;
    }
    const dim3 block_size_call2(256, 1, 1);
    const dim3 grid_size_call2((M * N + 255) / 256, 1, batchCount);
    reduce_kernel<<<grid_size_call2, block_size_call2>>>(SplitC, C, cS, split_num);
}
} // namespace sgemm_32x32_4x8_SplitK

} // namespace SpGT

#endif // __MATMUL_CU__