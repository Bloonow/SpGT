#pragma once

#include <cuda.h>
#include <cuComplex.h>
#include "utils.cu"

namespace SpGT {

namespace datamove {
__global__ void batched_transpose_2D_16x16(
    const float *ld_ptr, float *st_ptr, const int M, const int N, const int stride
) {
    float *smem_buf = SharedMemory<float, 16 * 16>().pointer();  // + 16 以避免 bank 冲突，提升不大
    const int brid = blockIdx.x / ((N + 15) / 16);
    const int bcid = blockIdx.x % ((N + 15) / 16);

    int baseX, baseY;
    float reg;  // 存储线程所搬运的值

    baseX = bcid * 16 + threadIdx.x;  // 向右增长
    baseY = brid * 16 + threadIdx.y;  // 向下增长
    if ((baseY < M) && (baseX < N)) {
        // 行主序
        reg = *reinterpret_cast<const float*>(ld_ptr + blockIdx.z * stride + baseY * N + baseX);
    }
    *reinterpret_cast<float*>(smem_buf + threadIdx.y * 16 + threadIdx.x) = reg;
    __syncthreads();
    reg = *reinterpret_cast<float*>(smem_buf + threadIdx.x * 16 + threadIdx.y);

    baseX = brid * 16 + threadIdx.x;  // 向下增长
    baseY = bcid * 16 + threadIdx.y;  // 向右增长
    if ((baseX < M) && (baseY < N)) {
        // 列主序
        *reinterpret_cast<float*>(st_ptr + blockIdx.z * stride + baseY * M + baseX) = reg;
    }
}

__global__ void gather_transpose_2D_16x16(
    const cuComplex *ld_ptr, cuComplex *st_ptr, const int M, const int N0, const int N1, const int n0, const int n1
) {
    cuComplex *smem_buf = SharedMemory<cuComplex, 16 * 16>().pointer();
    const int brid = blockIdx.y;
    const int bcid = blockIdx.x;
    int baseX, baseY;
    cuComplex reg;  // 存储线程所搬运的值

    baseX = bcid * 16 + threadIdx.x;  // 向右增长
    baseY = brid * 16 + threadIdx.y;  // 向下增长
    baseX = baseX + baseX / n0 * (N0 - n0);  // 处理 Slice 间隔
    if ((baseY < M) && (baseX < (n1 - 1) * N0 + n0)) {
        // 行主序
        reg = *reinterpret_cast<const cuComplex*>(ld_ptr + baseY * N0 * N1 + baseX);
    }
    *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.y * 16 + threadIdx.x) = reg;
    __syncthreads();
    reg = *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.x * 16 + threadIdx.y);

    baseX = brid * 16 + threadIdx.x;  // 向下增长
    baseY = bcid * 16 + threadIdx.y;  // 向右增长
    if ((baseX < M) && (baseY < n0 * n1)) {
        // 列主序
        *reinterpret_cast<cuComplex*>(st_ptr + baseY * M + baseX) = reg;
    }
}

__global__ void scatter_transpose_2D_16x16(
    const cuComplex *ld_ptr, cuComplex *st_ptr, const int M, const int N0, const int N1, const int n0, const int n1
) {
    cuComplex *smem_buf = SharedMemory<cuComplex, 16 * 16>().pointer();
    const int brid = blockIdx.y;
    const int bcid = blockIdx.x;
    int baseX, baseY;
    cuComplex reg;  // 存储线程所搬运的值

    baseX = brid * 16 + threadIdx.x;  // 向下增长
    baseY = bcid * 16 + threadIdx.y;  // 向右增长
    if ((baseX < M) && (baseY < n0 * n1)) {
        // 列主序
        reg = *reinterpret_cast<const cuComplex*>(ld_ptr + baseY * M + baseX);
    }
    *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.y * 16 + threadIdx.x) = reg;
    __syncthreads();
    reg = *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.x * 16 + threadIdx.y);

    baseX = bcid * 16 + threadIdx.x;  // 向右增长
    baseY = brid * 16 + threadIdx.y;  // 向下增长
    baseX = baseX + baseX / n0 * (N0 - n0);  // 处理 Slice 间隔
    if ((baseY < M) && (baseX < (n1 - 1) * N0 + n0)) {
        // 行主序
        *reinterpret_cast<cuComplex*>(st_ptr + baseY * N0 * N1 + baseX) = reg;
    }
}

__global__ void batched_gather_transpose_2D_16x16(
    const float *ld_ptr, float *st_ptr, const int M1, const int M0, const int N, const int m0
) {
    float *smem_buf = SharedMemory<float, 16 * 16>().pointer();  // + 16 以避免 bank 冲突，提升不大
    const int brid = blockIdx.y;
    const int bcid = blockIdx.x;
    int baseX, baseY;
    float reg;  // 存储线程所搬运的值

    baseX = bcid * 16 + threadIdx.x;  // 向右增长
    baseY = brid * 16 + threadIdx.y;  // 向下增长
    baseY = baseY + baseY / m0 * (M0 - m0);  // 处理 Slice 间隔
    if ((baseY < (M1 - 1) * M0 + m0) && (baseX < N)) {
        // 行主序
        reg = *reinterpret_cast<const float*>(ld_ptr + blockIdx.z * (M1 * M0 * N) + baseY * N + baseX);
    }
    *reinterpret_cast<float*>(smem_buf + threadIdx.y * 16 + threadIdx.x) = reg;
    __syncthreads();
    reg = *reinterpret_cast<float*>(smem_buf + threadIdx.x * 16 + threadIdx.y);

    baseX = brid * 16 + threadIdx.x;  // 向下增长
    baseY = bcid * 16 + threadIdx.y;  // 向右增长
    if ((baseX < M1 * m0) && (baseY < N)) {
        *reinterpret_cast<float*>(st_ptr + blockIdx.z * (M1 * m0 * N) + baseY * M1 * m0 + baseX) = reg;
    }
}

/* [batch, M, N] --> [batch, N, M] */
__host__ void batched_transpose_2D_cuda(
    const float *ld_ptr, float *st_ptr, const int M, const int N, const int stride, const int batchCount
) {
    const dim3 block_size(16, 16, 1);
    // 为避免 M,N 太大时导致超出 blockIdx.y 的上界，此处使用 blockIdx.x 平铺线程块
    const dim3 grid_size(((N + 15) / 16) * ((M + 15) / 16), 1, batchCount);
    batched_transpose_2D_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, M, N, stride);
}

/* [M, N1, N0] --> [M, :n1, :n0] --> [n1, n0, M] */
__host__ void gather_transpose_2D_cuda(
    const cuComplex *ld_ptr, cuComplex *st_ptr, const int M, const int N0, const int N1, const int n0, const int n1
) {
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size((n0 * n1 + 15) / 16, (M + 15) / 16, 1);
    gather_transpose_2D_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, M, N0, N1, n0, n1);
}

/* [:n1, :n0, M] --> [N1, N0, M] --> [M, N1, N0] */
__host__ void scatter_transpose_2D_cuda(
    const cuComplex *ld_ptr, cuComplex *st_ptr, const int M, const int N0, const int N1, const int n0, const int n1
) {
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size((n0 * n1 + 15) / 16, (M + 15) / 16, 1);
    scatter_transpose_2D_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, M, N0, N1, n0, n1);
}

/* [batch, M1, M0, N] --> [batch, M1, :m0, N] --> [batch, N, M1, m0] */
__host__ void batched_gather_transpose_2D(
    const float *ld_ptr, float *st_ptr, const int M1, const int M0, const int N, const int m0, const int batchCount
) {
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size((N + 15) / 16, (M1 * m0 + 15) / 16, batchCount);
    batched_gather_transpose_2D_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, M1, M0, N, m0);
}

} // namespace datamove
} // namespace SpGT
