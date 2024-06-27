#pragma once
#include <cuda.h>
#include <cuComplex.h>
#include "buffer.cu"

namespace datamove {

__global__ void batched_transpose_16x16(
    const float *ld_ptr, float *st_ptr, const int M, const int N, const int stride
) {
    float *smem_buf = buffer::SharedMemory<float, 16 * 16>().pointer();
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

__global__ void transpose_gather_16x16(
    const cuComplex *ld_ptr, cuComplex *st_ptr,
    const int dim, const int H_all, const int W_all, const int H_low, const int W_low
) {
    cuComplex *smem_buf = buffer::SharedMemory<cuComplex, 16 * 16>().pointer();
    const int brid = blockIdx.y;
    const int bcid = blockIdx.x;
    int baseX, baseY;
    cuComplex reg;  // 存储线程所搬运的值

    baseX = bcid * 16 + threadIdx.x;  // 向右增长
    baseY = brid * 16 + threadIdx.y;  // 向下增长
    baseX = baseX + baseX / W_low * (W_all - W_low);  // 处理 Slice 间隔
    if ((baseY < dim) && (baseX < (H_low - 1) * W_all + W_low)) {
        // 行主序
        reg = *reinterpret_cast<const cuComplex*>(ld_ptr + baseY * H_all * W_all + baseX);
    }
    *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.y * 16 + threadIdx.x) = reg;
    __syncthreads();
    reg = *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.x * 16 + threadIdx.y);

    baseX = brid * 16 + threadIdx.x;  // 向下增长
    baseY = bcid * 16 + threadIdx.y;  // 向右增长
    if ((baseX < dim) && (baseY < H_low * W_low)) {
        // 列主序
        *reinterpret_cast<cuComplex*>(st_ptr + baseY * dim + baseX) = reg;
    }
}

__global__ void transpose_scatter_16x16(
    const cuComplex *ld_ptr, cuComplex *st_ptr,
    const int dim, const int H_all, const int W_all, const int H_low, const int W_low
) {
    cuComplex *smem_buf = buffer::SharedMemory<cuComplex, 16 * 16>().pointer();
    const int brid = blockIdx.y;
    const int bcid = blockIdx.x;
    int baseX, baseY;
    cuComplex reg;  // 存储线程所搬运的值

    baseX = brid * 16 + threadIdx.x;  // 向下增长
    baseY = bcid * 16 + threadIdx.y;  // 向右增长
    if ((baseX < dim) && (baseY < H_low * W_low)) {
        // 列主序
        reg = *reinterpret_cast<const cuComplex*>(ld_ptr + baseY * dim + baseX);
    }
    *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.y * 16 + threadIdx.x) = reg;
    __syncthreads();
    reg = *reinterpret_cast<cuComplex*>(smem_buf + threadIdx.x * 16 + threadIdx.y);
    baseX = bcid * 16 + threadIdx.x;  // 向右增长
    baseY = brid * 16 + threadIdx.y;  // 向下增长
    baseX = baseX + baseX / W_low * (W_all - W_low);  // 处理 Slice 间隔
    if ((baseY < dim) && (baseX < (H_low - 1) * W_all + W_low)) {
        // 行主序
        *reinterpret_cast<cuComplex*>(st_ptr + baseY * H_all * W_all + baseX) = reg;
    }
}

/// [batch, M, N] --> [batch, N, M]
__host__ void batched_transpose_cuda(
    const float *ld_ptr, float *st_ptr, const int M, const int N, const int stride, const int batchCount
) {
    const dim3 block_size(16, 16, 1);
    // 为避免 M,N 太大时导致超出 blockIdx.y 的上界，此处使用 blockIdx.x 平铺线程块
    const dim3 grid_size(((N + 15) / 16) * ((M + 15) / 16), 1, batchCount);
    batched_transpose_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, M, N, stride);
}

/// [dim, H_all, W_all] --> [dim, :H_low, :W_low] --> [H_low, W_low, dim]
__host__ void transpose_gather_cuda(
    const cuComplex *ld_ptr, cuComplex *st_ptr,
    const int dim, const int H_all, const int W_all, const int H_low, const int W_low
) {
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size((H_low * W_low + 15) / 16, (dim + 15) / 16, 1);
    transpose_gather_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, dim, H_all, W_all, H_low, W_low);
}

/// [:H_low, :W_low, dim] --> [H_all, W_all, dim] --> [dim, H_all, W_all]
__host__ void transpose_scatter_cuda(
    const cuComplex *ld_ptr, cuComplex *st_ptr,
    const int dim, const int H_all, const int W_all, const int H_low, const int W_low
) {
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size((H_low * W_low + 15) / 16, (dim + 15) / 16, 1);
    transpose_scatter_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, dim, H_all, W_all, H_low, W_low);
}

__global__ void batched_gather_transpose_2D_16x16(
    const float *ld_ptr, float *st_ptr, const int M1, const int M0, const int N, const int m0
) {
    float *smem_buf = buffer::SharedMemory<float, 16 * 16>().pointer();  // + 16 以避免 bank 冲突，提升不大
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

// [batch, M1, M0, N] --> [batch, M1, :m0, N] --> [batch, N, M1, m0]
__host__ void batched_gather_transpose_2D_cuda(
    const float *ld_ptr, float *st_ptr, const int M1, const int M0, const int N, const int m0, const int batchCount
) {
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size((N + 15) / 16, (M1 * m0 + 15) / 16, batchCount);
    batched_gather_transpose_2D_16x16<<<grid_size, block_size>>>(ld_ptr, st_ptr, M1, M0, N, m0);
}

/// [batch, num_head, dim_position: dim_position + dim_head, seqlen]
/// [batch, seqlen, num_head, dim_head]
__global__ void batched_transpose_gather_16x16(
    const float *ld_ptr, float *st_ptr,
    const int num_head, const int dim_position, const int dim_head, const int seqlen,
    const int batchCount
) {
    !

}

} // namespace datamove
