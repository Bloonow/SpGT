#pragma once
#include <cuda.h>

namespace ptx {

__device__ __forceinline__
uint32_t smem_addr(const void *ptr) {
    /* 共享内存指针 ptr 转换为 addr 地址 */
    uint32_t addr;
    asm volatile (
        "{\n"
        ".reg .u64 u64addr;\n"
        "cvta.to.shared.u64 u64addr, %1;\n"
        "cvt.u32.u64 %0, u64addr;\n"
        "}\n"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
}
__device__ __forceinline__
void st_smem(const float &reg, const uint32_t &addr) {
    /* 向共享内存 addr 中写入 1 个 float 数据 */
    asm volatile (
        "st.shared.f32 [%1], %0;\n"
        : : "f"(reg), "r"(addr)
    );
}
__device__ __forceinline__
void st_smem(const float &r0, const float &r1, const float &r2, const float &r3, const uint32_t &addr) {
    /* 向共享内存 addr 中写入 4 个 float 数据 */
    asm volatile (
        "st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n"
        : : "f"(r0), "f"(r1), "f"(r2), "f"(r3), "r"(addr)
    );
}
__device__ __forceinline__
void ld_smem(float &reg, const uint32_t &addr) {
    /* 从共享内存 addr 中读取 1 个 float 数据 */
    asm volatile (
        "ld.shared.f32 %0, [%1];\n"
        : "=f"(reg)
        : "r"(addr)
    );
}
__device__ __forceinline__
void ld_smem(float &r0, float &r1, float &r2, float &r3, const uint32_t &addr) {
    /* 从共享内存 addr 中读取 4 个 float 数据 */
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "r"(addr)
    );
}
__device__ __forceinline__
void ld_gmem(float &reg, const void *ptr, bool guard) {
    /* 当 guard 为 true 时，从全局内存 ptr 中读取 1 个 float 数据 */
    #if (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750)
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p ld.global.nc.L2::128B.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #else
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p ld.global.nc.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #endif
}
__device__ __forceinline__
void ld_gmem_zero(float &reg, const void *ptr, bool guard) {
    /* 当 guard 为 true 时，从全局内存 ptr 中读取 1 个 float 数据，否则置零 */
    #if (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750)
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@!p mov.b32 %0, 0;\n"
        "@p ld.global.nc.L2::128B.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #else
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@!p mov.b32 %0, 0;\n"
        "@p ld.global.nc.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #endif
}
__device__ __forceinline__
void st_gmem(const float &reg, void *ptr, bool guard) {
    /* 当 guard 为 true 时，向全局内存 ptr 中写入 1 个 float 数据 */
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p st.global.f32 [%1], %0;\n"
        "}\n"
        : : "f"(reg), "l"(ptr), "r"((int)guard)
    );
}

} // namespace ptx