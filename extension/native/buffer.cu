#pragma once
#include <cuComplex.h>

namespace buffer {

template <typename Type, int num_datum> struct SharedMemory;
template <int num_datum> struct SharedMemory<float, num_datum> {
    __device__ float *pointer() {
        __shared__ __align__(128) float __shared_memory__[num_datum];
        return reinterpret_cast<float*>(__shared_memory__);
    }
};
template <int num_datum> struct SharedMemory<cuComplex, num_datum> {
    __device__ cuComplex *pointer() {
        __shared__ __align__(128) cuComplex __shared_memory__[num_datum];
        return reinterpret_cast<cuComplex*>(__shared_memory__);
    }
};

// #define GLOBAL_BUFFER_MESSAGE
class GlobalBuffer {
private:
    // 保留默认构造方法
    GlobalBuffer() = default;
    void *_buffer = NULL;
    size_t _bytes = 0;
    void try_free() {
        // 若存在已分配空间，则释放
        if (_buffer != NULL && _bytes != 0) {
            cudaFree(_buffer);
            #ifdef GLOBAL_BUFFER_MESSAGE
            printf("\n[GlobalBuffer] Free %.2f MiB\n\n", _bytes * 1. / (1024 * 1024));
            #endif
            _buffer = NULL; _bytes = 0;
        }
    }
    void try_malloc(const size_t bytes) {
        if (bytes > _bytes) {
            // 未分配空间，或已分配空间不满足要求，先尝试释放已分配空间
            try_free();
            // 重新分配空间
            _bytes = bytes;
            cudaMalloc(&_buffer, _bytes);
            #ifdef GLOBAL_BUFFER_MESSAGE
            printf("\n[GlobalBuffer] Malloc %.2f MiB\n\n", _bytes * 1. / (1024 * 1024));
            #endif
        }
    }
public:
    // 删除其他构造方法
    GlobalBuffer(const GlobalBuffer&) = delete;
    GlobalBuffer& operator=(const GlobalBuffer&) = delete;
    // 单例实例管理全局内存缓冲区，C++11 线程安全
    static GlobalBuffer& I() {
        static GlobalBuffer ctx;
        return ctx;
    }
    ~GlobalBuffer() {
        try_free();  // 释放已分配空间
    }
    void *pointer(size_t bytes) {
        try_malloc(bytes);
        return _buffer;
    }
    size_t bytes() { return _bytes; }
};

} // namespace buffer
