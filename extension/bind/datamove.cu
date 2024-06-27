#include <vector>
#include <numeric>
#include <torch/extension.h>
#include "../native/datamove.cu"

using Tensor = torch::Tensor;

Tensor batched_transpose(
    const Tensor &src, const std::vector<int64_t> &batch_list,
    const std::vector<int64_t> &M_list, const std::vector<int64_t> &N_list
) {
    const int batchCount = std::accumulate(batch_list.begin(), batch_list.end(), 1, std::multiplies<>());
    const int M = std::accumulate(M_list.begin(), M_list.end(), 1, std::multiplies<>());
    const int N = std::accumulate(N_list.begin(), N_list.end(), 1, std::multiplies<>());
    std::vector<int64_t> shape_vector;
    shape_vector.insert(shape_vector.end(), batch_list.begin(), batch_list.end());
    shape_vector.insert(shape_vector.end(), N_list.begin(), N_list.end());
    shape_vector.insert(shape_vector.end(), M_list.begin(), M_list.end());
    Tensor out = torch::empty(torch::makeArrayRef(shape_vector), src.options());

    datamove::batched_transpose_cuda(src.data_ptr<float>(), out.data_ptr<float>(), M, N, M * N, batchCount);
    return std::move(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "batched_transpose");
}