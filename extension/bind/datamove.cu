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

std::tuple<Tensor,Tensor> transpose_gather_dual(
    const Tensor &src, const std::vector<int64_t> &dim_list,
    const int H_all, const int W_all, const int H_low, const int W_low,
    const int H_base_1, const int W_base_1, const int H_base_2, const int W_base_2
) {
    const int dim = std::accumulate(dim_list.begin(), dim_list.end(), 1, std::multiplies<>());
    std::vector<int64_t> shape_vector;
    shape_vector.push_back(H_low);
    shape_vector.push_back(W_low);
    shape_vector.insert(shape_vector.end(), dim_list.begin(), dim_list.end());
    Tensor out1 = torch::empty(torch::makeArrayRef(shape_vector), src.options());
    Tensor out2 = torch::empty(torch::makeArrayRef(shape_vector), src.options());
    cuComplex *src_ptr = reinterpret_cast<cuComplex*>(src.data_ptr());
    cuComplex *out1_ptr = reinterpret_cast<cuComplex*>(out1.data_ptr());
    cuComplex *out2_ptr = reinterpret_cast<cuComplex*>(out2.data_ptr());

    datamove::transpose_gather_cuda(src_ptr + H_base_1 * W_all + W_base_1, out1_ptr, dim, H_all, W_all, H_low, W_low);
    datamove::transpose_gather_cuda(src_ptr + H_base_2 * W_all + W_base_2, out2_ptr, dim, H_all, W_all, H_low, W_low);
    return std::make_tuple(std::move(out1), std::move(out2));
}

Tensor transpose_scatter_dual(
    const Tensor &src1, const Tensor &src2, const std::vector<int64_t> &dim_list,
    const int H_all, const int W_all, const int H_low, const int W_low,
    const int H_base_1, const int W_base_1, const int H_base_2, const int W_base_2
) {
    const int dim = std::accumulate(dim_list.begin(), dim_list.end(), 1, std::multiplies<>());
    std::vector<int64_t> shape_vector;
    shape_vector.insert(shape_vector.end(), dim_list.begin(), dim_list.end());
    shape_vector.push_back(H_all);
    shape_vector.push_back(W_all);
    Tensor out = torch::zeros(torch::makeArrayRef(shape_vector), src1.options());
    cuComplex *src1_ptr = reinterpret_cast<cuComplex*>(src1.data_ptr());
    cuComplex *src2_ptr = reinterpret_cast<cuComplex*>(src2.data_ptr());
    cuComplex *out_ptr = reinterpret_cast<cuComplex*>(out.data_ptr());

    datamove::transpose_scatter_cuda(src1_ptr, out_ptr + H_base_1 * W_all + W_base_1, dim, H_all, W_all, H_low, W_low);
    datamove::transpose_scatter_cuda(src2_ptr, out_ptr + H_base_2 * W_all + W_base_2, dim, H_all, W_all, H_low, W_low);
    return std::move(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "batched_transpose");
    m.def("transpose_gather_dual", &transpose_gather_dual, "transpose_gather_dual");
    m.def("transpose_scatter_dual", &transpose_scatter_dual, "transpose_scatter_dual");
}