#include <torch/extension.h>
#include "../native/datamove.cu"

namespace exts {
using Tensor = torch::Tensor;

Tensor batched_transpose_2D(
    const Tensor &src, 
    const std::vector<int64_t> &M_list, const std::vector<int64_t> &N_list, 
    const std::vector<int64_t> &batch_list
) {
    const int batchCount = std::reduce(batch_list.begin(), batch_list.end(), 1, std::multiplies<>());
    const int M = std::reduce(M_list.begin(), M_list.end(), 1, std::multiplies<>());
    const int N = std::reduce(N_list.begin(), N_list.end(), 1, std::multiplies<>());
    std::vector<int64_t> shape_vector;
    shape_vector.insert(shape_vector.end(), batch_list.begin(), batch_list.end());
    shape_vector.insert(shape_vector.end(), N_list.begin(), N_list.end());
    shape_vector.insert(shape_vector.end(), M_list.begin(), M_list.end());
    Tensor out = torch::empty(torch::makeArrayRef(shape_vector), src.options());

    SpGT::datamove::batched_transpose_2D_cuda(src.data_ptr<float>(), out.data_ptr<float>(), M, N, M * N, batchCount);
    return std::move(out);
}

std::tuple<Tensor,Tensor> gather_transpose_dual(
    const Tensor &src, const std::vector<int64_t> &M_list, 
    const int N0, const int N1, const int n0, const int n1, const int N1_offset1, const int N1_offset2
) {
    const int M = std::reduce(M_list.begin(), M_list.end(), 1, std::multiplies<>());
    std::vector<int64_t> shape_vector;
    shape_vector.push_back(n1); shape_vector.push_back(n0);
    shape_vector.insert(shape_vector.end(), M_list.begin(), M_list.end());
    Tensor out1 = torch::empty(torch::makeArrayRef(shape_vector), src.options());
    Tensor out2 = torch::empty(torch::makeArrayRef(shape_vector), src.options());
    cuComplex *src_ptr = reinterpret_cast<cuComplex*>(src.data_ptr());
    cuComplex *out1_ptr = reinterpret_cast<cuComplex*>(out1.data_ptr());
    cuComplex *out2_ptr = reinterpret_cast<cuComplex*>(out2.data_ptr());

    SpGT::datamove::gather_transpose_2D_cuda(src_ptr + N1_offset1 * N0, out1_ptr, M, N0, N1, n0, n1);
    SpGT::datamove::gather_transpose_2D_cuda(src_ptr + N1_offset2 * N0, out2_ptr, M, N0, N1, n0, n1);
    return std::make_tuple(std::move(out1), std::move(out2));
}

Tensor scatter_transpose_dual(
    const Tensor &src1, const Tensor &src2, const std::vector<int64_t> &M_list, 
    const int N0, const int N1, const int n0, const int n1, const int N1_offset1, const int N1_offset2
) {
    const int M = std::reduce(M_list.begin(), M_list.end(), 1, std::multiplies<>());
    std::vector<int64_t> shape_vector;
    shape_vector.insert(shape_vector.end(), M_list.begin(), M_list.end());
    shape_vector.push_back(N1); shape_vector.push_back(N0);
    Tensor out = torch::zeros(torch::makeArrayRef(shape_vector), src1.options());
    cuComplex *src1_ptr = reinterpret_cast<cuComplex*>(src1.data_ptr());
    cuComplex *src2_ptr = reinterpret_cast<cuComplex*>(src2.data_ptr());
    cuComplex *out_ptr = reinterpret_cast<cuComplex*>(out.data_ptr());

    SpGT::datamove::scatter_transpose_2D_cuda(src1_ptr, out_ptr + N1_offset1 * N0, M, N0, N1, n0, n1);
    SpGT::datamove::scatter_transpose_2D_cuda(src2_ptr, out_ptr + N1_offset2 * N0, M, N0, N1, n0, n1);
    return std::move(out);
}

} // namespace exts

PYBIND11_MODULE(dmove, m) {
    m.def("batched_transpose_2D", &exts::batched_transpose_2D, "batched_transpose_2D");
    m.def("gather_transpose_dual", &exts::gather_transpose_dual, "gather_transpose_dual");
    m.def("scatter_transpose_dual", &exts::scatter_transpose_dual, "scatter_transpose_dual");
}

