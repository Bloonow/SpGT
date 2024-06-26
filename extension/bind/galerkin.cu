#include <torch/extension.h>
#include "../native/multihead_dim32.cu"
#include "../native/matmul.cu"
#include "../native/datamove.cu"

namespace exts {
using Tensor = torch::Tensor;

Tensor mh32_projpos_rrc_forward(
    const Tensor &input, const Tensor &weight, const Tensor &bias, const Tensor &pos,
    const int seqlen, const int d_model, const int n_head, const int d_k, const int d_pos,
    const int batch
) {
    torch::TensorOptions opts = input.options();
    Tensor out = torch::empty({batch, n_head, d_pos + d_k, seqlen}, opts);
    SpGT::mh32::sgemm_rrc_cuda(
        input.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), 1.F, bias.data_ptr<float>(),
        seqlen, d_model, d_model, seqlen * d_model, 0, seqlen * n_head * (d_pos + d_k),
        NULL, NULL, NULL, 0.F, 0,
        pos.data_ptr<float>(), d_pos,
        batch
    );
    return std::move(out);
}

std::tuple<Tensor,Tensor,Tensor> mh32_projpos_rrc_backward(
    const Tensor &grad_out_, const Tensor &input, const Tensor &weight,
    const int seqlen, const int d_model, const int n_head, const int d_k, const int d_pos, 
    const int batch
) {
    // grad_out_ = [batch, n_head, d_pos + d_k, seqlen]
    // Tensor grad_out = grad_out_.slice(-2, d_pos).contiguous().view({batch, d_model, seqlen}).transpose(-2, -1).contiguous();
    Tensor grad_out = torch::empty({batch, seqlen, n_head, d_k}, input.options());
    SpGT::datamove::batched_gather_transpose_2D(
        grad_out_.contiguous().data_ptr<float>() + d_pos * seqlen,
        grad_out.data_ptr<float>(), n_head, d_pos + d_k, seqlen, d_k, batch
    );
    grad_out = grad_out.view({batch, seqlen, d_model});

    Tensor grad_input = torch::matmul(grad_out, weight.transpose(0, 1));
    Tensor grad_bias = torch::sum(grad_out, torch::IntArrayRef({0, 1}));
    Tensor grad_weight = torch::matmul(
        input.reshape({batch * seqlen, d_model}).transpose(0, 1),
        grad_out.reshape({batch * seqlen, d_model})
    );
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

std::tuple<Tensor,Tensor> mh32_projpos_lnorm_rrc_forward(
    const Tensor &input, const Tensor &weight, const Tensor &bias,
    const Tensor &lnw, const Tensor &lnb, const float norm_eps, 
    const Tensor &pos,
    const int seqlen, const int d_model, const int n_head, const int d_k, const int d_pos,
    const int batch
) {
    torch::TensorOptions opts = input.options();
    Tensor hat_out = torch::empty({batch, n_head, d_pos + d_k, seqlen}, opts);
    Tensor invsigma = torch::empty({batch, n_head, seqlen}, opts);
    SpGT::mh32::sgemm_rrc_cuda(
        input.data_ptr<float>(), weight.data_ptr<float>(), hat_out.data_ptr<float>(), 1.F, bias.data_ptr<float>(),
        seqlen, d_model, d_model, seqlen * d_model, 0, seqlen * n_head * (d_pos + d_k),
        lnw.data_ptr<float>(), lnb.data_ptr<float>(), invsigma.data_ptr<float>(), norm_eps, seqlen * n_head,
        pos.data_ptr<float>(), d_pos,
        batch
    );
    return std::make_tuple(std::move(hat_out), std::move(invsigma));
}

std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> mh32_projpos_lnorm_rrc_backward(
    const Tensor &grad_ln_, const Tensor &input, const Tensor &weight,
    const Tensor &hat_ln, const Tensor &invsigma, const Tensor &lnw, const Tensor &lnb,
    const int seqlen, const int d_model, const int n_head, const int d_k, const int d_pos,
    const int batch
) {
    Tensor grad_ln = grad_ln_.contiguous();
    torch::TensorOptions opts = input.options();
    Tensor grad_out_col = torch::empty({batch, d_model, seqlen}, opts);  // Without Position
    Tensor grad_out = torch::empty({batch, seqlen, d_model}, opts);      // Without Position
    Tensor grad_lnw = torch::empty({n_head, d_k}, opts);
    Tensor grad_lnb = torch::empty({n_head, d_k}, opts);
    SpGT::mh32::lnorm_grad_ccc_cuda(
        grad_ln.data_ptr<float>(), hat_ln.data_ptr<float>(), invsigma.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(),
        grad_out_col.data_ptr<float>(), grad_lnw.data_ptr<float>(), grad_lnb.data_ptr<float>(), 
        seqlen, d_model, seqlen * n_head * (d_pos + d_k), seqlen * n_head, seqlen * d_model, d_pos,
        batch
    );
    SpGT::datamove::batched_transpose_2D_cuda(
        grad_out_col.data_ptr<float>(), grad_out.data_ptr<float>(), d_model, seqlen, d_model * seqlen, batch
    );
    // grad_out = [batch, seqlen, d_model]
    Tensor grad_input = torch::matmul(grad_out, weight.transpose(0, 1));
    Tensor grad_bias = torch::sum(grad_out, torch::IntArrayRef({0, 1}));
    Tensor grad_weight = torch::matmul(
        input.reshape({batch * seqlen, d_model}).transpose(0, 1),
        grad_out.reshape({batch * seqlen, d_model})
    );
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias), std::move(grad_lnw), std::move(grad_lnb));
}

Tensor mh32_galattn_cccr_forward(
    const Tensor &Q, const Tensor &K, const Tensor &V,
    const int batch, const int n_head, const int d_posk, const int seqlen
) {
    torch::TensorOptions opts = Q.options();
    Tensor tmp = torch::empty({batch, n_head, d_posk, d_posk}, opts);
    Tensor output = torch::empty({batch, n_head, seqlen, d_posk}, opts);  // 输出 output 为行主序
    float *tmp_ptr = tmp.data_ptr<float>();

    SpGT::sgemm_32x32_4x8_SplitK::sgemm_cuda(
        K.data_ptr<float>(), V.data_ptr<float>(), tmp_ptr, (1.F / seqlen),
        d_posk, d_posk, seqlen, d_posk * seqlen, seqlen * d_posk, d_posk * d_posk,
        SpGT::Gemm_Order::RCC, batch * n_head
    );
    SpGT::sgemm_32x32_4x4::sgemm_cuda(
        Q.data_ptr<float>(), tmp_ptr, output.data_ptr<float>(), 1.F,
        seqlen, d_posk, d_posk, seqlen * d_posk, d_posk * d_posk, seqlen * d_posk,
        SpGT::Gemm_Order::CCR, batch * n_head
    );
    return std::move(output);
}

std::tuple<Tensor, Tensor, Tensor> mh32_galattn_cccr_backward(
    const Tensor &grad_out_, const Tensor &Q, const Tensor &K, const Tensor &V,
    const int batch, const int n_head, const int d_posk, const int seqlen
) {
    Tensor grad_out = grad_out_.contiguous();  // 梯度 grad_out 为行主序
    torch::TensorOptions opts = Q.options();
    Tensor tmp = torch::empty({batch, n_head, d_posk, d_posk}, opts);
    Tensor grad_Q = torch::empty({batch, n_head, d_posk, seqlen}, opts);  // 对 Q,K,V 的梯度为列主序
    Tensor grad_K = torch::empty({batch, n_head, d_posk, seqlen}, opts);  // 对 Q,K,V 的梯度为列主序
    Tensor grad_V = torch::empty({batch, n_head, d_posk, seqlen}, opts);  // 对 Q,K,V 的梯度为列主序
    float *grad_ptr = grad_out.data_ptr<float>();
    float *q_ptr = Q.data_ptr<float>();
    float *k_ptr = K.data_ptr<float>();
    float *v_ptr = V.data_ptr<float>();
    float *tmp_ptr = tmp.data_ptr<float>();

    SpGT::sgemm_32x32_4x8_SplitK::sgemm_cuda(
        v_ptr, k_ptr, tmp_ptr, (1.F / seqlen),
        d_posk, d_posk, seqlen, d_posk * seqlen, seqlen * d_posk, d_posk * d_posk,
        SpGT::Gemm_Order::RCC, batch * n_head
    );
    SpGT::sgemm_32x32_4x4::sgemm_cuda(
        grad_ptr, tmp_ptr, grad_Q.data_ptr<float>(), 1.F, 
        seqlen, d_posk, d_posk, seqlen * d_posk, d_posk * d_posk, seqlen * d_posk,
        SpGT::Gemm_Order::RCC, batch * n_head
    );
    SpGT::sgemm_32x32_4x8_SplitK::sgemm_cuda(
        grad_ptr, q_ptr, tmp_ptr, (1.F / seqlen),
        d_posk, d_posk, seqlen, d_posk * seqlen, seqlen * d_posk, d_posk * d_posk,
        SpGT::Gemm_Order::CCC, batch * n_head
    );
    SpGT::sgemm_32x32_4x4::sgemm_cuda(
        v_ptr, tmp_ptr, grad_K.data_ptr<float>(), 1.F, 
        seqlen, d_posk, d_posk, seqlen * d_posk, d_posk * d_posk, seqlen * d_posk,
        SpGT::Gemm_Order::CCC, batch * n_head
    );
    SpGT::sgemm_32x32_4x8_SplitK::sgemm_cuda(
        q_ptr, grad_ptr, tmp_ptr, (1.F / seqlen),
        d_posk, d_posk, seqlen, d_posk * seqlen, seqlen * d_posk, d_posk * d_posk,
        SpGT::Gemm_Order::RRC, batch * n_head
    );
    SpGT::sgemm_32x32_4x4::sgemm_cuda(
        k_ptr, tmp_ptr, grad_V.data_ptr<float>(), 1.F, 
        seqlen, d_posk, d_posk, seqlen * d_posk, d_posk * d_posk, seqlen * d_posk,
        SpGT::Gemm_Order::CCC, batch * n_head
    );
    return std::make_tuple(std::move(grad_Q), std::move(grad_K), std::move(grad_V));
}

Tensor batched_skinny_gemm(
    const Tensor &A, const Tensor &B, const float alpha
) {
    const int ndim = A.ndimension();
    torch::IntArrayRef shape = A.sizes();
    std::vector<int64_t> shape_vector;
    int batchCount = 1;
    for (int i = 0; i < ndim - 2; ++i) {
        batchCount = batchCount * shape[i];
        shape_vector.push_back(shape[i]);
    }
    const int M = shape[ndim - 2];
    const int K = shape[ndim - 1];
    const int N = B.size(ndim - 1);
    shape_vector.push_back(M);
    shape_vector.push_back(N);
    Tensor C = torch::empty(torch::makeArrayRef(shape_vector), A.options());

    SpGT::sgemm_32x32_4x8_SplitK::sgemm_cuda(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), alpha,
        M, N, K, M * K, K * N, M * N, SpGT::Gemm_Order::RRR, batchCount
    );
    return std::move(C);
}

} // namespace exts

PYBIND11_MODULE(galerkin, m) {
    m.def("mh32_projpos_rrc_forward", &exts::mh32_projpos_rrc_forward, "mh32_projpos_rrc_forward");
    m.def("mh32_projpos_rrc_backward", &exts::mh32_projpos_rrc_backward, "mh32_projpos_rrc_backward");
    m.def("mh32_projpos_lnorm_rrc_forward", &exts::mh32_projpos_lnorm_rrc_forward, "mh32_projpos_lnorm_rrc_forward");
    m.def("mh32_projpos_lnorm_rrc_backward", &exts::mh32_projpos_lnorm_rrc_backward, "mh32_projpos_lnorm_rrc_backward");
    m.def("mh32_galattn_cccr_forward", &exts::mh32_galattn_cccr_forward, "mh32_galattn_cccr_forward");
    m.def("mh32_galattn_cccr_backward", &exts::mh32_galattn_cccr_backward, "mh32_galattn_cccr_backward");
    m.def("batched_skinny_gemm", &exts::batched_skinny_gemm, "batched_skinny_gemm");
}
