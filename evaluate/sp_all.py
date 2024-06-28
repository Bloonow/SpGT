from SpGT.evaluate.sp_darcy_ablation import time_darcy_ablation_wrt_batch, time_darcy_ablation_wrt_resolution
from SpGT.evaluate.sp_darcy_breakdown import time_darcy_breakdown_wrt_batch, time_darcy_breakdown_wrt_resolution
from SpGT.evaluate.sp_fno2d import time_fno2d_wrt_batch, time_fno2d_wrt_resolution
from SpGT.evaluate.sp_galerkin import time_projpos_lnorm_wrt_batch, time_projpos_lnorm_wrt_resolution
from SpGT.evaluate.sp_galerkin import time_skinny_gemm_wrt_batch, time_skinny_gemm_wrt_resolution


def time_all():
    # 以下列表中的取值经过测试，最大时为刚好不会爆显存的情况
    time_skinny_gemm_wrt_resolution(resolution_list=[32, 64, 128, 256, 512],)
    time_skinny_gemm_wrt_batch(batch_list=[2, 4, 8, 16, 32, 64, 128, 256,],)
    time_projpos_lnorm_wrt_resolution(resolution_list=[32, 64, 128, 256, 512,],)
    time_projpos_lnorm_wrt_batch(batch_list=[2, 4, 8, 16, 32, 64,],)
    time_fno2d_wrt_resolution(resolution_list=[32, 64, 128, 256, 512,],)
    time_fno2d_wrt_batch(batch_list=[2, 4, 8, 16, 32, 64,],)
    time_darcy_ablation_wrt_resolution(resolution_list=[32, 64, 128, 256, 512,],)
    time_darcy_ablation_wrt_batch(batch_list=[2, 4, 8, 16, 32, 64,],)
    time_darcy_breakdown_wrt_resolution(resolution_list=[32, 64, 128, 256, 512,],)
    time_darcy_breakdown_wrt_batch(batch_list=[2, 4, 8, 16, 32, 64,],)


if __name__ == '__main__':
    time_all()
