import datetime
import os
import math
import sys
from time import time
from contextlib import contextmanager
import traceback
from typing import Callable

import psutil
import numpy as np
import torch
import torch.utils.benchmark
import torch.distributed
from termcolor import colored


def set_seed(sd, cudnn_fix=True, printout=False) -> None:
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)
    if cudnn_fix:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if printout:
        message = f'''
        os.environ['PYTHONHASHSEED'] = str({sd})
        np.random.seed({sd})
        torch.manual_seed({sd})
        torch.cuda.manual_seed({sd})
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all({sd})
        '''
        print('\nThe following code snippets have been run.')
        print('=' * 64, message, '=' * 64, sep='\n')


@contextmanager
def timer(label: str, verbose=True):
    """
    1. the time the code block takes to run.
    2. the memory usage.
    """
    start_time = time()
    if verbose:
        proc = psutil.Process(os.getpid())
        memory_0 = proc.memory_info()[0] * 1.0 / 2 ** 30   # get rss (Resident Set Size) of Memory by GiB
        print(colored(f'{label}: Time start at {start_time:.2f}', color='blue'))
        print(colored(f'Local ram usage start at {memory_0:.2f} GiB', color='green'))
        try:
            yield    # yield to body of `with` statement
        finally:
            end_time = time()
            memory_1 = proc.memory_info()[0] * 1.0 / 2 ** 30
            mem_delta = memory_1 - memory_0
            sign = '+' if mem_delta >= 0 else '-'
            print(colored(f'{label}: Time end at {end_time:.2f} (Elapse {end_time - start_time:.4f} secs)', color='blue'))
            print(colored(f'Local ram usage end at {memory_1:.2f} GiB (Changed {sign}{math.fabs(mem_delta):.4} GiB)\n', color='green'))
    else:
        yield
        print(colored(f'{label}: Elapse {time() - start_time:.4f} secs\n', color='blue'))


def pooling_2D(matrix: np.ndarray, kernel_size: tuple = (2, 2), method='mean', padding=False) -> np.ndarray:
    """
    2D数据非重叠的池化操作 matrix = [M, N] or [batch, M, N]
    假设 x,y 轴对应于 [y,x] 维度
    """
    m, n = matrix.shape[-2:]
    ky, kx = kernel_size

    if padding:
        ny = int(np.ceil(m / float(ky)))
        nx = int(np.ceil(n / float(kx)))
        pad_size = matrix.shape[:-2] + (ny * ky, nx * kx)
        sy_0 = (ny * ky - m) // 2
        sx_0 = (nx * kx - n) // 2
        sy_1 = ny * ky - m - sy_0
        sx_1 = nx * kx - n - sx_0
        matrix_pad = np.full(pad_size, np.nan)
        matrix_pad[..., sy_0:-sy_1, sx_0:-sx_1] = matrix
    else:
        ny = m // ky
        nx = n // kx
        matrix_pad = matrix[..., :ny * ky, :nx * kx]

    re_shape = matrix.shape[:-2] + (ny, ky, nx, kx)
    if method == 'max':
        result = np.nanmax(matrix_pad.reshape(re_shape), axis=(-3, -1))
    elif method == 'mean':
        result = np.nanmean(matrix_pad.reshape(re_shape), axis=(-3, -1))
    else:
        raise NotImplementedError(f'pooling_2D do not support {method}.')

    return result


def get_down_up_size(r_large, r_small):
    """
    在 Encoder 前后分别进行两次插值，分别为两次降采样插值与两次升采样插值
    Data --> downsample x2 --> Encoders --> upsample x2 --> Decoders
    此处计算每次插值后的数据分辨率
    """
    factor = round(np.sqrt(r_small / r_large), ndigits=4)  # 保留 4 位小数
    last_digit = float(str(factor)[-1])    # 取最后一位小数
    factor = round(factor, ndigits=3)      # 保留 3 位小数
    if last_digit < 5:
        factor += 5.e-3
    factor = int(factor / 5.e-3 + 0.5) * 5.e-3
    r_in = round(r_large * factor) - 1
    down_size = ((r_in, r_in), (r_small, r_small))  # 两次降采样插值后的分辨率
    up_size = ((r_in, r_in), (r_large, r_large))    # 两次升采样插值后的分辨率
    return down_size, up_size


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        num_params += p.numel() * (1 + p.is_complex())
    return num_params


def get_daytime_string():
    tn = datetime.datetime.now()
    time_str = '{:0>2}{:0>2}_{:0>2}{:0>2}'.format(tn.month, tn.day, tn.hour, tn.minute)
    return time_str


class UnitGaussianNormalizer:
    def __init__(self, mean=0., std=1., eps=1.e-5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit_transform(self, X: np.ndarray):
        """
        X         : [N, r, r, 1]
        self.mean : [r, r, 1]
        self.std  : [r, r, 1]
        """
        # 用于训练集的构建
        self.mean = X.mean(0)
        self.std = X.std(0)
        return (X - self.mean) / (self.std + self.eps)

    def transform(self, X: np.ndarray):
        # 用于验证集的构建
        return (X - self.mean) / (self.std + self.eps)

    def inverse_transform(self, X: torch.Tensor):
        # 用于训练过程
        return (X * (self.std + self.eps)) + self.mean

    def to(self, device='cpu'):
        if not torch.is_tensor(self.mean):
            self.mean = torch.from_numpy(self.mean).float()
        if not torch.is_tensor(self.std):
            self.std = torch.from_numpy(self.std).float()
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def timize(func: Callable, desc: str, label: str, sublabel: str, min_run_time: float = 10., **kwargs):
    func(**kwargs)  # warmup
    return torch.utils.benchmark.Timer(
        stmt='func(**kwargs)',
        globals={
            '__name__': __name__, 'func': func, 'kwargs': kwargs
        },
        label=label,
        sub_label=sublabel,
        description=desc
    ).blocked_autorange(min_run_time=min_run_time)


def caller_name():
    call = traceback.extract_stack()[-3].line
    return call[: call.rfind('(')]


def distributed_initialize():
    torch.distributed.init_process_group('nccl', init_method='env://')
    if not torch.distributed.is_initialized():
        print('Error: torch.distributed.is_initialized == False')
        sys.exit()
    world_size = torch.distributed.get_world_size()
    my_rank = torch.distributed.get_rank()
    if my_rank != 0:
        sys.stdout, sys.stderr = None, None
    print(f'======== world_size: {world_size} ========')
    torch.cuda.set_device(my_rank)


def is_main_process():
    return torch.distributed.get_rank() == 0
