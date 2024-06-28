from contextlib import contextmanager
import datetime
import math
import os
import sys
from time import time
import traceback
from typing import Callable

import numpy as np
import psutil
from termcolor import colored
import torch
import torch.backends.cudnn
import torch.utils.benchmark
import torch.distributed


def assign(val, default):
    assert default is not None
    return val if val is not None else default


def set_seed(sd, use_cudnn=True, printout=False) -> None:
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)

    torch.backends.cudnn.enabled = use_cudnn
    if use_cudnn:
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
    Print the time usage and memory usage.
    """
    time_0 = time()
    if verbose:
        proc = psutil.Process(os.getpid())
        memory_0 = proc.memory_info()[0] * 1.0 / 2 ** 30   # get rss (Resident Set Size) of Memory by GiB
        print(colored(f'{label}: Time start at {time_0:.2f}', color='blue'))
        print(colored(f'Local ram usage start at {memory_0:.2f} GiB', color='green'))
        try:
            yield    # yield to body of `with` statement
        finally:
            time_1 = time()
            memory_1 = proc.memory_info()[0] * 1.0 / 2 ** 30
            mem_delta = memory_1 - memory_0
            sign = '+' if mem_delta >= 0 else '-'
            print(colored(f'{label}: Time end at {time_1:.2f} (Elapse {time_1 - time_0:.4f} secs)', color='blue'))
            print(colored(
                f'Local ram usage end at {memory_1:.2f} GiB (Changed {sign}{math.fabs(mem_delta):.4} GiB)\n',
                color='green'
            ))
    else:
        yield
        print(colored(f'{label}: Elapse {time() - time_0:.4f} secs\n', color='blue'))


def get_resolution_size(r_large, r_small):
    """
    在 Encoder 前后分别进行两次插值，分别为两次降采样插值与两次升采样插值
    Data --> downsample x2 --> Encoder --> upsample x2 --> Decoder
    此处计算每次插值后的数据分辨率
    """
    factor = round(np.sqrt(r_small / r_large), ndigits=4)  # 保留 4 位小数
    last_digit = float(str(factor)[-1])    # 取最后一位小数
    factor = round(factor, ndigits=3)      # 保留 3 位小数
    if last_digit < 5:
        factor += 5.e-3
    factor = int(factor / 5.e-3 + 0.5) * 5.e-3
    r_mid = round(r_large * factor) - 1
    resolution_size = (
        (r_large, r_large), (r_mid, r_mid),
        (r_small, r_small),
        (r_mid, r_mid), (r_large, r_large)
    )
    return resolution_size


def get_num_params(model: torch.nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        # A complex is seen as two parameters
        num_params += p.numel() * (1 + p.is_complex())
    return num_params


def get_daytime_string():
    tn = datetime.datetime.now()
    time_str = '{:0>2}{:0>2}_{:0>2}{:0>2}'.format(tn.month, tn.day, tn.hour, tn.minute)
    return time_str


def caller_name():
    call = traceback.extract_stack()[-3].line
    return call[: call.rfind('(')]


def timize(func: Callable, desc: str, label: str, sublabel: str, min_run_time: float = 10.0, **kwargs):
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