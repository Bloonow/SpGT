import gc
import numpy as np
from h5py import File
from scipy.io import loadmat


def read_matfile_old(filepath: str, key_list: list[str], dtype_list: list[np.dtype]) -> list[np.ndarray]:
    value_list = []
    # 旧格式存储 v4, v6, v7, v7.2
    data = loadmat(filepath)
    for key, dtype in zip(key_list, dtype_list):
        val: np.ndarray = data[key]
        value_list.append(val.astype(dtype))
    del data
    gc.collect()
    return value_list


def read_matfile_v73(filepath: str, key_list: list[str], dtype_list: list[np.dtype]) -> list[np.ndarray]:
    value_list = []
    # 新格式存储 v7.3
    data = File(filepath)
    for key, dtype in zip(key_list, dtype_list):
        val: np.ndarray = data[key][()]  # 在此处真正加载数据
        val = np.transpose(val, axes=range(len(val.shape) - 1, -1, -1))
        value_list.append(val.astype(dtype))
    del data
    gc.collect()
    return value_list


def read_matfile(filepath: str, key_list: list[str], dtype_list: list[np.dtype]) -> list[np.ndarray]:
    try:
        value_list = read_matfile_old(filepath, key_list, dtype_list)
    except NotImplementedError as ex:
        value_list = read_matfile_v73(filepath, key_list, dtype_list)
    finally:
        return value_list
