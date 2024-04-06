import gc
import numpy as np


def read_matfile(
    filepath: str, key_list: list[str], dtype_list: list[np.dtype]
) -> list[np.ndarray]:
    value_list = []
    try:
        # 旧格式存储
        from scipy.io import loadmat
        data = loadmat(filepath)
        for key, dtype in zip(key_list, dtype_list):
            val: np.ndarray = data[key]
            value_list.append(val.astype(dtype))
    except NotImplementedError:
        # 新格式存储 -v7.3
        from h5py import File
        data = File(filepath)
        for key, dtype in zip(key_list, dtype_list):
            val: np.ndarray = data[key][()]
            val = np.transpose(val, axes=range(len(val.shape) - 1, -1, -1))
            value_list.append(val.astype(dtype))
    finally:
        del data
        gc.collect()
    return value_list


def write_matfile(filepath: str, object_dict) -> None:
    from scipy.io import savemat
    savemat(filepath, object_dict)
