import os
from typing import Union
import numpy as np
import torch
import torch.utils.data.dataset

from SpGT.common.trivial import assign, set_seed, timer
from SpGT.dataset.data_accessor import read_matfile


class GaussNormalizer:
    def __init__(self, X: Union[np.ndarray, torch.Tensor] = None, eps: float = 1.e-5):
        self.eps = eps
        self.fit(X=X, axis=0)

    def fit(self, X: Union[np.ndarray, torch.Tensor], axis=0):
        if isinstance(X, np.ndarray):
            self.mean = np.mean(X, axis=axis)
            self.std = np.std(X, axis=axis)
        elif isinstance(X, torch.Tensor):
            self.mean = torch.mean(X, dim=axis)
            self.std = torch.std(X, dim=axis)
        else:
            self.mean = np.array([0.0,])
            self.std = np.array([1.0,])

    def transform(self, X: Union[np.ndarray, torch.Tensor]):
        return (X - self.mean) / (self.std + self.eps)

    def inverse_transform(self, X: Union[np.ndarray, torch.Tensor]):
        return X * (self.std + self.eps) + self.mean

    def numpy_to_torch(self, device='cpu'):
        if isinstance(self.mean, np.ndarray):
            self.mean = torch.from_numpy(self.mean).float().to(device=device)
        if isinstance(self.std, np.ndarray):
            self.std = torch.from_numpy(self.std).float().to(device=device)
        return self

    def torch_to_numpy(self):
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.cpu().numpy()
        if isinstance(self.std, torch.Tensor):
            self.std = self.std.cpu().numpy()
        return self


class DarcyDataset(torch.utils.data.dataset.Dataset):
    """
    原始数据格式为 [N, R, R]，其中，N 表示数据样本个数，R 表示数据的完整分辨率，使用 r 表示降采样后分辨率
    """

    def __init__(
        self, datapath, subsample_node, subsample_attn=None, num_data=None, fine_resolution=None,
        is_training=True, is_return_boundary=True, is_normalization=True, node_normalizer: GaussNormalizer = None,
        noise=0.0, random_seed=1127802
    ) -> None:
        super().__init__()
        self.is_training = is_training
        self.datapath = datapath
        self.num_data = num_data                # 样本数据的个数
        self.fine_resolution = fine_resolution  # 原始的最高分辨率
        self.subsample_node = subsample_node    # 对数据进行降采样
        self.subsample_attn = assign(subsample_attn, 2 * subsample_node)
        self.subsample_method = 'nearest'       # 将采样方法，现支持 nearest
        self.is_return_boundary = is_return_boundary  # 是否保留数据边界
        self.is_normalization = is_normalization      # 是否对数据进行正则化
        self.node_normalizer = node_normalizer        # 数据点的正则化器
        self.noise = noise                            # 噪声
        self.random_seed = random_seed                # 随机化种子
        self._initialize()

    def _initialize(self):
        set_seed(self.random_seed, printout=False)
        with timer(f'Load {self.datapath.split(os.path.sep)[-1]}'):
            # For generated Darcy dataset, the key name is ['a', 'u', 'a_mean', 'u_mean', 'a_std', 'u_std']
            a, u = read_matfile(self.datapath, ['a', 'u'], [np.float32, np.float32])
            # For downloaded Darcy dataset, the key name is ['Kcoeff', 'Kcoeff_x', 'Kcoeff_y', 'coeff', 'sol']
            # a, u = read_matfile(self.datapath, ['coeff', 'sol'], [np.float32, np.float32])
        N, R, R = a.shape
        self.num_data = assign(self.num_data, N)
        self.fine_resolution = assign(self.fine_resolution, R)
        if self.is_training:
            a, u = a[:self.num_data], u[:self.num_data]
        else:
            a, u = a[-self.num_data:], u[-self.num_data:]

        # 降采样，获得Darcy问题的坐标、数据点
        grid = DarcyDataset._meshgrid(S=R, sub=self.subsample_node, is_return_boundary=self.is_return_boundary)
        position = DarcyDataset._meshgrid(S=R, sub=self.subsample_attn, is_return_boundary=True)
        nodes, targets, targets_grad = DarcyDataset._subsample_data(a, u, self.subsample_node, self.subsample_method)
        coeffs = np.copy(nodes)  # 未经变换的数据点作为Darcy问题的系数

        # 对数据进行正则化
        if self.is_normalization:
            if self.is_training:
                self.node_normalizer = GaussNormalizer(nodes)
                self.target_normalizer = GaussNormalizer(
                    targets if self.is_return_boundary else targets[:, 1:-1, 1:-1, :]
                )
                nodes = self.node_normalizer.transform(nodes)
            else:
                assert self.node_normalizer != None, "node_normalizer can not be None when inference"
                nodes = self.node_normalizer.transform(nodes)

        # 添加噪声
        if self.noise > 0.0:
            nodes += self.noise * np.random.randn(*nodes.shape)

        self.grid = grid
        self.position = position
        self.coeffs = coeffs
        self.nodes = nodes
        self.targets = targets
        self.targets_grad = targets_grad

    @staticmethod
    def _meshgrid(S: int, sub: int, is_return_boundary: bool) -> np.ndarray:
        # S 表示数据 X 的一个维度上数据点的数目，即 X = [N, S, S]
        # 假设 x,y 轴对应于 [x,y] 维度
        x, y = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
        x = x[::sub, ::sub]
        y = y[::sub, ::sub]
        if not is_return_boundary:
            x = x[1:-1, 1:-1]
            y = y[1:-1, 1:-1]
        return np.stack([x, y], axis=-1)

    @staticmethod
    def _subsample_data(a: np.ndarray, u: np.ndarray, sub: int, method: str = 'nearest') -> list[np.ndarray]:
        # 解析原始最高分辨的数据 a, u = [N, R, R] 并进行降采样
        N, R, R = a.shape
        r = int((R - 1) / sub + 1)
        if sub >= 1 and method == 'nearest':
            nodes = a[:, ::sub, ::sub].reshape(N, r, r, 1)
            targets = u[:, ::sub, ::sub].reshape(N, r, r, 1)
            targets_gradx, targets_grady = DarcyDataset._central_diff(u, R)
            targets_gradx = targets_gradx[:, ::sub, ::sub]
            targets_grady = targets_grady[:, ::sub, ::sub]
            targets_grad = np.stack([targets_gradx, targets_grady], axis=-1).reshape(N, r, r, 2)
        else:
            raise NotImplementedError(f'subsample method {method} has not been implemented')
        return nodes, targets, targets_grad

    @staticmethod
    def _central_diff(X: np.ndarray, S: int, padding: bool = True) -> list[np.ndarray]:
        # 中心差分，S 表示数据 X 的一个维度上数据点的数目，即 X = [N, S, S]
        if padding:
            X = np.pad(X, pad_width=[(0, 0), (1, 1), (1, 1)], mode='constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (X[:, d:, s:-s] - X[:, :-d, s:-s]) / d
        grad_y = (X[:, s:-s, d:] - X[:, s:-s, :-d]) / d
        return grad_x * S, grad_y * S

    def __len__(self):
        return self.num_data

    def __getitem__(self, index) -> dict:
        """
        grid        : [r, r, 2]            原始最高分辨时，直接索引获得的位置编码
        position    : [r_attn, r_attn, 2]  降采样后数据点的位置编码
        coeff       : [r, r, 1]            未经过变换的原始的方程系数
        node        : [r, r, 1]            规则化后的方程系数
        target      : [r, r, 1]            方程的解
        target_grad : [r, r, 2]            方程的解的梯度
        """
        return dict(
            grid=torch.from_numpy(self.grid).float(),
            position=torch.from_numpy(self.position).float(),
            coeff=torch.from_numpy(self.coeffs[index]).float(),
            node=torch.from_numpy(self.nodes[index]).float(),
            target=torch.from_numpy(self.targets[index]).float(),
            target_grad=torch.from_numpy(self.targets_grad[index]).float()
        )


def get_guass_normalizer(datapath: str, sub: int) -> list[GaussNormalizer]:
    with timer(f'Load {datapath.split(os.path.sep)[-1]}'):
        # For generated Darcy dataset, the key name is ['a', 'u', 'a_mean', 'u_mean', 'a_std', 'u_std']
        a, u = read_matfile(datapath, ['a', 'u'], [np.float32, np.float32])
        # For downloaded Darcy dataset, the key name is ['Kcoeff', 'Kcoeff_x', 'Kcoeff_y', 'coeff', 'sol']
        # a, u = read_matfile(self.datapath, ['coeff', 'sol'], [np.float32, np.float32])
    N, R, R = a.shape
    r = int((R - 1) / sub + 1)
    nodes = a[:, ::sub, ::sub].reshape(N, r, r, 1)
    targets = u[:, ::sub, ::sub].reshape(N, r, r, 1)
    node_normalizer = GaussNormalizer(nodes)
    target_normalizer = GaussNormalizer(targets)
    return node_normalizer, target_normalizer
