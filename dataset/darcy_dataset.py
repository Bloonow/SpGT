import os
from typing import Union
import numpy as np
import torch
import torch.utils.data.dataset

from SpGT.common.trivial import UnitGaussianNormalizer, pooling_2D, set_seed, timer
from SpGT.dataset.matio import read_matfile


class DarcyDataset(torch.utils.data.dataset.Dataset):
    """
    原始数据格式为 [N, R, R]，其中，N 表示数据样本个数，R 表示数据的完整分辨率，使用 r 表示降采样后分辨率
    假设 x,y 轴对应于 [x,y] 维度
    """

    def __init__(
        self, datapath, num_data, fine_resolution, subsample_node, subsample_attn,
        is_training, is_inverse_problem=False, subsample_inverse=1,
        is_return_boundary=True, is_normalization=True, sample_normalizer=None,
        noise=0, random_seed=1127802
    ) -> None:
        super().__init__()
        self.datapath = datapath
        self.num_data = num_data
        self.fine_resolution = fine_resolution  # 原始的最高分辨率
        self.subsample_node = subsample_node    # 对数据进行降采样，每 subsample_node 个数据点降采样为 1 个数据点
        self.subsample_attn = subsample_attn
        self.is_training = is_training
        self.is_inverse_problem = is_inverse_problem  # 是否为达西渗流方程的逆问题
        self.subsample_inverse = subsample_inverse    # 对数据进行降采样，用于逆问题时的情况
        self.subsample_method = 'average' if is_inverse_problem else 'nearest'
        self.is_return_boundary = is_return_boundary
        self.is_normalization = is_normalization      # 是否对数据进行规则化
        self.sample_normalizer = sample_normalizer
        self.noise = noise
        self.random_seed = random_seed
        self._initialize()

    def _initialize(self):
        set_seed(self.random_seed, printout=False)
        with timer(f'Load {self.datapath.split(os.path.sep)[-1]}'):
            a, u = read_matfile(self.datapath, ['a', 'u'], [np.uint8, np.float32])
        N, R, R = a.shape
        assert R == self.fine_resolution, 'Error Data Rresolution'
        assert N >= self.num_data, 'Error Data Numbers'
        a = a[:self.num_data, ...]
        u = u[:self.num_data, ...]

        # 是否为达西渗流方程的逆问题
        if self.is_inverse_problem:
            a, u = u, a

        # 若用于达西渗流方程的逆问题，则该处的维度存在问题，待解决
        sub = self.subsample_node
        sub_attn = self.subsample_attn
        # 获取降采样后的再次降采样用于 Attention 的 pos 与 elem
        self.pos, _ = self._meshgrid(S=R, sub=sub_attn, is_return_element=True, is_return_boundary=True)
        # 获取降采样后的 grid，用于 Spectral Convolution Regressor 与 X 拼接
        self.grid = self._meshgrid(S=R, sub=sub, is_return_element=False, is_return_boundary=self.is_return_boundary)

        # coeffs 为未经过变换的原始的方程系数
        nodes, targets, targets_grad = self._subsample_data(a, u)
        self.coeffs = nodes

        # 对数据进行规则化
        if self.is_training and self.is_normalization:
            self.sample_normalizer = UnitGaussianNormalizer()
            self.target_normalizer = UnitGaussianNormalizer()
            nodes = self.sample_normalizer.fit_transform(nodes)
            if self.is_return_boundary:
                _ = self.target_normalizer.fit_transform(targets)
            else:
                _ = self.target_normalizer.fit_transform(targets[:, 1:-1, 1:-1, :])
        elif self.is_normalization:
            nodes = self.sample_normalizer.transform(nodes)

        if self.noise > 0:
            nodes += self.noise * np.random.randn(*nodes.shape)

        self.nodes = nodes
        self.targets = targets
        self.targets_grad = targets_grad

    def _subsample_data(self, a: np.ndarray, u: np.ndarray) -> list[np.ndarray]:
        # 解析原始最高分辨的数据 a, u = [N, R, R] 并进行降采样
        N, R, R = a.shape
        sub = self.subsample_node
        r = int((R - 1) / sub + 1)

        # 真实值
        if not self.is_inverse_problem:
            targets_gradx, targets_grady = self._central_diff(u, R)
            targets_gradx = targets_gradx[:, ::sub, ::sub]
            targets_grady = targets_grady[:, ::sub, ::sub]
            targets_grad = np.stack([targets_gradx, targets_grady], axis=-1).reshape(N, r, r, 2)
        else:
            targets_grad = np.zeros([N, 1, 1, 2])
        targets = u[:, ::sub, ::sub].reshape(N, r, r, 1)

        # 降采样方法
        if sub >= 1 and self.subsample_method == 'nearest':
            coeffs = a[:, ::sub, ::sub].reshape(N, r, r, 1)
        elif sub >= 1 and self.subsample_method == 'average':
            coeffs = pooling_2D(a, kernel_size=(sub, sub), padding=True).reshape(N, r, r, 1)

        return coeffs, targets, targets_grad

    @staticmethod
    def _central_diff(X: np.ndarray, S: int, padding=True) -> list[np.ndarray]:
        # 中心差分
        # S 表示数据 X 的一个维度上数据点的数目，即 X = [N, S, S]
        if padding:
            X = np.pad(X, [(0, 0), (1, 1), (1, 1)], mode='constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (X[:, d:,   s:-s] - X[:, :-d, s:-s]) / d
        grad_y = (X[:, s:-s, d:] - X[:, s:-s, :-d]) / d
        return grad_x * S, grad_y * S

    @staticmethod
    def _meshgrid(S, sub=1, is_return_element=True, is_return_boundary=True) -> Union[list[np.ndarray], np.ndarray]:
        # S 表示数据 X 的一个维度上数据点的数目，即 X = [N, S, S]
        # 假设 x,y 轴对应于 [x,y] 维度
        nx = ny = S
        x, y = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
        x = x[::sub, ::sub]
        y = y[::sub, ::sub]
        # grid = np.stack([x.ravel(), y.ravel()], axis=-1)
        grid = np.stack([x, y], axis=-1)

        if is_return_element:
            elem = []
            for j in range(ny - 1):
                for i in range(nx - 1):
                    lu = j * nx + i
                    ru = j * nx + (i+1)
                    lb = (j+1) * nx + i
                    rb = (j+1) * nx + (i+1)
                    elem += [[lu, rb, lb], [ru, rb, lu]]
            elem = np.asarray(elem, dtype=np.int32)
            return grid, elem
        else:
            if not is_return_boundary:
                x = x[1:-1, 1:-1]
                y = y[1:-1, 1:-1]
            return grid

    def __len__(self):
        return self.num_data

    def __getitem__(self, index) -> dict:
        """
        coeff       : [r, r]               未经过变换的原始的方程系数
        node        : [r, r]               规则化后的方程系数
        pos         : [r_attn, r_attn, 2]  降采样后数据点的位置编码
        grid        : [r, r, 2]            原始最高分辨时，直接索引获得的位置编码
        target      : [r, r]               方程的解
        target_grad : [r, r, 2]            方程的解的梯度
        """
        return dict(
            coeff=torch.from_numpy(self.coeffs[index]).float(),
            node=torch.from_numpy(self.nodes[index]).float(),
            pos=torch.from_numpy(self.pos).float(),
            grid=torch.from_numpy(self.grid).float(),
            target=torch.from_numpy(self.targets[index]).float(),
            target_grad=torch.from_numpy(self.targets_grad[index]).float()
        )
