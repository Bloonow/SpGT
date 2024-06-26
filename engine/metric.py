import torch
from torch import Tensor


class WeightedL2Loss2D(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self, S, dim=2, dilation=2, alpha=0.0, beta=1.0, gamma=0.1, delta=0.0, eps=1.e-10,
        is_regularization=False, is_normalization=True, metric_reduction='L1'
    ) -> None:
        super().__init__()
        # 原始数据格式为 [N, S, S]，其中，S 表示一个维度上数据点的数目，则 H 可表示两个点之间的步长
        self.S = S
        self.H = 1.0 / S
        self.dim = dim
        self.dilation = dilation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta * self.H ** self.dim
        self.eps = eps
        self.is_regularization = is_regularization
        self.is_normalization = is_normalization
        self.metric_reduction = metric_reduction

    def _central_diff(self, X: Tensor, S: int):
        # S 表示数据 X 的一个维度上数据点的数目，即 X = [N, S, S]
        # dilation, stride = 2, 1
        d, s = self.dilation, self.dilation // 2
        grad_x = (X[:, d:, s:-s] - X[:, :-d, s:-s]) / d
        grad_y = (X[:, s:-s, d:] - X[:, s:-s, :-d]) / d
        grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad * S

    def forward(
        self, pred: Tensor, target: Tensor,
        pred_grad: Tensor = None, target_grad: Tensor = None, coeff: Tensor = None
    ):
        """
        pred        : [N, r, r, 1]
        target      : [N, r, r, 1]
        pred_grad   : [N, r, r, 2]
        target_grad : [N, r, r, 2]
        coeff       : [N, r, r, 1]
        Outputs     : loss, regularizer, metric
        loss        = alpha * loss_grad + beta * loss_pred
        """
        pred, target = pred.squeeze(dim=-1), target.squeeze(dim=-1)
        target_norm = target.pow(2).mean(dim=[1, 2]) + self.eps
        if target_grad is not None:
            target_grad_norm = self.dim * (coeff * target_grad.pow(2)).mean(dim=[1, 2, 3]) + self.eps
        else:
            target_grad_norm = 1

        loss = self.beta * ((pred - target).pow(2)).mean(dim=[1, 2]) / target_norm
        if self.alpha > 0 and pred_grad is not None:
            pred_grad_diff = (coeff * (pred_grad - target_grad)).pow(2)
            loss_grad = self.alpha * pred_grad_diff.mean(dim=[1, 2, 3]) / target_grad_norm
            loss = loss + loss_grad

        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt()
        elif self.metric_reduction == 'L1':
            metric = loss.sqrt().mean()
        elif self.metric_reduction == 'Linf':
            metric = loss.sqrt().max()

        loss = loss.sqrt().mean() if self.is_normalization else loss.mean()

        if self.is_regularization and target_grad is not None:
            pred_diff = self._central_diff(pred, self.S)
            s = self.dilation // 2
            target_grad = target_grad[:, s:-s, s:-s, :].contiguous()
            coeff = coeff[:, s:-s, s:-s].contiguous()
            regularizer = self.gamma * self.H * (
                (coeff * (target_grad - pred_diff)).pow(2)
            ).mean(dim=[1, 2, 3]) / target_grad_norm
            regularizer = regularizer.sqrt().mean() if self.is_normalization else regularizer.mean()
        else:
            regularizer = torch.tensor([0.], requires_grad=loss.requires_grad, device=loss.device)

        return loss, regularizer, metric
