% 在问题域 [0,1]^2 上对高斯随机场进行采样，采样点为 S 个
% 参数 alpha, tau 控制平滑性，alpha, tau 越大，函数越平滑，要求 alpha > d / 2，此处 d = 2
% 期望
%   mean = 0
% 协方差 (covariance operator)
%   C = (-Delta + tau^2)^{-alpha}
%     = tau^{2*alpha} * (-Laplacian + tau^2 I)^{-alpha}
% 其中，Delta 采用拉普拉斯算子 Laplacian，具有齐次诺伊曼边界

function U = Gaussian_Random_Field(alpha, tau, S)
	% 高斯随机场采样
	rand_val = normrnd(0, 1, S);
    [K1, K2] = meshgrid(0: S-1, 0: S-1);
    % 构造系数
    coef = tau.^(alpha-1) .* (pi.^2 * (K1.^2 + K2.^2) + tau.^2).^(-alpha/2);
	L = S * coef .* rand_val;
    L(1, 1) = 0;
    U = idct2(L);
end