% 达西方程数据的生成脚本，问题域为二维标准网格 [0,1]^2，没问题，数据可以

function [thresh_a, thresh_p] = Darcy_Gen(S)
    % S 表示一个维度上数据点的数目，包括边界位置，则 S 最小为 2，即网格的 4 个顶点
    % S = 1024;
    % H 表示网格两个相邻点之间的步长，即 H = 1/(S-1)，也可表示网格的分辨率
    % H = 1/(S-1);
    % X,Y  二维网格，仅用于绘制
    % [X, Y] = meshgrid(0: (1/(S-1)): 1);

    % 从 GRF 中生成随机系数，alpha = 2, tau = 3
    norm_a = Gaussian_Random_Field(2, 3, S);

    % 取为自然常数的幂，以确保 a(x) > 0，如此使得PDE方程为椭圆形方程
    % lognorm_a = exp(norm_a);
    % 另一种方式，直接取阈值，将PDE变为椭圆形方程
    thresh_a = zeros(S, S);
    thresh_a(norm_a >= 0) = 12;
    thresh_a(norm_a <  0) = 3;

    % Forcing 函数, f(x) = 1
    f = ones(S, S);

    % 解方程，-div{a(x) * grad(p(x))} = f(x)
    % lognorm_p = Solve_GWF(lognorm_a, f);
    thresh_p = Solve_GWF(thresh_a, f);
end
