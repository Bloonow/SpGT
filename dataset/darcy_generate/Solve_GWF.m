% 解方程，-div{a(x) * grad(p(x))} = f(x)

function P = Solve_GWF(coef, F)
	K = length(coef);
    % 用于插值的坐标点，均为 K 个点
	[X1,Y1] = meshgrid(1/(2*K): 1/K: (2*K-1)/(2*K), 1/(2*K): 1/K: (2*K-1)/(2*K));
	[X2,Y2] = meshgrid(0: 1/(K-1): 1, 0: 1/(K-1): 1);
	% 二维样条插值 spline
	coef = interp2(X1, Y1, coef, X2, Y2, 'spline');
	F    = interp2(X1, Y1, F,    X2, Y2, 'spline');
	% F 去掉四周边界处的点
    F = F(2: K-1, 2: K-1);

	% 创建 {K-2,K-2} 大小的元胞数组，且每个位置的元素为 [K-2,K-2] 的稀疏矩阵，deal() 将元素分配到每个位置
	D = cell(K-2, K-2);
	[D{:}] = deal(sparse(zeros(K-2)));
	% spdiags(B, d, m, n) 获取 B 的列，将其排列在 d 所指定的 [m,n] 矩阵的对角线上
	% 依次为元胞数组 D 中的元素赋值
	for j = 2: K-1
		D{j-1, j-1} = spdiags([
			[-(coef(2:K-2, j) + coef(3:K-1, j)) / 2; 0], ...
			(coef(1:K-2, j) + coef(3:K, j) + coef(2:K-1, j-1) + coef(2:K-1, j+1) + 4 .* coef(2:K-1, j)) / 2, ...
			[0; -(coef(2:K-2, j) + coef(3:K-1, j)) / 2]
		], -1:1, K-2, K-2);
		% (coef(1:K-2, j)   + coef(2:K-1, j) + ...
		%  coef(3:K, j)     + coef(2:K-1, j) + ...
		%  coef(2:K-1, j-1) + coef(2:K-1, j) + ...
		%  coef(2:K-1, j+1) + coef(2:K-1, j)) / 2, ...
		if j ~= K-1
			D{j-1, j} = spdiags(-(coef(2:K-1, j) + coef(2:K-1, j+1)) / 2, 0, K-2, K-2);
			D{j, j-1} = D{j-1, j};
		end
	end
	A = cell2mat(D) .* (K-1) .^ 2;

	% 解线性方程组 Ax = b，则 x = A \ b
	P = [zeros(1, K); [zeros(K-2, 1), vec2mat(A \ F(:), K-2), zeros(K-2, 1)]; zeros(1, K)];
	% 插值并转置
	P = interp2(X2, Y2, P, X1, Y1, 'spline')';
end