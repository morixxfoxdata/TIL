import numpy as np

np.random.seed(0)
N = 10000
D = 2
# N: サンプル数, D: 各サンプルの次元数
xs = np.random.rand(N, D)   # 一様分布からのランダムデータ

# mu: (D,)
mu = np.sum(xs, axis=0)
mu /= N

cov = 0

for n in range(N):
    # x: 各サンプル(D,)
    x = xs[n]
    z = x - mu
    z = z[:, np.newaxis]
    cov += z @ z.T

cov /= N

print(mu)
print(cov)