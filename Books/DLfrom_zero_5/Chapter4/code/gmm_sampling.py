import numpy as np
import matplotlib.pyplot as plt
# ====== 学習済パラメータ ======
# 平均行列(0: 1つ目の平均, 1: 2つ目の平均)
mus = np.array([[2.0, 54.50],
                [4.3, 80.0]])
# 共分散行列の行列
covs = np.array([[[0.07, 0.44],
                 [0.44, 33.7]],
                 [[0.17, 0.94],
                  [0.94, 36.00]]])
# 各正規分布が選ばれる確率
phis = np.array([0.35, 0.65])
# ===========================

def sample():
    z = np.random.choice(2, p=phis)
    mu, cov = mus[z], covs[z]
    x = np.random.multivariate_normal(mu, cov)
    return x

N = 500
xs = np.zeros((N, 2))
for i in range(N):
    xs[i] = sample()

# print(xs)
plt.scatter(xs[:, 0], xs[:, 1], color='orange', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.show()