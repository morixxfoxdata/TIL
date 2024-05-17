import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# パラメータ設定
mu = [0, 0]  # 平均
sigma = [[1, 0.5], [0.5, 1]]  # 共分散行列

# 確率密度関数の作成
x, y = np.mgrid[-3:3:.05, -3:3:.05]
pos = np.dstack((x, y))
rv = multivariate_normal(mu, sigma)

# 確率密度関数を評価
z = rv.pdf(pos)

# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# サーフェスプロット
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# 積分範囲の描画
a, b = -1, 1
c, d = -1, 1

# 領域の縁を強調表示するために境界を描画
x_range = np.linspace(a, b, 100)
y_range = np.linspace(c, d, 100)
x_rect, y_rect = np.meshgrid(x_range, y_range)
z_rect = rv.pdf(np.dstack((x_rect, y_rect)))

# 境界のワイヤーフレームプロット
ax.plot_wireframe(x_rect, y_rect, z_rect, color='r')

# ラベルの追加
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Probability Density')
ax.set_title('3D Visualization of Probability Density Function with Integration Area')

# グラフの表示
plt.show()
