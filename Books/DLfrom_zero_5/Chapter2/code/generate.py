import os
import numpy as np
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(__file__), 'height.txt')
# print(os.path.dirname(__file__))
xs = np.loadtxt(path)

mu = np.mean(xs)
sigma = np.std(xs)
# 生成モデルから10000のサンプルを作成
samples = np.random.normal(mu, sigma, 10000)

plt.hist(xs, bins='auto', density=True, alpha=0.7, label='original')
plt.hist(samples, bins='auto', density=True, alpha=0.7, label='generated')
plt.xlabel('Height[cm]')
plt.ylabel('Probability Density')
plt.legend()
plt.show()