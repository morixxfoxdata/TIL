import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
# imgディレクトリのパスを設定
img_dir = 'Books/DLfrom_zero_5/Chapter1/img'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
# 離散型確率分布生成
def discrete_dist():
    # dice
    dice_faces = [1, 2, 3, 4, 5, 6]
    
    # Probability of each faces
    probabilities = [1/6] * 6
    # 画像の保存
    img_path = os.path.join(img_dir, 'plot1.png')
    
    plt.bar(dice_faces, probabilities, width=0.6, color='skyblue')
    plt.xlabel('dice faces')
    plt.ylabel('Probabilities')
    plt.title('Discrete Probability disitribution')
    plt.ylim(0, 0.2)
    plt.savefig(img_path)
    plt.show()

def continuous_dist():
    img_path = os.path.join(img_dir, 'plot2.png')
    # 平均0, 標準偏差1を定義
    mu, sigma = 0, 1
    x = np.linspace(mu - sigma*3, mu + sigma*3, 100)
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    plt.plot(x, y, color='blue')
    plt.xlabel('value')
    plt.ylabel('probability density')
    plt.savefig(img_path)
    plt.show()


def continuous_dist_prob():
    mu, sigma = 170, 10
    img_path = os.path.join(img_dir, 'plot3.png')
    x = np.linspace(140, 200, 1000)
    y = norm.pdf(x, mu, sigma)

    plt.plot(x, y, label='Normal distribution')
    plt.fill_between(x, y, where=(x >= 170) & (x <= 180), color='orange', alpha=0.5, label='170 <= x <= 180')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Normal distribution')
    plt.savefig(img_path)
    plt.show()


if __name__ == "__main__":
    # discrete_dist()
    # continuous_dist()
    continuous_dist_prob()