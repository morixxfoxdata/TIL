import numpy as np
import matplotlib.pyplot as plt
import os
# imgディレクトリのパスを設定
img_dir = 'Books/DLfrom_zero_5/Chapter1/img'
def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

def plot_normal():
    img_path = os.path.join(img_dir, 'plot4.png')
    x = np.linspace(-5, 5, 100)
    y = normal(x)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(img_path)
    plt.show()

if __name__ == '__main__':
    plot_normal()