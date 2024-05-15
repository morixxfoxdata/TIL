import numpy as np
import matplotlib.pyplot as plt
import os
from norm_dist import normal
# imgディレクトリのパスを設定
img_dir = 'Books/DLfrom_zero_5/Chapter1/img'
def norm_param_mu():
    img_path = os.path.join(img_dir, 'plot5.png')
    x = np.linspace(-10, 10, 1000)
    y0 = normal(x, mu=-3)
    y1 = normal(x, mu=0)
    y2 = normal(x, mu=5)

    plt.plot(x, y0, label='$\mu$=-3')
    plt.plot(x, y1, label='$\mu$=0')
    plt.plot(x, y2, label='$\mu$=5')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(img_path)
    plt.show()

def norm_param_simga():
    img_path = os.path.join(img_dir, 'plot6.png')
    x = np.linspace(-10, 10, 1000)
    y0 = normal(x, mu=0, sigma=0.5)
    y1 = normal(x, mu=0, sigma=1)
    y2 = normal(x, mu=0, sigma=2)

    plt.plot(x, y0, label='$\sigma$=0.5')
    plt.plot(x, y1, label='$\sigma$=1')
    plt.plot(x, y2, label='$\sigma$=2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(img_path)
    plt.show()
if __name__ == '__main__':
    # norm_param_mu()
    norm_param_simga()