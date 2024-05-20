# from Chapter1.code.norm_dist import normal
# from hist import export_xs
import numpy as np
import os
import matplotlib.pyplot as plt
def export_xs():
    path = os.path.join(os.path.dirname(__file__), 'height.txt')
    xs = np.loadtxt(path)
    return xs
def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

def fit_plot():
    xs = export_xs()
    mu = np.mean(xs)
    sigma = np.std(xs)
    x = np.linspace(150, 190, 1000)
    y = normal(x, mu, sigma)
    plt.hist(xs, bins='auto', density=True)
    plt.plot(x, y)
    plt.xlabel('Height(cm)')
    plt.ylabel('Probability Density')
    plt.show()

if __name__ == '__main__':
    # print(fit_plot())
    fit_plot()
