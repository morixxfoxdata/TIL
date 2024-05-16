import matplotlib.pyplot as plt
import numpy as np

def sample_avg(size, num_sampling):
    x_means = []
    for _ in range(num_sampling):
        xs = []
        for i in range(size):
            x = np.random.rand()
            xs.append(x)
        mean = np.mean(xs)
        x_means.append(mean)
    
    plt.hist(x_means, bins='auto', density=True)
    # if density is True, return probability density
    plt.ylim(0, 5)
    plt.title(f'N={size}')
    plt.show()

if __name__ == "__main__":
    # sample_avg(1, 10000)
    # sample_avg(10, 10000)
    sample_avg(4, 10000)