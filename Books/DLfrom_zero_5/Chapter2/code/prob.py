import os
import numpy as np
from scipy.stats import norm

path = os.path.join(os.path.dirname(__file__), 'height.txt')
# print(os.path.dirname(__file__))
xs = np.loadtxt(path)

mu = np.mean(xs)
sigma = np.std(xs)

p1 = norm.cdf(160, loc=mu, scale=sigma)
print('p(x <= 160):', p1)

p2 = norm.cdf(180, loc=mu, scale=sigma)
print('p(x > 180):', 1-p2)