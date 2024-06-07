import os
import numpy as np
path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
# print(path)
xs = np.loadtxt(path)
# print(xs.shape)

"""
Parameter
============
phis
mus
covs
"""
phis = np.array([0.5, 0.5])
mus =np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)])     # np.eyeで単位行列生成
# print(np.eye(2))

K = len(phis)
N = len(xs)

MAX_ITER = 100
THESHOLD = 1e-4


def multivative_normal(x, mu, cov):
    det =np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y

def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivative_normal(x, mu, cov)
    return y


def likelihood(xs, phis, mus, covs):
    eps = 1e-8     # log(0)を防ぐ
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N
