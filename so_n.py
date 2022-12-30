import numpy as np
from numpy import sqrt
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
np.random.seed(10)


def G_col(X, k, l):
    # Assume X is dxd matrix
    d = X.shape[0]
    col = np.zeros(d * d)
    if k == l:
        for i in range(d):
            if i == k:
                for j in range(d):
                    col[i * d + j] = 2 * X[i, j]
    else:
        for i in range(d):
            if i == k or i == l:
                for j in range(d):
                        col[i * d + j] = X[i, j]
    return col


def G(X):
    d = int(sqrt(X.shape[0]))
    X = X.reshape(d, d)
    m = int(d * (d + 1) / 2)
    res = np.zeros((d * d, m))
    for k in range(d):
        for l in range(k, d):
            res[:, d * k + l - int(k * (k + 1) / 2)] = G_col(X, k, l)
    return res


def q_col(X, k, l):
    # Assume X is dxd matrix
    if k == l:
        return np.linalg.norm(X[k, :])**2 - 1
    else:
        return X[k, :].dot(X[l, :])


def q(X):
    d = int(sqrt(X.shape[0]))
    X = X.reshape(d, d)
    m = int(d * (d + 1) / 2)
    res = np.zeros(m)
    for k in range(d):
        for l in range(k, d):
            res[d * k + l - int(k * (k + 1) / 2)] = q_col(X, k, l)
    return res


def is_so(X):
    d = int(sqrt(X.shape[0]))
    return np.linalg.det(X.reshape(d, d)) > 0


N = 100
d = 5
sigma = 0.1
x0 = np.eye(d)
np.random.shuffle(x0)
X, prob = mcmc_manifold(N, d * d, int(d * (d + 1) / 2), G, q, x0.flatten(), sigma, is_so)
print(prob)
X = X.reshape(N + 1, d, d)
traces = np.array([np.trace(Xi) for Xi in X])
plt.subplots()
plt.hist(traces, density=True)
plt.show()