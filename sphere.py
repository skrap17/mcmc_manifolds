import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
np.random.seed(10)


def q(X):
    [x, y, z] = X
    return np.array([x**2 + y**2 + z**2 - 1])


def G(X):
    [x, y, z] = X
    return np.array([2*x, 2*y, 2*z]).reshape(-1, 1)


def h(X):
    [x, y, z] = X
    return z > 0 and x > 0


N = 10000
x0 = np.array([1, 0, 0])
sigma = 1
X, prob = mcmc_manifold(N, 3, 1, G, q, x0, sigma)
print(prob)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='MCMC')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.1, 1.1)
ax.legend()
plt.show()