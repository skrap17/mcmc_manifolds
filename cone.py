import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
np.random.seed(10)


def q(X):
    [x, y, z] = X
    return np.array([z - sqrt(x**2 + y**2)])


def G(X):
    [x, y, z] = X
    s = sqrt(x**2 + y**2)
    return np.array([-x / s, -y / s, 1]).reshape(-1, 1)


def h(X):
    [x, y, z] = X
    return z > 0 and x**2 + y**2 < 1


N = 10000
x0 = np.array([0.5, 0.5, 0.5 * sqrt(2)])
sigma = 0.9
X, prob = mcmc_manifold(N, 3, 1, G, q, x0, sigma, h)
print(prob)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='MCMC')
ax.set_xlim(-0.9, 0.9)
ax.set_ylim(-0.9, 0.9)
ax.set_zlim(0, 1.5)
ax.legend()
plt.show()