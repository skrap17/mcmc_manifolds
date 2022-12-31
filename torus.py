import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
np.random.seed(10)

r = 0.5
R = 1


def q(X):
    [x, y, z] = X
    return np.array([(R - sqrt(x**2 + y**2))**2 + z**2 - r**2])


def G(X):
    [x, y, z] = X
    s = sqrt(x**2 + y**2)
    return np.array([-2 * x * (R - s) / s, -2 * y * (R - s) / s, 2 * z]).reshape(-1, 1)


N = 10000
x0 = np.array([R, 0, r])
sigma = 0.47
X, prob = mcmc_manifold(N, 3, 1, G, q, x0, sigma)
print(prob)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='MCMC')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1, 1)
ax.legend()
plt.show()