import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
np.random.seed(10)


def q(X):
    [x, y, z] = X
    return np.array([x**2 + y**2 + z**2 - 1, (x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2 - 1])


def q1(X):
    [x, y, z] = X
    return np.array([x**2 + y**2 + z**2 - 1, (x-0.5)**2 + y**2 + z**2 - 1])

def G(X):
    [x, y, z] = X
    return np.array([[2*x, 2*x - 1], [2*y, 2*y - 1], [2*z, 2*z - 1]])

def G1(X):
    [x, y, z] = X
    return np.array([[2*x, 2*x - 1], [2*y, 2*y], [2*z, 2*z]])


N = 1000
a = (3. - sqrt(23.)) / 8.
x0 = np.array([a, sqrt(1-a**2), 0])
y0 = np.array([0.25, sqrt(15.) / 4., 0])
sigma = 1.3
X, prob = mcmc_manifold(N, 3, 2, G, q, x0, sigma)
print(prob)
Y, prob = mcmc_manifold(N, 3, 2, G1, q1, y0, sigma)
print(prob)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='first circle')
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=1, label='second circle')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.1, 1.1)
ax.legend()
plt.show()