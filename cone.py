import numpy as np
from numpy import sqrt
import pandas as pd
import seaborn as sns
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
sigma = 0.8
X, _ = mcmc_manifold(N, 3, 1, G, q, x0, sigma, h)
Z_mcmc = np.mean(X[:, 1]**2 + X[:, 0]**2)
print("MCMC Integral estimate:", Z_mcmc)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='MCMC')
ax.set_xlim(-0.9, 0.9)
ax.set_ylim(-0.9, 0.9)
ax.set_zlim(0, 1.5)
fig1, ax1 = plt.subplots(3, 1)
# ax1[0].hist(Theta, bins=100, label='theta sample dist', density=True)
# ax1[1].hist(Phi, bins=100, label='phi sample dist', density=True)
df = pd.DataFrame(data=X, columns=['X', 'Y', 'Z'])
sns.histplot(ax=ax1[0], data=df, x='X', bins=100, stat='density')
sns.histplot(ax=ax1[1], data=df, x='Y', bins=100,  stat='density')
sns.histplot(ax=ax1[2], data=df, x='Z', bins=100,  stat='density')
x = np.linspace(-1, 1, 200)
ax1[0].plot(x, 2 * sqrt(1 - x**2) / np.pi, color='red', label='True density')
ax1[1].plot(x, 2 * sqrt(1 - x**2) / np.pi, color='red', label='True density')
ax1[2].plot(x[100:], 2 * x[100:], color='red', label='True density')
ax1[0].legend()
ax1[1].legend()
ax1[2].legend()
ax.legend()
plt.show()