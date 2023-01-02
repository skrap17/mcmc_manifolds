import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
import seaborn as sns
import pandas as pd
np.random.seed(10)

r = 0.5
R = 1


def get_phi(X):
    [x, y, z] = X
    phi = 0
    s = sqrt(x**2 + y**2)
    if z >= 0:
        if s >= R:
            phi = np.arcsin(z / r)
        else:
            phi = np.pi - np.arcsin(z / r)
    else:
        if s <= R:
            phi = np.pi + np.arcsin(-z / r)
        else:
            phi = 2 * np.pi - np.arcsin( -z / r)

    return phi


def get_theta(X):
    [x, y, z] = X
    theta = 0
    if x > 0:
        if y > 0:
            theta = np.arctan(y / x)
        else:
            theta = 2 * np.pi - np.arctan(-y / x)
    else:
        if y > 0:
            theta = np.pi - np.arctan(-y / x)
        else:
            theta = np.pi + np.arctan(y / x)

    return theta


def q(X):
    [x, y, z] = X
    return np.array([(R - sqrt(x**2 + y**2))**2 + z**2 - r**2])


def G(X):
    [x, y, z] = X
    s = sqrt(x**2 + y**2)
    return np.array([-2 * x * (R - s) / s, -2 * y * (R - s) / s, 2 * z]).reshape(-1, 1)


N = 1000000
x0 = np.array([R, 0, r])
sigma = 0.47
X, prob = mcmc_manifold(N, 3, 1, G, q, x0, sigma)
print(prob)
# np.save("./chains/mcmc_10_6_new.npy", X)
# X = np.load("./chains/mcmc_10_6_new.npy")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='MCMC')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1, 1)
ax.legend()

Phi = np.array([get_phi(Xi) for Xi in X])
Theta = np.array([get_theta(Xi) for Xi in X])
fig1, ax1 = plt.subplots(2, 1)
# ax1[0].hist(Theta, bins=100, label='theta sample dist', density=True)
# ax1[1].hist(Phi, bins=100, label='phi sample dist', density=True)
df = pd.DataFrame(data=np.vstack((Theta, Phi)).T, columns=['Theta', 'Phi'])
sns.histplot(ax=ax1[0], data=df, x='Theta', bins=100, stat='density')
sns.histplot(ax=ax1[1], data=df, x='Phi', bins=100,  stat='density')
x = np.linspace(0, 6.28, 200)
ax1[0].plot(x, np.full(x.shape[0], 0.5 / np.pi), color='red', label='True density')
ax1[1].plot(x, (1 + r / R * np.cos(x)) / 2 / np.pi, color='red', label='True density')
ax1[0].legend()
ax1[1].legend()
plt.show()