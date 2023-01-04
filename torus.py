import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
import seaborn as sns
import pandas as pd
np.random.seed(42)

r = 0.5
R = 1


def theoretical_x_moment(R, r):
    return 0.5 * R**2 + 0.625 * r**2


def theoretical_z_moment(R, r):
    return R**2 + 0.75 * r**2


def crude_mc(N, R, r):
    X = np.zeros((N + 1, 3))
    i = -1
    fails = 0
    while i < N:
        [U, V, W] = np.random.uniform(size=3)
        Phi = 2 * np.pi * U
        Theta = 2 * np.pi * V
        if W <= (R + r * np.cos(Phi)) / (r + R):
            X[i + 1] = np.array([(R + r * np.cos(Phi)) * np.cos(Theta), (R + r * np.cos(Phi)) * np.sin(Theta),
                                 r * np.sin(Phi)])
            i += 1
        else:
            fails += 1
    return X, N / (N + fails)


def get_phi(X):
    [x, y, z] = X
    if abs(z) > r:
        z = r * np.sign(z)
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


N = 100000
Z = 4 * np.pi**2 * r * R
x0 = np.array([R, 0, r])
sigma = 0.5
X, _ = mcmc_manifold(N, 3, 1, G, q, x0, sigma)
# np.save("./chains/mcmc_10_6_torus.npy", X)
# X = np.load("./chains/mcmc_10_5_new.npy")
Z_mcmc = np.mean(X[:, 1]**2 + X[:, 2]**2)
print("MCMC Integral estimate:", Z_mcmc)
Y, prob = crude_mc(N, R, r)
print(prob)
# np.save("./chains/cmc_10_6_torus.npy", Y)
Z_cmc = np.mean(Y[:, 1]**2 + Y[:, 2]**2)
print("CMC Integral estimate:", Z_cmc)
print("Theoretical value:", theoretical_x_moment(R, r))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='MCMC')
ax.set_xlim(-(R + r), R + r)
ax.set_ylim(-(R + r), R + r)
ax.set_zlim(-2 * r, 2 * r)
ax.legend()

Phi = np.array([get_phi(Xi) for Xi in X])
Theta = np.array([get_theta(Xi) for Xi in X])
# Theta = Theta[~np.isnan(Phi)]
# Phi = Phi[~np.isnan(Phi)]
print(np.mean(r * (R + r * np.cos(Theta))) * 4 * np.pi**2)
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