import numpy as np
from numpy import sqrt
from tqdm import tqdm
import scipy
from torus import get_phi, get_theta
from matplotlib import pyplot as plt
import matplotlib
from mcmc_manifolds import mcmc_manifold
import seaborn as sns
import pandas as pd
sns.set()
np.random.seed(42)
r = 0.5
R = 1

N = 1000
Z = 4 * np.pi ** 2 * r * R
x0 = np.array([R, 0, r])
sigma = 0.5
# X, _ = mcmc_manifold(N, 3, 1, G, q, x0, sigma)
# # np.save("./chains/mcmc_10_6_torus.npy", X)
# X = np.load("./chains/torus/mcmc/1000000_123456.npy")
X = np.load("./chains/bad_names/mcmc_10_6_torus.npy")

X_ = X[0::100]
# X = X[0::5]
# Z_mcmc = np.mean(X[:, 1]**2 + X[:, 2]**2)
# print("MCMC Integral estimate:", Z_mcmc)
# Y, prob = crude_mc(N, R, r)
# # np.save("./chains/cmc_10_6_torus.npy", Y)
# Z_cmc = np.mean(Y[:, 1]**2 + Y[:, 2]**2)
# print("CMC Integral estimate:", Z_cmc)
# print("Theoretical value:", theoretical_x_moment(R, r))
#
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_[:, 0], X_[:, 1], X_[:, 2], s=0.2, label='MCMC', alpha=0.7)
ax.set_xlim(-(R + r), R + r)
ax.set_ylim(-(R + r), R + r)
ax.set_zlim(-2 * r, 2 * r)
# ax.legend()
#
Phi = np.array([get_phi(Xi) for Xi in X])
Theta = np.array([get_theta(Xi) for Xi in X])
# Theta = Theta[~np.isnan(Phi)]
# Phi = Phi[~np.isnan(Phi)]
# print(np.mean(r * (R + r * np.cos(Theta))) * 4 * np.pi**2)
fig1, ax1 = plt.subplots(1, 2)
df = pd.DataFrame(data=np.vstack((Theta, Phi)).T, columns=['Theta', 'Phi'])
sns.histplot(ax=ax1[0], data=df, x='Theta', bins=50, stat='density')
h = sns.histplot(ax=ax1[1], data=df, x='Phi', bins=50, stat='density', legend=False)
h.set(ylabel=None)
# ax1[0].hist(Theta)
# ax1[1].hist(Phi)
x = np.linspace(0, 6.28, 200)
ax1[0].plot(x, np.full(x.shape[0], 0.5 / np.pi), color='red', label='True density', linewidth=2)
ax1[1].plot(x, (1 + r / R * np.cos(x)) / 2 / np.pi, color='red', label='True density', linewidth=2)
ax1[0].legend()
ax1[1].legend()
# ax1[1].get_yaxis().set_visible(False)


def cdf(x):
    if x < 0 or x > 2 * np.pi:
        return 0
    return (R * x + r * np.sin(x)) / 2 / np.pi / R


def cdf1(x):
    if x < 0 or x > 2 * np.pi:
        return 0
    return x / 2. / np.pi
plt.subplots()
sns.ecdfplot(x=Phi)
plt.plot(x, np.vectorize(cdf)(x))
sns.ecdfplot(x=Theta)
plt.plot(x, np.vectorize(cdf1)(x))

print("KS test for Phi:", scipy.stats.kstest(Phi, np.vectorize(cdf)))
print("KS test for Theta:", scipy.stats.kstest(Theta, np.vectorize(cdf1)))
# plt.subplots()
# scipy.stats.probplot(Theta / 2. / np.pi, plot=plt)
plt.show()
