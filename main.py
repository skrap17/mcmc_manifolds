import numpy as np
from numpy import sqrt
from scipy.optimize import newton
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import line_profiler
# np.random.seed(10)


def q(a, X, v, R, r):
    [x, y, z] = X + v + a * G(X, R, r)
    return (R - sqrt(x**2 + y**2))**2 + z**2 - r**2


def G(X, R, r):
    [x, y, z] = X
    s = sqrt(x**2 + y**2)
    return np.array([-2 * x * (R - s) / s, -2 * y * (R - s) / s, 2 * z])


# @profile
# line_profiler
@profile
def mcmc_manifold(N, R, r, x0):
    X = np.zeros((N + 1, 3))
    X[0] = x0
    sigma = 1.25
    accepted = 0
    for i in range(N):
        X[i + 1] = X[i]
        Gx = G(X[i], R, r)
        qrx = np.linalg.qr(Gx.reshape(-1, 1), mode='complete')[0][:, 1:]
        v = qrx @ np.random.normal(size=2) * sigma
        a = newton(q, 0, args=(X[i], v, R, r,), full_output=True, disp=False)
        if not a[1].converged or q(a[0], X[i], v, R, r) > 0.01:
            continue
        w = a[0] * Gx
        Y = X[i] + v + w
        Gy = G(Y, R, r)
        qry = np.linalg.qr(Gy.reshape(-1, 1), mode='complete')[0][:, 1:]
        v_ = qry @ qry.T @ (X[i] - Y)
        # t = multivariate_normal.pdf(v_, mean=np.zeros(3))
        alpha = min(1, multivariate_normal.pdf(v_,  mean=np.zeros(3)) / multivariate_normal.pdf(v,  mean=np.zeros(3)))
        U = np.random.uniform()
        if U > alpha:
            continue
        reversebility_check = newton(q, 0, args=(Y, v_, R, r,), full_output=True, disp=False)
        if not reversebility_check[1].converged or q(reversebility_check[0], Y, v_, R, r) > 0.01:
            continue
        X[i + 1] = Y
        accepted += 1

    return X, accepted / N


def crude_mc(N, R, r, x0):
    X = np.zeros((N + 1, 3))
    X[0] = x0
    i = 0
    fails = 0
    while i < N:
        [U, V, W] = np.random.uniform(size=3)
        Theta = 2 * np.pi * U
        Phi = 2 * np.pi * V
        # t = (R + r * np.cos(Theta)) / (r * R)
        if W <= (R + r * np.cos(Theta)) / (r + R):
            X[i + 1] = np.array([(R + r * np.cos(Theta)) * np.cos(Phi), (R + r * np.cos(Theta)) * np.sin(Phi),
                                 r * np.sin(Theta)])
            i += 1
        else:
            fails += 1
    return X, N / (N + fails)


# def update_graph(num):
#     data = X[0:num]
#     graph._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
#     # title.set_text('3D Test, time={}'.format(num))


R = 1
r = 0.5
Y, prob = crude_mc(1000000, 1, 0.5, [R, 0, r])
np.save("chains/old/cmc_10_6_2.npy", Y)
Z_cmc = 4 * np.pi**2 * r * R * np.mean(Y[:, 1]**2 + Y[:, 2]**2)
print("Crude MC acceptance rate:", prob)
print("Crude MC Integral estimate:", Z_cmc)
print()
X, prob = mcmc_manifold(1000000, 1, 0.5, [R, 0, r])
# np.save("./chains/mcmc_10_6_2.npy", X)
# X = X[0::10]
Z_mcmc = 4 * np.pi**2 * r * R * np.mean(X[:, 1]**2 + X[:, 2]**2 )
print("MCMC acceptance rate:", prob)
print("MCMC Integral estimate:", Z_mcmc)
# print(np.where(X[:, 2] > 0.6))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, label='MCMC')
# graph = ax.scatter(X[0, 0], X[0, 1], X[0, 2], s=2)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1, 1)
ax.legend()
# ani = matplotlib.animation.FuncAnimation(fig, update_graph, 1000, interval=10, blit=False)
# theta = np.arccos(X[:, 2] / sqrt(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2))
# phi = np.sign(X[:, 1]) * np.arccos(X[:, 0] / sqrt(X[:, 0]**2 + X[:, 1]**2))
# fig1, ax1 = plt.subplots(2, 1)
# sns.histplot(ax=ax1[0], x=theta, bins=200)
# sns.histplot(ax=ax1[1], x=phi, bins=200)
# ax1[0].hist(theta, bins=100, label='theta sample dist', density=True)
# ax1[1].hist(phi, bins=100, label='phi sample dist', density=True)
# x = np.linspace(0, 6, 200)
# ax1[1].plot(x, (1 + r / R * np.cos(x)) / 2 / np.pi, color='orange')
# ax1[0].legend()
# ax1[1].legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=2, color='red', label="Crude MC")
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(-1, 1)
ax2.legend()


plt.show()
