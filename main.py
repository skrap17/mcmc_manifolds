import numpy as np
from numpy import sqrt
from scipy.optimize import newton
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
np.random.seed(0)


def q(a, X, v, R, r):
    [x, y, z] = X + v + a * G(X, R, r)
    return (R - sqrt(x**2 + y**2))**2 + z**2 - r**2


def G(X, R, r):
    [x, y, z] = X
    s = sqrt(x**2 + y**2)
    return np.array([-2 * x * (R - s) / s, -2 * y * (R - s) / s, 2 * z])


def mcmc_manifold(N, R, r):
    X = np.zeros((N + 1, 3))
    X[0] = [R, 0, r]
    sigma = 1
    for i in range(N):
        X[i + 1] = X[i]
        Gx = G(X[i], R, r)
        qrx = np.linalg.qr(Gx.reshape(-1, 1), mode='complete')[0][:, 1:]
        v = qrx @ np.random.normal(size=2) * sigma
        a = newton(q, 0, args=(X[i], v, R, r,), full_output=True, disp=False)
        if not a[1].converged:
            continue
        w = a[0] * Gx
        Y = X[i] + v + w
        Gy = G(Y, R, r)
        qry = np.linalg.qr(Gy.reshape(-1, 1), mode='complete')[0][:, 1:]
        v_ = qry @ qry.T @ (X[i] - Y)
        # t = multivariate_normal.pdf(v_, mean=np.zeros(3))
        alpha = min(1, multivariate_normal.pdf(v_,  mean=np.zeros(3)) / multivariate_normal.pdf(v,  mean=np.zeros(3)))
        U = np.random.uniform()
        if U <= alpha:
            X[i + 1] = Y
    return X


X = mcmc_manifold(10000, 1, 0.5)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1, 1)
plt.show()