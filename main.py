import numpy as np
from numpy import sqrt
from scipy.optimize import newton
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
np.random.seed(10)


def q(a, X, v, R, r):
    [x, y, z] = X + v + a * G(X, R, r)
    return (R - sqrt(x**2 + y**2))**2 + z**2 - r**2


def G(X, R, r):
    [x, y, z] = X
    s = sqrt(x**2 + y**2)
    return np.array([-2 * x * (R - s) / s, -2 * y * (R - s) / s, 2 * z])

# @profile
# line_profiler
def mcmc_manifold(N, R, r):
    X = np.zeros((N + 1, 3))
    X[0] = [R, 0, r]
    sigma = 1
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


def update_graph(num):
    data = X[0:num]
    graph._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
    # title.set_text('3D Test, time={}'.format(num))


X, prob = mcmc_manifold(100000, 1, 0.5)
print(prob)
print(np.where(X[:, 2] > 0.6))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
graph = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2)
# graph = ax.scatter(X[0, 0], X[0, 1], X[0, 2], s=2)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1, 1)
# ani = matplotlib.animation.FuncAnimation(fig, update_graph, 1000, interval=10, blit=False)
plt.show()