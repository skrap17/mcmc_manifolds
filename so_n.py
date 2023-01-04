import numpy as np
import scipy
from numpy import sqrt
from scipy.stats import norm
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
from scipy.stats import special_ortho_group
import pandas as pd
import seaborn as sns
np.random.seed(10)


# @profile
# def project(q, z, Q, grad, dim, nmax=0, tol=0.001):
#     a = np.zeros(dim)
#     i = 0
#     if nmax == 0:
#         nmax = 3 * dim
#     stop = False
#     while not stop:
#         arg = z + Q @ a
#         q_arg = q(arg)
#         # da = -np.linalg.pinv(grad(arg).T @ Q) @ q_arg
#         da = np.linalg.solve(grad(arg).T @ Q, -q_arg)
#         a = a + da
#         i += 1
#         stop = np.linalg.norm(q_arg) < tol
#         if i > nmax:
#             return a, False
#     return a, np.linalg.norm(q(z + Q @ a)) < tol
#

# @profile
# def mcmc_manifold(N, d, m, grad, q, x0, sigma, ineq_constraints=None):
#     da = d - m
#     X = np.zeros((N + 1, d))
#     X[0] = x0
#     accepted = 0
#     # rev_proj_fail = 0
#     for i in range(N):
#         print(i)
#         X[i + 1] = X[i]
#         Gx = grad(X[i])
#         qrx = np.linalg.qr(Gx, mode='complete')[0][:, m:]
#         t = np.random.normal(size=da) * sigma
#         if not isinstance(t, np.ndarray):
#             t = [t]
#         v = qrx @ t
#         a, flag = project(q, X[i] + v, Gx, grad, m)
#         if not flag:
#             continue
#         w = Gx @ a
#         Y = X[i] + v + w
#         if ineq_constraints is not None and not ineq_constraints(Y):
#             continue
#         Gy = grad(Y)
#         qry = np.linalg.qr(Gy, mode='complete')[0][:, m:]
#         v_ = qry @ qry.T @ (X[i] - Y)
#         alpha = min(1, np.exp(-(np.linalg.norm(v_) ** 2 - np.linalg.norm(v) ** 2) / 2 / sigma ** 2))
#         U = np.random.uniform()
#         if U > alpha:
#             continue
#         reversebility_check, flag = project(q, Y + v_, Gy, grad, m)
#         if not flag:
#             continue
#         X[i + 1] = Y
#         accepted += 1
#
#     return X, accepted / N


def gen_son(d, N):
    X = np.zeros((N + 1, d, d))
    i = 0
    fails = 0
    while i <= N:
        A = (np.random.normal(size=(d, d)) + 1j * np.random.normal(size=(d, d))) / sqrt(2.)
        Q, R = np.linalg.qr(A)
        D = np.diag(R)
        ph = np.diag(D / np.abs(D))
        X[i] = Q @ ph @ Q
        i += 1

    return X


def G_pre_process(d):
    gmap = np.zeros((int(d * (d + 1) / 2), d * d, d * d))
    for k in range(d):
        for l in range(k, d):
            if k == l:
                for j in range(d):
                    gmap[d * k + l - int(k * (k + 1) / 2), d * k + j, d * k + j] = 2
            else:
                for j in range(d):
                    gmap[d * k + l - int(k * (k + 1) / 2), d * l + j, d * k + j] = 1
                    gmap[d * k + l - int(k * (k + 1) / 2), d * k + j, d * l + j] = 1

    return gmap


def G_new(X):
    return (grad_map @ X).T


def q_new(X):
    d = int(sqrt(X.shape[0]))
    X = X.reshape(d, d)
    res = (np.tensordot(X, X, axes=(1, 1)) - q_rhs)[indices]
    return res


def G_col(X, k, l):
    # Assume X is dxd matrix
    d = X.shape[0]
    col = np.zeros(d * d)
    if k == l:
        for j in range(d):
            col[k * d + j] = 2 * X[k, j]
    else:
        for j in range(d):
            col[k * d + j] = X[l, j]
            col[l * d + j] = X[k, j]
    return col


def G(X):
    d = int(sqrt(X.shape[0]))
    X = X.reshape(d, d)
    m = int(d * (d + 1) / 2)
    res = np.zeros((d * d, m))
    for k in range(d):
        for l in range(k, d):
            res[:, d * k + l - int(k * (k + 1) / 2)] = G_col(X, k, l)
    return res


def q_col(X, k, l):
    # Assume X is dxd matrix
    if k == l:
        return np.linalg.norm(X[k, :]) ** 2 - 1
    else:
        return X[k, :].dot(X[l, :])


def q(X):
    d = int(sqrt(X.shape[0]))
    X = X.reshape(d, d)
    m = int(d * (d + 1) / 2)
    res = np.zeros(m)
    for k in range(d):
        for l in range(k, d):
            res[d * k + l - int(k * (k + 1) / 2)] = q_col(X, k, l)
    return res


def is_so(X):
    d = int(sqrt(X.shape[0]))
    return np.linalg.det(X.reshape(d, d)) > 0


N = 100
d = 11
# X = special_ortho_group.rvs(d, size=N + 1)
grad_map = G_pre_process(d)
q_rhs = np.eye(d)
indices = np.triu_indices(d)
# sigma = 2.1
sigma = 0.28
x0 = np.eye(d)
X, _ = mcmc_manifold(N, d * d, int(d * (d + 1) / 2), G_new, q_new, x0.flatten(), sigma, is_so)
# np.save("./chains/mcmc_son_10_6_.npy", X)
# X = np.load("./chains/mcmc_son_10_6.npy")
X = X.reshape((N + 1, d, d))
traces = np.array([np.trace(Xi) for Xi in X])
plt.subplots()
df = pd.DataFrame(data=traces.T, columns=['Trace'])
hist = sns.histplot(data=df, x='Trace', stat='density')
x = np.linspace(-5, 5, 200)
plt.plot(x, norm.pdf(x), color='red', linewidth=2, label='Theoretical density')
plt.xlim((-5, 5))
plt.legend()
# phi = np.array([np.arctan2(Xi[0, 0], Xi[1, 0]) for Xi in X])
# plt.subplots()
# plt.hist(phi, density=True)
plt.show()
