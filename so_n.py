import sys
import numpy as np
from tqdm import tqdm
from numpy import sqrt
from scipy.stats import norm, kstest
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold, profile
import pandas as pd
import seaborn as sns
np.random.seed(42)
sns.set()
# np.random.seed(12345)
# np.random.seed(10)

# for profiling run in terminal
# kernprof -l -v script_to_profile.py


def gen_son(d, N, f):
    X = np.zeros((N + 1, d, d))
    i = 0
    fails = 0
    pbar = tqdm(range(N + 1), desc="Samples generated", total=N + 1)
    while i <= N:
        A = np.random.normal(size=(d, d))
        Q, R = np.linalg.qr(A)
        D = np.diagonal(R)
        ph = np.diag(D / np.abs(D))
        Y = Q @ ph
        if np.linalg.det(Y) > 0:
            X[i] = Y
            i += 1
            pbar.update(1)
        else:
            if f:
                Y[[0, 1]] = Y[[1, 0]]
                X[i] = Y
                i += 1
                pbar.update(1)
            else:
                fails += 1

    pbar.close()
    print("Acceptance probability: " + "{0:.2%}".format(N / (N + fails)) + '\n')
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


N = 200000
d = 2
grad_map = G_pre_process(d)
q_rhs = np.eye(d)
indices = np.triu_indices(d)
sigma = 2.1
# sigma = 0.28
x0 = np.eye(d)
# x0 = x0[:, np.random.permutation(d)]
# while np.linalg.det(x0) < 0:
#     x0 = x0[:, np.random.permutation(d)]
# X, _ = mcmc_manifold(N, d * d, int(d * (d + 1) / 2), G_new, q_new, x0.flatten(), sigma, is_so)
if '-p' in sys.argv:
    profile.print_stats()
# X = gen_son(d, N)
# np.save("./chains/so_n/so_2.npy", X)
# X = np.load("./chains/so_n/mcmc_son_10_6.npy")
X = np.load("./chains/so_n/so_2.npy")
# X = X[0::10]
# Y = gen_son(d, N, False)
# Y_ = gen_son(d, N, True)
# x0 = X[25000]
# N = 10000
# X, _ = mcmc_manifold(N, d * d, int(d * (d + 1) / 2), G_new, q_new, x0.flatten(), sigma, is_so)
X = X.reshape((X.shape[0], d, d))
# b = X[-1]
# X = X[:29000]
# a = X[-1]
# aa = np.linalg.det(a)
# aaa = q_new(a.reshape(d*d))
# X = X[:30000]
traces = np.array([np.trace(Xi) for Xi in X])
# traces1 = np.array([np.trace(Xi) for Xi in Y])
# traces2 = np.array([np.trace(Xi) for Xi in Y_])
# print(kstest(traces, traces1))
# print(kstest(traces, traces2))
# print(kstest(traces1, traces2))
plt.subplots()
df = pd.DataFrame(data=traces.T, columns=['Trace'])
hist = sns.histplot(data=df, x='Trace', bins=201, stat='density')
x = np.linspace(-5, 5, 200)
plt.plot(x, norm.pdf(x), color='red', linewidth=2, label='Theoretical density')
plt.xlim((-5, 5))
plt.legend()
phi = np.array([np.arctan2(Xi[0, 0], Xi[1, 0]) for Xi in X])
plt.subplots()
df1 = pd.DataFrame(data=phi.T, columns=['Phi'])
sns.histplot(data=df1, x='Phi', stat='density')


def cdf1(x):
    if x < -np.pi or x > np.pi:
        return 0
    return (x + np.pi ) / 2. / np.pi


x = np.linspace(-np.pi, np.pi, 100)
plt.subplots()
sns.ecdfplot(x=phi)
plt.plot(x, np.vectorize(cdf1)(x))
print("KS test for Phi:", kstest(phi, np.vectorize(cdf1)))
# plt.hist(phi, density=True)
plt.show()
