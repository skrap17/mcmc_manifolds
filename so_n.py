import numpy as np
from numpy import sqrt
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
np.random.seed(10)


def project(q, z, Q, dim, tol=0.0001, nmax=0):
    a = np.zeros(dim)
    i = 0
    if nmax == 0:
        nmax = 3 * dim
    while i == 0 or np.linalg.norm(q_arg) > tol:
        arg = z + (Q @ a).reshape(z.shape[0], z.shape[0])
        q_arg = q(arg)
        t = G(arg).T @ Q
        # da = np.linalg.lstsq(G(arg).T @ Q, -q_arg)
        # a = a + da[0]
        da = -np.linalg.pinv(G(arg).T @ Q) @ q_arg
        a = a + da
        i += 1
        if i > nmax:
            return a, False
    return a,  np.linalg.norm(q(z + (Q @ a).reshape(z.shape[0], z.shape[0]))) < tol


def G_col(X, k, l):
    # Assume X is dxd matrix
    d = X.shape[0]
    col = np.zeros(d * d)
    if k == l:
        for i in range(d):
            if i == k:
                for j in range(d):
                    col[i * d + j] = 2 * X[i, j]
    else:
        for i in range(d):
            if i == k or i == l:
                for j in range(d):
                        col[i * d + j] = X[i, j]
    return col


def G(X):
    # Assume X is dxd matrix
    d = X.shape[0]
    m = int(d * (d + 1) / 2)
    res = np.zeros((d * d, m))
    for k in range(d):
        for l in range(k, d):
            res[:, d * k + l - int(k * (k + 1) / 2)] = G_col(X, k, l)
    return res


def q_col(X, k, l):
    # Assume X is dxd matrix
    if k == l:
        return np.linalg.norm(X[k, :])**2 - 1
    else:
        return X[k, :].dot(X[l, :])


def q(X):
    d = X.shape[0]
    m = int(d * (d + 1) / 2)
    res = np.zeros(m)
    # g = G(X)
    # arg = X + v + (G(X) @ a).reshape(d, d)
    for k in range(d):
        for l in range(k, d):
            res[d * k + l - int(k * (k + 1) / 2)] = q_col(X, k, l)
    return res
    # [x, y, z] = X + v + a * G(X)


def mcmc_manifold(N, d):
    X = np.zeros((N + 1, d, d))
    X[0] = np.eye(d)
    m = int(d * (d + 1) / 2)
    sigma = 0.00001
    accepted = 0
    for i in range(N):
        print(i)
        X[i + 1] = X[i]
        Gx = G(X[i])
        qrx = np.linalg.qr(Gx, mode='complete')[0][:, d * d - m:]
        v = (qrx @ np.random.normal(size=m)).reshape(d, d) * sigma
        a, flag = project(q, X[i] + v, Gx, m)
        if not flag:
            continue
        # a = fsolve(q, np.zeros(m), args=(X[i], v), full_output=True)
        # print(3)
        # t = q(a[0], X[i], v)
        # if not a[2] == 1 or np.linalg.norm(q(a[0], X[i], v)) > 0.01:
        #     continue
        w = (Gx @ a).reshape(d, d)
        Y = X[i] + v + w
        Gy = G(Y)
        qry = np.linalg.qr(Gy, mode='complete')[0][:,  d * d - m:]
        v_ = (qry @ qry.T @ (X[i] - Y).flatten()).reshape(d, d)
        # t = multivariate_normal.pdf(v_, mean=np.zeros(3))
        alpha = min(1, multivariate_normal.pdf(v_.flatten(),  mean=np.zeros(d * d))
                    / multivariate_normal.pdf(v.flatten(),  mean=np.zeros(d * d)))
        U = np.random.uniform()
        if U > alpha:
            continue
        reversebility_check, flag = project(q, Y + v_, Gy, m)
        if not flag:
            continue
        # reversebility_check = fsolve(q, np.zeros(m), args=(Y, v_,), full_output=True)
        # if not reversebility_check[2] == 1 or np.linalg.norm(q(reversebility_check[0], Y, v_)) > 0.01:
        #     continue
        if np.linalg.det(Y) < 0:
            continue
        X[i + 1] = Y
        accepted += 1

    return X, accepted / N


X, prob = mcmc_manifold(100, 20)
for Xi in X:
    print(np.linalg.det(Xi))
traces = np.array([np.trace(Xi) for Xi in X])
plt.subplots()
plt.hist(traces, density=True)
plt.show()