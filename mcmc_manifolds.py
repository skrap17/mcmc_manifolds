import line_profiler
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve, inv
from scipy.stats import multivariate_normal

profile = line_profiler.LineProfiler()


# @profile
def project(q, z, Q, grad, dim, nmax=0, tol=0.0001):
    a = np.zeros(dim)
    i = 0
    if nmax == 0:
        nmax = min(3 * dim, 40)
    stop = False
    while not stop:
        arg = z + Q @ a
        q_arg = q(arg)
        try:
            # da = solve(grad(arg).T @ Q, -q_arg)
            # da = -np.linalg.pinv(grad(arg).T @ Q) @ q_arg
            da = -inv(grad(arg).T @ Q) @ q_arg
        except:
            # print("Bad matrix")
            da = -np.linalg.pinv(grad(arg).T @ Q) @ q_arg
        a = a + da
        i += 1
        stop = np.linalg.norm(q_arg) < tol
        if i > nmax:
            return a, False
    return a, True


# @profile
def mcmc_manifold(N, d, m, grad, q, x0, sigma, ineq_constraints=None):
    da = d - m
    X = np.zeros((N + 1, d))
    X[0] = x0
    accepted = 0
    # cov = np.eye(da) * sigma * sigma
    pbar = tqdm(range(N), desc="Elements of Markov chain generated")
    # pbar = range(N)
    for i in pbar:
        X[i + 1] = X[i]
        Gx = grad(X[i])
        qrx = np.linalg.qr(Gx, mode='complete')[0][:, m:]
        t = np.random.normal(size=da) * sigma
        # t = multivariate_normal.rvs(cov=cov)
        if not isinstance(t, np.ndarray):
            t = [t]
        v = qrx @ t
        a, flag = project(q, X[i] + v, Gx, grad, m)
        if not flag:
            continue
        w = Gx @ a
        Y = X[i] + v + w
        if ineq_constraints is not None and not ineq_constraints(Y):
            continue
        Gy = grad(Y)
        qry = np.linalg.qr(Gy, mode='complete')[0][:, m:]
        v_ = qry @ qry.T @ (X[i] - Y)

        alpha = min(1, np.exp(-(np.linalg.norm(v_) ** 2 - np.linalg.norm(v) ** 2) / 2 / sigma ** 2))
        U = np.random.uniform()
        if U > alpha:
            continue
        reversebility_check, flag = project(q, Y + v_, Gy, grad, m)
        if not flag:
            continue
        X[i + 1] = Y
        accepted += 1

    # pbar.close()
    # print("Acceptance probability: " + "{0:.2%}".format(accepted / N) + '\n')
    return X, accepted / N
