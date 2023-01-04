import numpy as np
from tqdm import tqdm


def project(q, z, Q, grad, dim, nmax=0, tol=0.001):
    a = np.zeros(dim)
    i = 0
    if nmax == 0:
        nmax = 3 * dim
    stop = False
    while not stop:
        arg = z + Q @ a
        q_arg = q(arg)
        try:
            da = np.linalg.solve(grad(arg).T @ Q, -q_arg)
        except:
            da = -np.linalg.pinv(grad(arg).T @ Q) @ q_arg
        a = a + da
        i += 1
        stop = np.linalg.norm(q_arg) < tol
        if i > nmax:
            return a, False
    return a, np.linalg.norm(q(z + Q @ a)) < tol


def mcmc_manifold(N, d, m, grad, q, x0, sigma, ineq_constraints=None):
    da = d - m
    X = np.zeros((N + 1, d))
    X[0] = x0
    accepted = 0
    pbar = tqdm(range(N), desc="Samples generated")
    for i in pbar:
        X[i + 1] = X[i]
        Gx = grad(X[i])
        qrx = np.linalg.qr(Gx, mode='complete')[0][:, m:]
        t = np.random.normal(size=da) * sigma
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

    pbar.write("Acceptance probability: " + "{0:.2%}".format(accepted / N) + '\n')
    return X, accepted / N
