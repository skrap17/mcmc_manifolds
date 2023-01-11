import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from mcmc_manifolds import mcmc_manifold
from torus import G, q, crude_mc
import seaborn as sns
import scipy.stats as st
seed = 73
np.random.seed(seed)
sns.set()


def get_chain_len(tol, nmin, nmax, x0, f, *args):
    X, _ = mcmc_manifold(nmin, 3, 1, G, q, x0, 0.5)
    i = nmin
    k = 100
    # m = []
    # s = np.sqrt(sigma_bm(f(X, *args)))
    while np.sqrt(sigma_bm(f(X, *args)) / X.shape[0]) > tol and i < nmax:
        # chain = f(X, *args)
        s = np.sqrt(sigma_bm(f(X, *args)) / X.shape[0])
        print(s)
        # m.append(np.mean(f(X, *args)))
        Y = mcmc_manifold(k, 3, 1, G, q, X[-1], 0.5)[0][1:]
        X = np.vstack((X, Y))
        i += k
    return i


def get_chain_len_1(tol, nmin, nmax, f, *args):
    X, _ = crude_mc(nmin, R, r)
    k = 1
    i = nmin
    while np.sqrt(np.var(f(X, *args)) / X.shape[0]) > tol and i < nmax:
        s = np.sqrt(np.var(f(X, *args)) / X.shape[0])
        print(s)
        Y, _ = crude_mc(k, R, r)
        X = np.vstack((X, Y))
        i += k
    return i


def point_moment_of_inertia(X):
    return (X[:, 1]**2 + X[:, 2]**2) * 4 * np.pi**2 * r * R


def true_moment_of_inertia():
    return (1.25 * r ** 2 + 0.5 * R ** 2) * 4 * np.pi**2 * r * R


def sigma_bm(Xchain, a=0.8):
    n = len(Xchain)
    Nb = int(np.floor(n ** a))
    Nl = int(np.floor(n / Nb))
    batch_means = []
    mcmc_mean = np.mean(Xchain)
    for i in range(1, Nb + 1):
        batch_means.append(np.mean([Xchain[j] for j in range((i - 1) * Nl, i * Nl)]))

    batch_means = np.array(batch_means)
    # x =  np.sum((batch_means - mcmc_mean) ** 2) / (Nb - 1)
    # y = x   / Nb
    return np.sum((batch_means - mcmc_mean) ** 2) * n / (Nb - 1) / Nb


def plot_convergence(X, Y, lens, f, true_f):
    true_value = true_f()
    true_values = np.full(len(lens), true_value)

    Z_mcmc = []
    Z_cmc = []
    errors_mcmc = []
    errors_cmc = []
    vars_mcmc = []
    vars_cmc = []

    alpha = 0.05
    coeff = st.norm.ppf(1 - alpha / 2)
    pbar = tqdm(lens, desc='Iterations covered')
    for n in pbar:
        Xchain_n = f(X[0:n])
        Z_mcmc.append(np.mean(Xchain_n))
        sigma_mcmc = np.sqrt(sigma_bm(Xchain_n))
        error_mcmc = coeff * sigma_mcmc / np.sqrt(n)
        errors_mcmc.append(error_mcmc)
        vars_mcmc.append(sigma_mcmc ** 2 / n)

        Ychain_n = f(Y[0:n])
        Z_cmc.append(np.mean(Ychain_n))
        sigma_cmc = np.std(Ychain_n, ddof=1)
        error_cmc = coeff * sigma_cmc / np.sqrt(n)
        errors_cmc.append(error_cmc)
        vars_cmc.append(sigma_cmc ** 2 / n)

    pbar.close()
    Z_mcmc = np.array(Z_mcmc)
    Z_cmc = np.array(Z_cmc)
    plt.subplots()
    plt.plot(lens, true_values, label="True value", color='red', linestyle='--', linewidth=1)
    plt.plot(lens, Z_mcmc, label="MCMC")
    plt.fill_between(lens, Z_mcmc - errors_mcmc, Z_mcmc + errors_mcmc, color='b', alpha=0.2)
    plt.plot(lens, Z_cmc, label="Crude MC")
    plt.fill_between(lens, Z_cmc - errors_cmc, Z_cmc + errors_cmc, color='orange', alpha=0.2)
    plt.ylim(14, 17.5)
    plt.legend()
    plt.subplots()
    plt.loglog(lens, np.abs(true_values - Z_mcmc) / true_value, label='Error for MCMC')
    plt.loglog(lens, np.abs(true_values - Z_cmc) / true_value, label='Error for CMC')
    plt.loglog(lens, np.power(lens, -0.5), color='red', linestyle='--', label='$N^{-1/2}$')
    # plt.loglog(lens, np.power(lens, -1.), color='red', linestyle='--', label='$N^{-1}$')
    plt.legend()
    plt.subplots()
    plt.loglog(lens, vars_mcmc, label='MCMC variance')
    plt.loglog(lens, vars_cmc, label='CMC variance')
    plt.loglog(lens, 1. / lens, '--', color='red', label=r'$N^{-1}$')
    plt.legend()


r = 0.5
R = 1
N = 10**6
# X, _ = mcmc_manifold(N, 3, 1, G, q, np.array([R, 0, r]), 0.5)
# Y, _ = crude_mc(N, R, r)
# np.save("./chains/torus/mcmc/" + str(N) + '_' + str(seed) + ".npy", X)
# np.save("./chains/torus/cmc/" + str(N) + '_' + str(seed) + ".npy", Y)
mcmc_name = './chains/bad_names/' + 'mcmc_10_6_torus.npy'
cmc_name = './chains/bad_names/' + 'cmc_10_6_torus.npy'
# mcmc_name = './chains/torus/mcmc/' + '1000000_17.npy'
# cmc_name = './chains/torus/cmc/' + '1000000_17.npy'
X = np.load(mcmc_name)
# # X = X[0::10]
Y = np.load(cmc_name)
# # Y = Y[0::10]
# lens = np.linspace(500, N, 300, dtype=int)
lens = (10**np.linspace(2.7, 6, 300)).astype(int)
# plot_convergence(X, Y, lens, point_moment_of_inertia, true_moment_of_inertia)
# lens_1 = np.linspace(500, int(N / 10), 200, dtype=int)
lens_1 = (10**np.linspace(2.7, 5, 200)).astype(int)
# plot_convergence(X[0::10], Y[0::10], lens_1, point_moment_of_inertia, true_moment_of_inertia)
# plt.show()
# 357000
n = get_chain_len(0.2, 1000, 1000000, np.array([R, 0, r]), point_moment_of_inertia)

# n = get_chain_len_1(0.2, 1000, 1000000, point_moment_of_inertia)
print(n)