import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as st
sns.set()


def get_K(auto_covars):
    K = None
    for k in range(int(len(auto_covars) / 2)):
        if (auto_covars[2 * k] + auto_covars[2 * k + 1]) <= 0:
            K = k
            break
    if K is None:
        K = int(len(auto_covars) / 2 - 1)
    return K


def compute_auto_covars(Xchain):
    Xmean = np.mean(Xchain)
    N = len(Xchain)
    autocov = np.zeros(N)
    for k in range(N):
        for i in range(N - k):
            autocov[k] += ((Xchain[i + k]) - Xmean) * (Xchain[i] - Xmean)
        autocov[k] = (1 / (N - 1)) * autocov[k]
    return autocov


def sigma_ipse(auto_covars):
    K = get_K(auto_covars)
    return -auto_covars[0] + 2 * np.sum([auto_covars[2 * k] + auto_covars[2 * k + 1] for k in range(1, K + 1)])


def sigma_bm(Xchain, a=0.5):
    n = len(Xchain)
    Nb = int(np.floor(n ** a))
    Nl = int(np.floor(n / Nb))
    batch_means = []
    mcmc_mean = np.mean(Xchain)
    for i in range(1, Nb + 1):
        batch_means.append(np.mean([Xchain[j] for j in range((i - 1) * Nl, i * Nl)]))

    batch_means = np.array(batch_means)
    return np.sum((batch_means - mcmc_mean) ** 2) * n / (Nb - 1) / Nb


r = 0.5
R = 1
mcmc_name = './chains/' + 'mcmc_10_6_torus.npy'
cmc_name = './chains/' + 'cmc_10_6_torus.npy'
X = np.load(mcmc_name)
X = X[0::10]
Y = np.load(cmc_name)
# Y = Y[0::10]
lens = np.linspace(500, 100000, 300, dtype=int)


Z = 4 * np.pi**2 * r * R
true_value = (1.25 * r**2 + 0.5 * R**2) * Z
Z_mcmc = np.zeros(len(lens))
Z_cmc = np.zeros(len(lens))
for i in range(len(lens)):
    Z_mcmc[i] = np.mean(X[0:lens[i], 1]**2 + X[0:lens[i], 2]**2) * Z
    Z_cmc[i] = np.mean(Y[0:lens[i], 1]**2 + Y[0:lens[i], 2]**2) * Z

errors_mcmc = []
errors_cmc = []
vars_mcmc = []
vars_cmc = []

alpha = 0.05
coeff = st.norm.ppf(1 - alpha / 2)
pbar = tqdm(lens, desc='Iterations covered')
for n in pbar:
    Xchain_n = (X[0:n, 1]**2 + X[0:n, 2]**2) * Z
    # auto_covars = compute_auto_covars(Xchain_n)
    # sigma_mcmc = sigma_ipse(auto_covars)
    sigma_mcmc = np.sqrt(sigma_bm(Xchain_n))
    error_mcmc = coeff * sigma_mcmc / np.sqrt(n)
    errors_mcmc.append(error_mcmc)
    vars_mcmc.append(sigma_mcmc**2 / n)

    sigma_cmc = np.std((Y[0:n, 1]**2 + Y[0:n, 2]**2) * Z)
    error_cmc = coeff * sigma_cmc / np.sqrt(n)
    errors_cmc.append(error_cmc)
    vars_cmc.append(sigma_cmc ** 2 / n)

true_values = np.full(len(lens), true_value)
plt.subplots()
plt.plot(lens, true_values, label="True value", color='red', linestyle='--', linewidth=1)
plt.plot(lens, Z_mcmc, label="MCMC")
plt.fill_between(lens, Z_mcmc - errors_mcmc, Z_mcmc + errors_mcmc, color='b', alpha=0.2)
plt.plot(lens, Z_cmc, label="Crude MC")
plt.fill_between(lens, Z_cmc - errors_cmc, Z_cmc + errors_cmc, color='orange', alpha=0.2)
plt.legend()
plt.subplots()
plt.loglog(lens, np.abs(true_values - Z_mcmc), label='Error for MCMC')
plt.loglog(lens, np.abs(true_values - Z_cmc), label='Error for CMC')
plt.loglog(lens, np.power(lens, -0.5), color='red', linestyle='--', label='$N^{-1/2}$')
plt.legend()
plt.subplots()
plt.loglog(lens, errors_mcmc, label='MCMC errors')
plt.loglog(lens, errors_cmc, label='CMC errors')
plt.loglog(lens, lens ** -0.5, '--', color='red', label=r'$N^{-1/2}$')
plt.legend()
plt.subplots()
plt.loglog(lens, vars_mcmc, label='MCMC variance')
plt.loglog(lens, vars_cmc, label='CMC variance')
plt.loglog(lens, 1. / lens, '--', color='red', label=r'$N^{-1}$')
plt.legend()
plt.show()