"""Re-estimate the Zeng-Lin resampling sandwich of the ODE-Flex fit with a given number of
draws and seed, and use it for the standard errors and pointwise bands reported in the
paper (B = 8000, seed = 1). The re-estimated covariance replaces the one stored by
fit_final_horizon.py in <OUT>/final_<tag>.npz and <OUT>/final_<tag>_coef.csv; a copy is
also kept in <OUT>/final_<tag>_bands_B<B>_s<seed>.npz.

Usage:  python resample_bands.py [tag] [B] [seed]      (paper: 2y 8000 1)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.linalg import null_space

import fit_final_horizon as F

tag = sys.argv[1] if len(sys.argv) > 1 else '2y'
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
z = np.load(os.path.join(F.OUT, f'final_{tag}.npz'), allow_pickle=True)
X, t, delta, idv, _ = F.load(tag)
kn0, knq, K = z['knots_0'], z['knots_q'], int(z['k'])
Xs = X.copy()
Xs[:, 0] *= float(z['c0'])
be, th, al = z['beta_flex'], z['theta'], z['alpha_flex']
p, q_q = X.shape[1], th.size
row = F.spcol(kn0, K, np.array([float(z['t_anchor'])]))[0]
est = np.concatenate([be, th, al])
d = est.size
crow = np.zeros(d)
crow[p - 1 + q_q:] = row
Nul = null_space(crow.reshape(1, -1))
N = int(idv.max())
g = F.IOFS(np.concatenate([th, al]), Xs, t, delta, idv, be, kn0, knq, K, K, ci=True)
V = g.T @ g / N
rng = np.random.default_rng(seed)
Zr = rng.standard_normal((Nul.shape[1], B))
Y = np.zeros((B, d))
for i in range(B):
    e = est + Nul @ Zr[:, i] / np.sqrt(N)
    Y[i] = F.IOFS(np.concatenate([e[p - 1:p - 1 + q_q], e[p - 1 + q_q:]]), Xs, t, delta,
                  idv, e[:p - 1], kn0, knq, K, K, ci=False) / np.sqrt(N)
    if (i + 1) % 1000 == 0:
        print(f'  {i + 1}/{B}', flush=True)
Y -= Y.mean(0)
H = np.linalg.solve(Zr @ Zr.T, Zr @ Y) @ Nul
H = 0.5 * (H + H.T)
Hi = np.linalg.inv(H)
Sigma = Nul @ (Hi @ (Nul.T @ V @ Nul) @ Hi / N) @ Nul.T
Bq = F.spcol(knq, K, z['gu'])
B0 = F.spcol(kn0, K, z['gt'])
se_q = np.sqrt(np.abs(np.einsum('ij,jk,ik->i', Bq, Sigma[p - 1:p - 1 + q_q, p - 1:p - 1 + q_q], Bq)))
se_a = np.sqrt(np.abs(np.einsum('ij,jk,ik->i', B0, Sigma[p - 1 + q_q:, p - 1 + q_q:], B0)))
np.savez_compressed(os.path.join(F.OUT, f'final_{tag}_bands_B{B}_s{seed}.npz'), Sigma=Sigma,
                    se_q=se_q, se_alpha=se_a, se_flex=np.sqrt(np.abs(np.diag(Sigma))), B=B, seed=seed)
final = dict(z)
final.update(Sigma=Sigma, se_flex=np.sqrt(np.abs(np.diag(Sigma))), se_q=se_q, se_alpha=se_a,
             B=B, band_seed=seed)
np.savez_compressed(os.path.join(F.OUT, f'final_{tag}.npz'), **final)
coef_path = os.path.join(F.OUT, f'final_{tag}_coef.csv')
if os.path.exists(coef_path):
    import pandas as pd
    from scipy.stats import norm
    tab = pd.read_csv(coef_path)
    se = np.concatenate([[np.nan], final['se_flex'][:p - 1]])
    tab['flex_se'] = se
    tab['flex_p'] = 2 * norm.sf(np.abs(tab['flex_est'] / se))
    tab.to_csv(coef_path, index=False)
print(f'updated final_{tag}.npz and final_{tag}_coef.csv with B={B}, seed={seed}', flush=True)
