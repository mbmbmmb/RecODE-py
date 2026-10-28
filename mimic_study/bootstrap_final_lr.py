"""Parametric bootstrap of the LR test of H0: q == 1 for the final MIMIC-III analysis.

Every replicate regenerates the readmission process from the fitted ODE-Cox null
(non-homogeneous Poisson with mean exp(x'beta_cox) A_cox(t)), holding X_i and C_i
fixed, and re-runs the WHOLE procedure of fit_final_horizon.py on it: knot
placement, the free ODE-Cox fit, its own c0 and alpha anchor, and the ODE-Flex fit.

Usage:  python bootstrap_final_lr.py [tag] [B] [workers]      (paper: 2y 200 10)
"""
from __future__ import annotations

import os
import sys
import time as _time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from multiprocessing import Pool
from scipy.integrate import cumulative_trapezoid

import fit_final_horizon as F

TAG = sys.argv[1] if len(sys.argv) > 1 else '2y'
B = int(sys.argv[2]) if len(sys.argv) > 2 else 500
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 10


def lr_procedure(X, t, delta, beta_start=None, log_alpha_start=None):
    """Knots -> free ODE-Cox -> anchored ODE-Flex -> LR (as in fit_final_horizon)."""
    m = int((delta == 0).sum())
    p = X.shape[1]
    n_int = int(np.ceil(m ** (1 / 7)))
    horizon = float(t.max())
    kn0 = F.quantile_knots(t[delta == 1], 0.0, horizon, n_int)
    q_0 = len(kn0) - F.K
    X0 = np.column_stack([np.zeros(len(t)), X])
    knq_d = F.augknt(np.array([0.0, 1.0]), F.K)
    # Warm start at the generating null (projected onto this replicate's knots):
    # it changes only how fast the fit converges, not where it converges.
    b0 = np.zeros(p) if beta_start is None else np.asarray(beta_start, float)
    a0 = np.zeros(q_0)
    if log_alpha_start is not None:
        gg = np.linspace(0.0, horizon, 400)
        a0 = np.linalg.lstsq(F.spcol(kn0, F.K, gg), log_alpha_start(gg), rcond=None)[0]
    cox = F.alternate(X0, t, delta, b0, np.zeros(len(knq_d) - F.K),
                      a0, kn0, knq_d, free_theta=False, label='b-cox')
    b_cox, a_cox = cox['beta'], cox['alpha']
    u, bi = np.unique(t, return_inverse=True)
    grid = np.concatenate([[0.0], u])
    rate = np.exp(F.spcol(kn0, F.K, grid) @ a_cox)
    A = cumulative_trapezoid(rate, grid, initial=0.0)[1:]
    mu = A[bi] * np.exp(X @ b_cox)
    knq = F.quantile_knots(mu[delta == 0], 0.0, 1.05 * float(mu.max()), n_int)
    q_q = len(knq) - F.K
    Xs = X.copy()
    Xs[:, 0] *= b_cox[0]
    row = F.spcol(kn0, F.K, np.array([float(np.median(t[delta == 1]))]))[0]
    nl_seed, _ = F.OFS(np.concatenate([np.zeros(q_q), a_cox]), Xs, t, delta,
                       b_cox[1:], kn0, knq, F.K, F.K)
    flex = F.fit_flex(Xs, t, delta, b_cox[1:], a_cox, kn0, knq,
                      (row, float(row @ a_cox)), starts=1, workers=1)
    return 2 * m * (nl_seed - flex['negll']), cox['converged'], flex['converged']


def one(b):
    z = np.load(os.path.join(F.OUT, f'final_{TAG}.npz'), allow_pickle=True)
    X, t, delta, idv, _ = F.load(TAG)
    Xb = X[delta == 0]                       # one row per patient (its censoring row)
    C = t[delta == 0]
    kn0, a_cox, b_cox = z['knots_0'], z['alpha_cox'], z['beta_cox']
    grid = np.linspace(0.0, float(C.max()), 20001)
    A = cumulative_trapezoid(np.exp(F.spcol(kn0, F.K, grid) @ a_cox), grid, initial=0.0)
    rng = np.random.default_rng(10_000 + b)
    lam = np.exp(Xb @ b_cox)
    AC = np.interp(C, grid, A)
    n_ev = rng.poisson(lam * AC)
    rows_t, rows_d, rows_x = [], [], []
    for i in np.nonzero(n_ev)[0]:
        ev = np.sort(np.interp(rng.uniform(0, AC[i], n_ev[i]), A, grid))
        rows_t.append(ev); rows_d.append(np.ones(n_ev[i], int))
        rows_x.append(np.repeat(Xb[i:i + 1], n_ev[i], axis=0))
    tt = np.concatenate(rows_t + [C])
    dd = np.concatenate(rows_d + [np.zeros(len(C), int)])
    xx = np.vstack(rows_x + [Xb])
    t0 = _time.time()
    try:
        lr, cc, fc = lr_procedure(xx, tt, dd, beta_start=b_cox,
                                  log_alpha_start=lambda g: F.spcol(kn0, F.K, g) @ a_cox)
    except Exception as e:                   # record, never drop silently
        return b, np.nan, False, False, repr(e)[:120]
    return b, lr, cc, fc, f'{_time.time() - t0:.0f}s'


def main():
    out = os.path.join(F.OUT, f'final_{TAG}_bootstrap.csv')
    done = set()
    if os.path.exists(out):
        done = {int(l.split(',')[0]) for l in open(out).read().splitlines()[1:] if l}
    else:
        open(out, 'w').write('b,LR,cox_conv,flex_conv,note\n')
    todo = [b for b in range(B) if b not in done]
    print(f'{len(done)} done, {len(todo)} to go, {WORKERS} workers', flush=True)
    with Pool(WORKERS) as pool:
        for k, (b, lr, cc, fc, note) in enumerate(pool.imap_unordered(one, todo), 1):
            with open(out, 'a') as fh:
                fh.write(f'{b},{lr},{cc},{fc},{note}\n')
            if k % 25 == 0:
                print(f'  {k}/{len(todo)}', flush=True)
    z = np.load(os.path.join(F.OUT, f'final_{TAG}.npz'))
    import pandas as pd
    d = pd.read_csv(out)
    ok = d.LR.notna()
    lr0 = float(z['LR'])
    p = (1 + (d.LR[ok] >= lr0).sum()) / (1 + ok.sum())
    print(f'observed LR={lr0:.2f}; {ok.sum()}/{len(d)} replicates ok; '
          f'null median={d.LR[ok].median():.2f} q90={d.LR[ok].quantile(.9):.2f} '
          f'q99={d.LR[ok].quantile(.99):.2f} max={d.LR[ok].max():.2f}; bootstrap p={p:.4f}')


if __name__ == '__main__':
    main()
