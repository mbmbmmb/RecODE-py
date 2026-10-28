"""Parametric-bootstrap likelihood ratio test of the Cox-type restriction
(Section 5.4 and Table 6 of the paper; Section D.4 of the Supplementary Material).

H0: q == 1 (ODE-Cox)   against   H1: q unspecified (ODE-Flex).

Both models are nested restrictions of one sieve parameterisation, so the statistic is

    LR_n = 2 n { l_n(beta, a, b) - l_n(beta_0, a_0, 0) } .

Its null distribution is not chi-squared: the nuisance parameters are
infinite-dimensional and the sieve grows with n. Following the generalized likelihood
ratio framework of Fan, Zhang and Zhang (2001), the null distribution is approximated by
a parametric bootstrap from the fitted null, holding the covariates and censoring times
fixed. For the size/power study we use the warp-speed bootstrap of Giacomini, Politis
and White (2013): one bootstrap draw per Monte Carlo replication, pooled across
replications to form the reference distribution.

Sieve: quadratic B-splines for log alpha and log q, each with ceil(m^(1/7)) intervals
whose knots are placed at sample quantiles (event times for alpha, fitted ODE-Cox
mean functions for q), where m is the number of subjects.

Settings: 1 (H0 true), 7 (H1 true: alpha = t + 1, q = 1 + 0.7 sin(1.5u)) and
3 (Box-Cox transformation model, outside the sieve; Section D.4).

Usage (from this directory):

    python lr_bootstrap.py --reps 300 --grid 1:1000 3:1000 7:1000 \
        --out bic_model_selection/results/lr_bootstrap.csv
    python lr_bootstrap.py --reps 200 --grid 1:2000 7:2000 1:4000 7:4000 \
        --out bic_model_selection/results/lr_settings_table.csv
"""
from __future__ import annotations

import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ode_unify._engine.common import augknt, spcol, solve_sieve_step   # noqa: E402
from ode_unify._engine.ltm.cox_rec import cox_rec                      # noqa: E402
from ode_unify._engine.ltm.objective_func_sieve import objective_func_sieve as OFS  # noqa: E402,E501
from ode_unify._engine.ltm.objective_func_beta import objective_func_beta as OFB    # noqa: E402,E501
from ode_unify.dgp import _intensity_factory                         # noqa: E402
from ode_unify.paper_dgp import simulate_paper                       # noqa: E402
from scipy.optimize import LinearConstraint, minimize                # noqa: E402

K0 = KQ = 3
ALPHA_ANCHOR = 2.0
ITERS = 60


def _setup(x, t, delta, aex=4):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        temp2, binit = cox_rec(x, t, delta)
    # CORRECTED SIEVE. The original used l0 = ceil(#unique t ^ 0.2) + 4 and
    # lq = ceil(#ROWS ^ 0.2), both EQUALLY SPACED, with the q knots spread over
    # [0, 2*max(mu)] -- so about half the q basis functions had no data behind
    # them, and alpha was sized from rows rather than from subjects. Here both
    # functional parameters get the SAME number of interior knots, ceil(m^(1/7))
    # with m = number of SUBJECTS, placed at QUANTILES over the realised support.
    m = int(np.sum(delta == 0))
    nk = int(np.ceil(m ** (1.0 / 7.0)))
    u = np.linspace(0.0, 1.0, nk + 1)
    knots_0 = augknt(np.quantile(np.concatenate([[0.0], t[delta == 1]]), u), K0)
    knots_q = augknt(np.quantile(np.concatenate([[0.0], temp2.ravel()]), u), KQ)
    return binit, knots_0, knots_q


def _fit(x, t, delta, idv, binit, knots_0, knots_q, free_theta):
    """Alternating fit; free_theta=False gives ODE-Cox, True gives ODE-Flex."""
    q_0, q_q = len(knots_0) - K0, len(knots_q) - KQ
    Aeq = spcol(knots_0, K0, np.array([ALPHA_ANCHOR]))[0]
    beta = (binit / binit[0])[1:].copy()
    theta, alpha = np.zeros(q_q), np.zeros(q_0)
    use_trust = False
    for _ in range(ITERS):
        bp, tp, ap = beta.copy(), theta.copy(), alpha.copy()

        def f(v):
            th = v[:q_q] if free_theta else theta
            al = v[q_q:] if free_theta else v
            val, g = OFS(np.concatenate([th, al]), x, t, delta, idv, beta,
                         knots_0, knots_q, K0, KQ)
            return val, (g if free_theta else g[q_q:])

        v0 = np.concatenate([theta, alpha]) if free_theta else alpha
        # The anchor alpha(2) = 1 only pins the scale freedom (alpha -> k alpha,
        # q -> q/k) of the general transformation model.  Under Cox, q == 1 is
        # fixed, alpha is already identified, and the anchor would remove a real
        # degree of freedom -- so impose it only for the Flex fit.
        if free_theta:
            row = np.zeros(v0.size); row[q_q:] = Aeq
            cons = [LinearConstraint(row.reshape(1, -1), 0., 0.)]
        else:
            cons = []
        r, use_trust = solve_sieve_step(f, v0, cons,
                                        prefer_trust=use_trust,
                                        slsqp_options={'maxiter': 120, 'ftol': 1e-8})
        if free_theta:
            theta, alpha = r.x[:q_q], r.x[q_q:]
        else:
            alpha = r.x
        rb = minimize(lambda bb: OFB(bb, x, t, delta, idv, theta, alpha,
                                     knots_0, knots_q, K0, KQ),
                      beta, jac=True, method='BFGS', options={'maxiter': 400})
        beta = rb.x
        if max(np.max(np.abs(beta - bp)), np.max(np.abs(theta - tp)),
               np.max(np.abs(alpha - ap))) < 1e-4:
            break
    val, _ = OFS(np.concatenate([theta, alpha]), x, t, delta, idv, beta,
                 knots_0, knots_q, K0, KQ)
    return float(val), beta, theta, alpha


def _lr(x, t, delta, idv, aex=4):
    """LR = 2 n (l_flex - l_cox); n = number of subjects."""
    binit, kn0, knq = _setup(x, t, delta, aex)
    nll_cox, b0, _, a0 = _fit(x, t, delta, idv, binit, kn0, knq, False)
    nll_flex, *_ = _fit(x, t, delta, idv, binit, kn0, knq, True)
    n = int((delta == 0).sum())
    return 2 * n * (nll_cox - nll_flex), b0, a0, kn0


def _regen_from_null(x, u, beta0, alpha_coef, knots_0, seed):
    """Regenerate the counting process from the fitted ODE-Cox model, holding the
    covariates x and the censoring times u fixed."""
    def a_fun(tt):
        arr = np.atleast_1d(np.asarray(tt, dtype=float))
        v = np.exp(spcol(knots_0, K0, arr) @ alpha_coef)
        return v if np.ndim(tt) else float(v[0])
    make_lam = _intensity_factory(None, None, a_fun, None, 0.5, None)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(x.shape[0]):
        sub = np.random.default_rng(seed * (i + 1) + 7)
        m = float(np.exp(x[i] @ beta0)); ui = float(u[i])
        lam = make_lam(m, ui)
        gt = np.linspace(0.0, ui, 200)
        lmax = max(float(np.max(lam(gt))), 1e-9)
        ev, s = [], 0.0
        while s < ui:
            s = s - np.log(sub.random()) / lmax
            if s < ui and sub.random() <= float(lam(s)) / lmax:
                ev.append(s)
        for e in ev:
            rows.append((i + 1, e, 1, *x[i]))
        rows.append((i + 1, ui, 0, *x[i]))
    arr = np.asarray(rows, float)
    return (np.ascontiguousarray(arr[:, 3:]), arr[:, 1], arr[:, 2].astype(int),
            arr[:, 0].astype(int))


def _one(args):
    setting, seed, N, aex = args
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            d = simulate_paper(N, seed, setting)
            x = np.ascontiguousarray(d['x']); t = d['time'].ravel()
            delta = d['delta'].ravel(); idv = d['id'].ravel().astype(int)
            # observed statistic and the fitted null
            lr_obs, b0, a0, kn0 = _lr(x, t, delta, idv, aex)
            # one warp-speed bootstrap draw from the fitted null
            u = np.array([t[idv == i].max() for i in np.unique(idv)])
            xs = np.array([x[idv == i][0] for i in np.unique(idv)])
            xb, tb, db, ib = _regen_from_null(xs, u, np.concatenate([[1.0], b0]),
                                              a0, kn0, seed + 10 ** 6)
            lr_boot, *_ = _lr(xb, tb, db, ib, aex)
        return setting, N, seed, float(lr_obs), float(lr_boot)
    except Exception:
        return setting, N, seed, np.nan, np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--grid', nargs='*', default=['1:1000', '3:1000', '7:1000'])
    ap.add_argument('--alpha-knots-extra', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(HERE, 'bic_model_selection',
                                                  'results', 'lr_bootstrap.csv'))
    a = ap.parse_args(argv)
    grid = [tuple(int(v) for v in g.split(':')) for g in a.grid]
    tasks = [(s, seed, N, a.alpha_knots_extra)
             for s, N in grid for seed in range(1, a.reps + 1)]
    print(f'{len(tasks)} replicates ({len(grid)} configs x {a.reps}), '
          f'{a.workers} workers', flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0:
                print(f'  {i}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows, columns=['setting', 'n', 'seed', 'lr_obs', 'lr_boot'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f'\n{"setting":>8s}{"n":>7s}{"ok":>5s}{"0.10":>8s}{"0.05":>8s}{"0.01":>8s}')
    for (s, N) in grid:
        sub = df[(df.setting == s) & (df.n == N)].dropna()
        if not len(sub):
            continue
        null = sub.lr_boot.values           # pooled warp-speed null draws
        rej = {lv: float(np.mean([(1 + (null >= o).sum()) / (len(null) + 1) <= lv
                                  for o in sub.lr_obs.values]))
               for lv in (0.10, 0.05, 0.01)}
        print(f'{s:>8d}{N:>7d}{len(sub):>5d}'
              f'{rej[0.10]:>8.3f}{rej[0.05]:>8.3f}{rej[0.01]:>8.3f}')
    print(f'\nwrote {a.out}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
