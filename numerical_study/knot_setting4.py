"""Knot sensitivity on the paper's own ODE-Flex setting (Setting 4), quantile knots.

Setting 4: alpha(t) = t+1, q(u) = 2/(1+u) -- both functional parameters free, and the
setting the paper itself fits with quartile-placed cubic B-splines and the constraints
beta_1 = 1, alpha(2) = 1. Unlike a non-monotone q, this target is smooth and estimable
at moderate n, so a sensitivity study on it measures the effect of the KNOT COUNT rather
than the inability of any sieve to represent the truth.

Knots are placed at quantiles for both functional parameters:
  log alpha : quantiles of the distinct observed times
  log q     : quantiles of temp2 (a proxy for the occupation measure of the mu paths)
Quantile placement spans the full support by construction -- its outermost knots are the
data min and max -- so the sup-norm approximation condition behind Theorem 2 is not
violated, which is not true of a truncated range.

Reports, per sieve exponent p in #knots = ceil(m^p): BIC, the accuracy of the fitted
functions against the truth, and the coefficient bias.
"""
from __future__ import annotations

import argparse, contextlib, io, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

from ode_unify._engine.common import augknt, spcol                     # noqa: E402
from ode_unify._engine.ltm.cox_rec import cox_rec                      # noqa: E402
from ode_unify.paper_dgp import simulate_paper                       # noqa: E402
import lr_bootstrap_equal as LR                                       # noqa: E402

SETTING = int(os.environ.get('KNOT_SETTING', 4))
ANCHOR_T = 2.0                      # the paper's alpha(2) = 1
# Setting 4 (paper's Flex): both functional parameters smooth.
# Setting 7: same alpha, but a non-monotone q that minimal knots cannot represent.
_TRUE = {
    4: (lambda t: np.asarray(t, float) + 1.0,
        lambda u: 2.0 / (1.0 + np.asarray(u, float))),
    7: (lambda t: np.asarray(t, float) + 1.0,
        lambda u: 1.0 + 0.7 * np.sin(1.5 * np.asarray(u, float))),
    8: (lambda t: 1.0 + np.sqrt(np.maximum(np.asarray(t, float), 0.0))
                  + np.asarray(t, float) ** 2,
        lambda u: 1.0 / (1.0 + np.sqrt(np.maximum(np.asarray(u, float), 0.0)))),
}
TRUE_ALPHA, TRUE_Q = _TRUE[SETTING]
NG = 80


def _one(args):
    seed, N, p_exp = args
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            d = simulate_paper(N, seed, SETTING)
            x = np.ascontiguousarray(d['x']); t = d['time'].ravel()
            delta = d['delta'].ravel(); idv = d['id'].ravel().astype(int)
            temp2, binit = cox_rec(x, t, delta)
            ut = np.unique(t)
            # ONE interior-knot count, used for both functional parameters, from the
            # paper's base N = total number of observed event times. Both sieves also
            # use the same spline order (LR.K0 == LR.KQ), so alpha and q differ only in
            # where their knots sit, never in how many or of what degree.
            n_ev = int((delta == 1).sum())
            k_int = max(int(np.ceil(n_ev ** p_exp)), 2)
            l0 = k_int
            # quantile placement for BOTH functional parameters
            kn0 = augknt(np.quantile(np.concatenate([[0.0], ut]),
                                     np.linspace(0, 1, k_int + 1)), LR.K0)
            t2 = np.concatenate([[0.0], np.asarray(temp2).ravel()])
            knq = augknt(np.quantile(t2, np.linspace(0, 1, k_int + 1)), LR.KQ)
            nll, beta, theta, alpha = LR._fit(x, t, delta, idv, binit, kn0, knq, True)
        q_0, q_q = len(kn0) - LR.K0, len(knq) - LR.KQ
        n = int((delta == 0).sum())
        npar = (x.shape[1] - 1) + q_0 + q_q
        bic = 2 * n * nll + npar * np.log(n)

        gt = np.linspace(0.0, float(ut.max()), NG)
        gu = np.linspace(0.0, float(np.quantile(t2, 0.95)), NG)
        # ODE-Flex is the ONLY case with a genuine scale freedom, alpha -> k*alpha and
        # q -> q/k, and it is ONE-dimensional. Estimate a single log k by aligning
        # alpha at ANCHOR_T, then apply it to BOTH curves: subtract from log alpha and
        # ADD to log q. Anchoring the two curves independently spends two parameters on
        # a one-parameter freedom and hides real error in q.
        log_k = (float(spcol(kn0, LR.K0, np.array([ANCHOR_T]))[0] @ alpha)
                 - float(np.log(TRUE_ALPHA(ANCHOR_T))))
        la = spcol(kn0, LR.K0, gt) @ alpha - log_k
        lq = spcol(knq, LR.KQ, gu) @ theta + log_k
        tla = np.log(TRUE_ALPHA(gt))
        tlq = np.log(TRUE_Q(gu))
        return dict(seed=seed, p=p_exp, l0=k_int, lq_n=k_int, npar=npar,
                    negll=nll, BIC=bic, ok=True,
                    err_alpha=float(np.abs(la - tla).max()),
                    err_q=float(np.abs(lq - tlq).max()),
                    rmse_alpha=float(np.sqrt(np.mean((la - tla) ** 2))),
                    rmse_q=float(np.sqrt(np.mean((lq - tlq) ** 2))),
                    b2=float(beta[0]), b3=float(beta[1]),
                    la=la.tolist(), lqc=lq.tolist(),
                    gt=gt.tolist(), gu=gu.tolist(),
                    tla=tla.tolist(), tlq=tlq.tolist())
    except Exception:
        return dict(seed=seed, p=p_exp, ok=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--p', nargs='*', type=float,
                    default=[0.10, 0.14, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument('--tag', default='s4')
    a = ap.parse_args(argv)
    RES = os.path.join(HERE, 'bic_model_selection', 'results')
    os.makedirs(RES, exist_ok=True)
    tasks = [(s, a.n, p) for p in a.p for s in range(1, a.reps + 1)]
    print(f'Setting {SETTING}, quantile knots, n={a.n}: {len(tasks)} fits '
          f'({len(a.p)} sieve sizes x {a.reps} reps)', flush=True)
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 20 == 0:
                print(f'  {i}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    ok = [r for r in res if r.get('ok')]
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ('la', 'lqc', 'gt', 'gu', 'tla', 'tlq')}
                       for r in ok])
    df.to_csv(os.path.join(RES, f'knot_{a.tag}.csv'), index=False)
    g = df.groupby('p').agg(knots_a=('l0', 'first'), knots_q=('lq_n', 'first'),
                            npar=('npar', 'first'), ok=('BIC', 'size'),
                            BIC=('BIC', 'median'),
                            err_alpha=('err_alpha', 'median'),
                            err_q=('err_q', 'median'),
                            rmse_alpha=('rmse_alpha', 'median'),
                            rmse_q=('rmse_q', 'median'),
                            bias_b2=('b2', lambda s: s.median() - 1.0),
                            bias_b3=('b3', lambda s: s.median() - 1.0)).reset_index()
    g['dBIC'] = (g.BIC - g.BIC.min()).round(1)
    print('\n' + g[['p', 'knots_a', 'knots_q', 'npar', 'ok', 'BIC', 'dBIC',
                    'rmse_alpha', 'rmse_q', 'err_alpha', 'err_q',
                    'bias_b2', 'bias_b3']]
          .round(4).to_string(index=False))
    # Accuracy is reported by MEAN-SQUARED error (RMSE on the log scale); the
    # sup-norm columns are kept only for reference. RMSE is the better-behaved
    # metric -- the sup norm is driven by a single grid point, usually at an
    # endpoint of the support where the spline is least determined.
    print(f'\nBIC-optimal sieve:  p = {g.p[g.BIC.idxmin()]:.2f}')
    print(f'RMSE-optimal sieve: p = {g.p[g.rmse_alpha.idxmin()]:.2f} (log alpha), '
          f'p = {g.p[g.rmse_q.idxmin()]:.2f} (log q)')

    ps = sorted({r['p'] for r in ok})
    A = np.array([np.median([r['la'] for r in ok if r['p'] == p], 0) for p in ps])
    Q = np.array([np.median([r['lqc'] for r in ok if r['p'] == p], 0) for p in ps])
    r0 = ok[0]
    np.savez_compressed(os.path.join(RES, f'knot_{a.tag}_curves.npz'),
                        ps=np.array(ps), med_alpha=A, med_q=Q,
                        gt=np.array(r0['gt']), gu=np.array(r0['gu']),
                        true_la=np.array(r0['tla']), true_lq=np.array(r0['tlq']))
    sa = max(np.abs(A[i] - A[j]).max() for i in range(len(ps)) for j in range(len(ps)))
    sq = max(np.abs(Q[i] - Q[j]).max() for i in range(len(ps)) for j in range(len(ps)))
    print(f'max spread of the median curve across sieve sizes: '
          f'log alpha {sa:.4f}   log q {sq:.4f}')
    print(f'wrote knot_{a.tag}.csv and knot_{a.tag}_curves.npz  '
          f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
