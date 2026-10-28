"""Where should the knots for log q(u) be placed?

q(u) is evaluated at mu(t) along every subject's trajectory, so the design measure that
matters is the OCCUPATION measure of the mu paths -- how much time the processes spend
near each value of u -- not the distribution of their endpoints, and certainly not the
range [0, 2*max]. Three placements are compared on Setting 7, whose q(u) = 1+0.7sin(1.5u)
is non-monotone and so cannot be recovered by a sieve whose knots miss the region where
it varies:

  equal      knots equally spaced on [0, 2*max(temp2)]        (the current default)
  quantile   knots at quantiles of temp2 over all rows        (proxy for the occupation measure)
  trunc      knots equally spaced on [0, q95(temp2)]          (max-driven range, tail removed)

Accuracy is measured on the realised support of mu rather than an arbitrary grid.
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
from ode_unify.paper_dgp import CUSTOM_FUNCS, simulate_paper         # noqa: E402
import lr_bootstrap_equal as LR                                       # noqa: E402

SETTING, CF = 7, CUSTOM_FUNCS[7]
NG = 60


def q_knots(kind, temp2, lq, kq):
    t2 = np.concatenate([[0.0], np.asarray(temp2).ravel()])
    if kind == 'equal':
        return augknt(np.linspace(0.0, 2 * float(t2.max()), lq + 1), kq)
    if kind == 'quantile':
        return augknt(np.quantile(t2, np.linspace(0, 1, lq + 1)), kq)
    if kind == 'trunc':
        return augknt(np.linspace(0.0, float(np.quantile(t2, 0.95)), lq + 1), kq)
    if kind == 'maxrange':
        # equal spacing on [0, max(mu-hat)] -- the obvious repair of 'equal', and what
        # the engine's K2/K4 settings approximate by using max as the upper endpoint.
        return augknt(np.linspace(0.0, float(t2.max()), lq + 1), kq)
    raise ValueError(kind)


def _one(args):
    seed, N, p_exp, kind = args
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            d = simulate_paper(N, seed, SETTING)
            x = np.ascontiguousarray(d['x']); t = d['time'].ravel()
            delta = d['delta'].ravel(); idv = d['id'].ravel().astype(int)
            temp2, binit = cox_rec(x, t, delta)
            l0 = max(int(np.ceil(len(np.unique(t)) ** p_exp)), 2)
            lq = max(int(np.ceil(x.shape[0] ** p_exp)), 2)
            kn0 = augknt(np.linspace(0, float(np.max(t)), l0 + 1), LR.K0)
            knq = q_knots(kind, temp2, lq, LR.KQ)
            nll, beta, theta, alpha = LR._fit(x, t, delta, idv, binit, kn0, knq, True)
        # evaluate on the FULL realised support. Scoring only up to a low
        # quantile would flatter any placement that truncates there, since the
        # spline clamps beyond its last knot and is never penalised for the
        # region it does not cover.
        t2 = np.asarray(temp2).ravel()
        gu = np.linspace(0.0, float(t2.max()), NG)
        lq_hat = spcol(knq, LR.KQ, gu) @ theta
        lq_hat -= float(spcol(knq, LR.KQ, np.array([0.0]))[0] @ theta)
        true = np.log(CF['q'](gu)) - np.log(CF['q'](np.array([0.0]))[0])
        return dict(seed=seed, p=p_exp, kind=kind, lq=lq, ok=True,
                    err_q=float(np.abs(lq_hat - true).max()),
                    rmse_q=float(np.sqrt(np.mean((lq_hat - true) ** 2))),
                    negll=nll, b2=float(beta[0]), umax=float(gu[-1]))
    except Exception:
        return dict(seed=seed, p=p_exp, kind=kind, ok=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--p', type=float, default=0.20)
    ap.add_argument('--kinds', nargs='*', default=['equal', 'quantile', 'trunc'])
    ap.add_argument('--out', default=os.path.join(HERE, 'bic_model_selection',
                                                  'results', 'knot_placement.csv'))
    a = ap.parse_args(argv)
    tasks = [(s, a.n, a.p, k) for k in a.kinds for s in range(1, a.reps + 1)]
    print(f'{len(tasks)} fits ({len(a.kinds)} placements x {a.reps} reps), '
          f'p={a.p}, n={a.n}', flush=True)
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 25 == 0:
                print(f'  {i}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame([r for r in res if r.get('ok')])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    g = df.groupby('kind').agg(ok=('err_q', 'size'), knots_q=('lq', 'first'),
                               u_max=('umax', 'median'),
                               err_q=('err_q', 'median'),
                               rmse_q=('rmse_q', 'median'),
                               bias_b2=('b2', lambda s: s.median() - 1.0)).reset_index()
    print('\n' + g.round(4).to_string(index=False))
    best = g.kind[g.rmse_q.idxmin()]
    print(f'\nbest placement by median RMSE in log q: {best}'
          f'   (by sup-error: {g.kind[g.err_q.idxmin()]})')
    print(f'wrote {a.out}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
