"""Does the common-objective BIC select the true model?

Cox (q==1), AM (alpha==1) and Flex (both free) are exact nested restrictions of
the LTM sieve parameterisation, so fitting all three under ONE objective makes
their information criteria directly comparable. For each paper setting with a
known truth we record which model BIC (and AIC) selects.

    Setting 1  truth = Cox-type      (alpha = t^2+1, q = 1)
    Setting 2  truth = AM/AFT-type   (alpha = 1, q = 2/(1+u))
    Setting 4  truth = general LT    (alpha = t+1, q = 2/(1+u))

Usage::

    python -m ode_unify.paper_setting.bic_selection --reps 100 --workers 10
    python -m ode_unify.paper_setting.bic_selection --grid 4:1000 4:2000 4:4000
"""
from __future__ import annotations
import argparse, contextlib, io, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from scipy.optimize import minimize, LinearConstraint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from RecurrentODE_py.common import augknt, spcol                       # noqa: E402
from RecurrentODE_py.ltm.cox_rec import cox_rec                        # noqa: E402
from RecurrentODE_py.ltm.objective_func_sieve import objective_func_sieve as OFS  # noqa: E402
from RecurrentODE_py.ltm.objective_func_beta import objective_func_beta as OFB    # noqa: E402
from ode_unify.paper_dgp import simulate_paper                         # noqa: E402

TRUTH = {1: 'Cox', 2: 'AM', 4: 'Flex'}


def _one(args):
    setting, seed, N = args
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            d = simulate_paper(N, seed, setting)
            x = np.ascontiguousarray(d['x']); t = d['time'].ravel()
            delta = d['delta'].ravel(); idv = d['id'].ravel().astype(int)
            m = int((delta == 0).sum()); p = x.shape[1]; k0 = kq = 3
            temp2, binit = cox_rec(x, t, delta)
            l0 = int(np.ceil(len(np.unique(t)) ** 0.2))
            lq = int(np.ceil(x.shape[0] ** 0.2))
            knots_0 = augknt(np.linspace(0, float(np.max(t)), l0 + 1), k0)
            knots_q = augknt(np.linspace(0, 2 * float(np.max(temp2)), lq + 1), kq)
            q_0 = len(knots_0) - k0; q_q = len(knots_q) - kq
            Aeq = spcol(knots_0, k0, np.array([2.0]))[0]

            def fit(ft, fa, iters=60):
                beta = (binit / binit[0])[1:].copy()
                theta = np.zeros(q_q); alpha = np.zeros(q_0)
                for _ in range(iters):
                    bp, tp, ap = beta.copy(), theta.copy(), alpha.copy()

                    def f(v):
                        th = v[:q_q] if ft else theta
                        al = (v[q_q:] if ft else v) if fa else alpha
                        val, g = OFS(np.concatenate([th, al]), x, t, delta, idv,
                                     beta, knots_0, knots_q, k0, kq)
                        gg = []
                        if ft: gg.append(g[:q_q])
                        if fa: gg.append(g[q_q:])
                        return val, np.concatenate(gg)

                    v0 = np.concatenate(([theta] if ft else []) + ([alpha] if fa else []))
                    cons = []
                    if fa:
                        row = np.zeros(v0.size); row[(q_q if ft else 0):] = Aeq
                        cons = [LinearConstraint(row.reshape(1, -1), 0.0, 0.0)]
                    r = minimize(f, v0, jac=True, method='SLSQP', constraints=cons,
                                 options={'maxiter': 120, 'ftol': 1e-8})
                    if ft: theta = r.x[:q_q]
                    if fa: alpha = r.x[q_q:] if ft else r.x
                    rb = minimize(lambda bb: OFB(bb, x, t, delta, idv, theta, alpha,
                                                 knots_0, knots_q, k0, kq),
                                  beta, jac=True, method='BFGS',
                                  options={'maxiter': 400})
                    beta = rb.x
                    if max(np.max(np.abs(beta - bp)), np.max(np.abs(theta - tp)),
                           np.max(np.abs(alpha - ap))) < 1e-4:
                        break
                v, _ = OFS(np.concatenate([theta, alpha]), x, t, delta, idv, beta,
                           knots_0, knots_q, k0, kq)
                return float(v)

            out = {'_m': m}
            for nm, ft, fa, npar in [('Cox', False, True, (p - 1) + q_0),
                                     ('AM', True, False, (p - 1) + q_q),
                                     ('Flex', True, True, (p - 1) + q_q + q_0)]:
                out[nm] = (fit(ft, fa), npar)
        return setting, N, seed, out
    except Exception:                                        # noqa: BLE001
        return setting, N, seed, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--grid', nargs='*', default=['1:1000', '2:1000',
                                                  '4:1000', '4:2000', '4:4000'],
                    help='setting:N pairs')
    ap.add_argument('--out', default=os.path.join(HERE, 'results',
                                                  'bic_selection.csv'))
    a = ap.parse_args(argv)
    grid = [tuple(int(v) for v in g.split(':')) for g in a.grid]
    tasks = [(s, seed, N) for s, N in grid for seed in range(1, a.reps + 1)]
    got = {g: [] for g in grid}
    t0 = time.time(); nerr = 0
    print(f'{len(tasks)} fits over {len(grid)} configs, {a.workers} workers',
          flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            s, N, seed, r = f.result()
            if r is not None: got[(s, N)].append(r)
            else: nerr += 1
            if i % 100 == 0:
                print(f'  {i}/{len(tasks)} ({time.time()-t0:.0f}s, err={nerr})',
                      flush=True)
    rows = []
    print(f'\n{"setting":>8s}{"n":>7s}{"truth":>7s}{"ok":>5s}'
          f'{"Cox":>8s}{"AM":>8s}{"Flex":>8s}{"correct":>9s}', flush=True)
    for (s, N) in grid:
        R = got[(s, N)]
        if not R:
            continue
        for crit, pen in [('BIC', lambda k, m: k * np.log(m)),
                          ('AIC', lambda k, m: 2 * k)]:
            picks = []
            for r in R:
                m = r['_m']
                sc = {k: 2 * m * r[k][0] + pen(r[k][1], m)
                      for k in ('Cox', 'AM', 'Flex')}
                picks.append(min(sc, key=sc.get))
            fr = {k: sum(p == k for p in picks) / len(picks)
                  for k in ('Cox', 'AM', 'Flex')}
            corr = fr[TRUTH[s]]
            rows.append(dict(setting=s, n=N, truth=TRUTH[s], crit=crit,
                             n_ok=len(R), **fr, correct=corr))
            if crit == 'BIC':
                print(f'{s:>8d}{N:>7d}{TRUTH[s]:>7s}{len(R):>5d}'
                      f'{fr["Cox"]:>8.1%}{fr["AM"]:>8.1%}{fr["Flex"]:>8.1%}'
                      f'{corr:>9.1%}', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f'\nwrote {a.out}   ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
