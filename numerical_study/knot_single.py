"""How many knots does one functional parameter need, and can BIC find it?

Fitting ODE-Flex confounds the question: alpha and q compete for the same likelihood,
and the likelihood is dominated by the event times, which mainly inform alpha. BIC then
under-resolves q for reasons that have nothing to do with how complex q is.

Here exactly one functional parameter is free, so BIC and the estimation error refer to
the same object:

  ODE-Cox  (q == 1)      : only log alpha is estimated   -- settings 9 (complex), 10 (simple)
  ODE-AM   (alpha == 1)  : only log q is estimated       -- settings 11 (complex), 12 (simple)

Each model is paired with a simple and a complex truth, so the two questions separate:
(1) does the number of knots required track the complexity of the truth, and
(2) does BIC, minimised over the sieve size, land near the sieve that minimises error?

Knots are placed at quantiles, which spans the full support by construction.
"""
from __future__ import annotations

import argparse, contextlib, io, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

from ode_unify._engine.common import augknt, spcol, solve_sieve_step   # noqa: E402
from ode_unify._engine.ltm.cox_rec import cox_rec                      # noqa: E402
from ode_unify._engine.ltm.objective_func_sieve import objective_func_sieve as OFS  # noqa: E402,E501
from ode_unify._engine.ltm.objective_func_beta import objective_func_beta as OFB    # noqa: E402,E501
from ode_unify.paper_dgp import CUSTOM_FUNCS, simulate_paper         # noqa: E402
import lr_bootstrap_equal as LR                                       # noqa: E402

ANCHOR_T, NG, ITERS = 2.0, 80, 60
MODEL = {9: 'cox', 10: 'cox', 11: 'am', 12: 'am', 13: 'am', 14: 'am', 15: 'am'}


def _fit_one(x, t, delta, idv, binit, kn0, knq, model):
    """model='cox' frees only alpha; model='am' frees only q."""
    q_0, q_q = len(kn0) - LR.K0, len(knq) - LR.KQ
    beta = (binit / binit[0])[1:].copy()
    theta, alpha = np.zeros(q_q), np.zeros(q_0)
    Aeq = spcol(kn0, LR.K0, np.array([ANCHOR_T]))[0]
    use_trust = False
    for _ in range(ITERS):
        bp, tp, ap = beta.copy(), theta.copy(), alpha.copy()

        def f(v):
            th = theta if model == 'cox' else v
            al = v if model == 'cox' else alpha
            val, g = OFS(np.concatenate([th, al]), x, t, delta, idv, beta,
                         kn0, knq, LR.K0, LR.KQ)
            return val, (g[q_q:] if model == 'cox' else g[:q_q])

        # No alpha(2) = 1 constraint here. That constraint exists only to pin the
        # scale freedom alpha -> k*alpha, q -> q/k of the general transformation
        # model. Under ODE-Cox q is fixed at 1 and under ODE-AM alpha is fixed at 1,
        # so in both cases the remaining functional parameter is already identified
        # once beta_1 = 1; imposing it as well removes a genuine degree of freedom
        # and distorts the fit.
        v0, cons = (alpha if model == 'cox' else theta), []
        r, use_trust = solve_sieve_step(f, v0, cons, prefer_trust=use_trust,
                                        slsqp_options={'maxiter': 120, 'ftol': 1e-8})
        if model == 'cox':
            alpha = r.x
        else:
            theta = r.x
        rb = minimize(lambda bb: OFB(bb, x, t, delta, idv, theta, alpha,
                                     kn0, knq, LR.K0, LR.KQ),
                      beta, jac=True, method='BFGS', options={'maxiter': 400})
        beta = rb.x
        if max(np.max(np.abs(beta - bp)), np.max(np.abs(theta - tp)),
               np.max(np.abs(alpha - ap))) < 1e-4:
            break
    val, _ = OFS(np.concatenate([theta, alpha]), x, t, delta, idv, beta,
                 kn0, knq, LR.K0, LR.KQ)
    return float(val), beta, theta, alpha


def _one(args):
    setting, seed, N, p_exp = args
    model = MODEL[setting]
    cf = CUSTOM_FUNCS[setting]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            d = simulate_paper(N, seed, setting)
            x = np.ascontiguousarray(d['x']); t = d['time'].ravel()
            delta = d['delta'].ravel(); idv = d['id'].ravel().astype(int)
            temp2, binit = cox_rec(x, t, delta)
            ut = np.unique(t); t2 = np.concatenate([[0.0], np.asarray(temp2).ravel()])
            k = max(int(np.ceil((len(ut) if model == 'cox'
                                 else x.shape[0]) ** p_exp)), 2)
            if model == 'cox':
                kn0 = augknt(np.quantile(np.concatenate([[0.0], ut]),
                                         np.linspace(0, 1, k + 1)), LR.K0)
                knq = augknt(np.linspace(0, 2 * float(t2.max()), 3), LR.KQ)
            else:
                kn0 = augknt(np.linspace(0, float(ut.max()), 3), LR.K0)
                knq = augknt(np.quantile(t2, np.linspace(0, 1, k + 1)), LR.KQ)
            nll, beta, theta, alpha = _fit_one(x, t, delta, idv, binit,
                                               kn0, knq, model)
        n = int((delta == 0).sum())
        nfree = (len(kn0) - LR.K0) if model == 'cox' else (len(knq) - LR.KQ)
        npar = (x.shape[1] - 1) + nfree
        bic = 2 * n * nll + npar * np.log(n)
        if model == 'cox':
            g = np.linspace(0.0, float(ut.max()), NG)
            fit = spcol(kn0, LR.K0, g) @ alpha
            true = np.log(cf['alpha'](g))
        else:
            g = np.linspace(0.0, float(np.quantile(t2, 0.95)), NG)
            # NO anchoring. Under ODE-AM alpha == 1, so A(t) = t and the scale map
            # (Q, A, beta) -> (k Q^c, k A^c, c beta) forces k = c = 1: the LEVEL of
            # log q is identified, exactly as the level of log alpha is under ODE-Cox.
            # Anchoring would discard identified information and, at u = 0 (a boundary
            # knot, and the singular point of q = 1/(1+sqrt u)), it inflated the
            # reported RMSE ~5x and manufactured a spurious "complex q needs more
            # knots" conclusion. Anchoring is legitimate only for ODE-Flex, where both
            # functional parameters are free and the scale freedom is real.
            fit = spcol(knq, LR.KQ, g) @ theta
            true = np.log(cf['q'](g))
        return dict(setting=setting, seed=seed, p=p_exp, knots=k, npar=npar,
                    BIC=bic, err=float(np.abs(fit - true).max()),
                    rmse=float(np.sqrt(np.mean((fit - true) ** 2))),
                    b2=float(beta[0]), ok=True,
                    fit=fit.tolist(), g=g.tolist(), true=true.tolist())
    except Exception:
        return dict(setting=setting, seed=seed, p=p_exp, ok=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--settings', nargs='*', type=int, default=[9, 10, 11, 12])
    ap.add_argument('--p', nargs='*', type=float,
                    default=[0.05, 0.10, 0.14, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument('--tag', default='single')
    a = ap.parse_args(argv)
    RES = os.path.join(HERE, 'bic_model_selection', 'results')
    os.makedirs(RES, exist_ok=True)
    tasks = [(s, sd, a.n, p) for s in a.settings for p in a.p
             for sd in range(1, a.reps + 1)]
    print(f'{len(tasks)} fits: {len(a.settings)} settings x {len(a.p)} sieve sizes '
          f'x {a.reps} reps, n={a.n}', flush=True)
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 50 == 0:
                print(f'  {i}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    ok = [r for r in res if r.get('ok')]
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ('fit', 'g', 'true')} for r in ok])
    df.to_csv(os.path.join(RES, f'knot_{a.tag}.csv'), index=False)
    for s in a.settings:
        sub = df[df.setting == s]
        if not len(sub):
            continue
        g = sub.groupby('p').agg(knots=('knots', 'first'), npar=('npar', 'first'),
                                 ok=('BIC', 'size'), BIC=('BIC', 'median'),
                                 err=('err', 'median'),
                                 rmse=('rmse', 'median')).reset_index()
        g['dBIC'] = (g.BIC - g.BIC.min()).round(1)
        print(f"\n--- setting {s}: {CUSTOM_FUNCS[s]['desc']}  ({MODEL[s].upper()}) ---")
        print(g[['p', 'knots', 'npar', 'ok', 'BIC', 'dBIC', 'err', 'rmse']]
              .round(4).to_string(index=False))
        print(f"  BIC-optimal p = {g.p[g.BIC.idxmin()]:.2f} "
              f"({int(g.knots[g.BIC.idxmin()])} knots)   |   "
              f"RMSE-optimal p = {g.p[g.rmse.idxmin()]:.2f} "
              f"({int(g.knots[g.rmse.idxmin()])} knots)   |   "
              f"sup-optimal p = {g.p[g.err.idxmin()]:.2f} "
              f"({int(g.knots[g.err.idxmin()])} knots)")
    np.savez_compressed(os.path.join(RES, f'knot_{a.tag}_curves.npz'),
                        **{f's{s}_p{p}': np.median(
                            [r['fit'] for r in ok if r['setting'] == s
                             and r['p'] == p] or [np.zeros(NG)], 0)
                           for s in a.settings for p in a.p},
                        **{f's{s}_true': np.array(
                            [r['true'] for r in ok if r['setting'] == s][0])
                           for s in a.settings if any(r['setting'] == s for r in ok)},
                        **{f's{s}_grid': np.array(
                            [r['g'] for r in ok if r['setting'] == s][0])
                           for s in a.settings if any(r['setting'] == s for r in ok)},
                        ps=np.array(a.p))
    print(f'\nwrote knot_{a.tag}.csv and knot_{a.tag}_curves.npz '
          f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
