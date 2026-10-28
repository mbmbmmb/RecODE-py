"""Model selection across the ODE family by BIC (Section 5.5 and Table 7 of the paper).

ODE-Cox (q == 1), ODE-AM (alpha == 1), ODE-LT (q specified) and ODE-Flex (both free)
are nested restrictions of one sieve parameterisation, so fitting all of them with the
same objective makes their information criteria directly comparable. For each setting
with a known truth we record which model BIC (and AIC) selects.

    Setting 1  truth = Cox-type      (alpha = t^2+1, q = 1)
    Setting 2  truth = AFT-type      (alpha = 1, q = 2/(1+u))
    Setting 3  truth = Box-Cox LT    (alpha = 0.2/(1+t), q = 1/(1+u/2))
    Setting 7  truth = general model (alpha = t+1, q = 1+0.7 sin(1.5u))

Sieve: ceil(m^(1/7)) intervals for both log alpha and log q, knots at sample quantiles.
The same study with the equally spaced knots of bic_selection.py is reported in
Section D.3 of the Supplementary Material.

Usage (from this directory):

    python bic_selection_quantile.py --reps 100 --grid 1:2000 2:2000 3:2000 7:2000 \
        --out bic_model_selection/results/bic_selection_q17_n2000.csv
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

from ode_unify._engine.common import augknt, spcol, solve_sieve_step                       # noqa: E402
from ode_unify._engine.ltm.cox_rec import cox_rec                        # noqa: E402
from ode_unify._engine.ltm.objective_func_sieve import objective_func_sieve as OFS  # noqa: E402
from ode_unify._engine.ltm.objective_func_beta import objective_func_beta as OFB    # noqa: E402
from ode_unify.paper_dgp import simulate_paper                         # noqa: E402

TRUTH = {1: 'Cox', 2: 'AM', 3: 'LT', 4: 'Flex', 7: 'Flex'}
RHO = 0.5          # Box-Cox index of the LT candidate (the paper's Setting 3)


def _one(args):
    setting, seed, N, l0_extra, lq_extra = args
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            d = simulate_paper(N, seed, setting)
            x = np.ascontiguousarray(d['x']); t = d['time'].ravel()
            delta = d['delta'].ravel(); idv = d['id'].ravel().astype(int)
            m = int((delta == 0).sum()); p = x.shape[1]; k0 = kq = 3
            temp2, binit = cox_rec(x, t, delta)
            # l0_extra widens the alpha sieve. With the default sieve the
            # q-spline can absorb whatever alpha(t) fails to capture, which
            # shows up as a spurious Flex-over-Cox likelihood gain.
            # CORRECTED SIEVE (see knot_sensitivity_results.md). The original used
            #   l0 = ceil(#unique t ^ 0.2) + extra,  lq = ceil(#ROWS ^ 0.2) + extra
            #   both EQUALLY SPACED, q knots over [0, 2*max(mu)]
            # so alpha and q were sized from different quantities, about half the q
            # basis functions had no data behind them, and n^(1/5) sits past the flat
            # region of the error curve. Here both get the SAME count, ceil(m^(1/7))
            # with m = number of SUBJECTS, placed at QUANTILES over the realised support.
            # This matters more for BIC than for the LR test: the penalty is
            # (#coefficients)*log n, so the sieve dimension enters the decision directly.
            m_subj = int(np.sum(delta == 0))
            nk = int(np.ceil(m_subj ** (1.0 / 7.0)))
            l0 = nk + l0_extra
            lq = nk + lq_extra
            u0 = np.linspace(0.0, 1.0, l0 + 1)
            uq = np.linspace(0.0, 1.0, lq + 1)
            knots_0 = augknt(np.quantile(np.concatenate([[0.0], t[delta == 1]]), u0), k0)
            knots_q = augknt(np.quantile(np.concatenate([[0.0], temp2.ravel()]), uq), kq)
            q_0 = len(knots_0) - k0; q_q = len(knots_q) - kq
            Aeq = spcol(knots_0, k0, np.array([2.0]))[0]
            # Box-Cox candidate: log q(u) = -log(1 + rho*u), projected on the
            # q-spline basis and then held fixed (alpha stays free).
            # project on the region where the q argument actually falls. The
            # basis spans [0, 2*max(temp2)], about twice the realised mu range,
            # so a uniform grid over the whole domain would fit the empty tail
            # as hard as the region carrying the data.
            ug = np.quantile(temp2, np.linspace(0.001, 0.999, 400))
            theta_lt = np.linalg.lstsq(spcol(knots_q, kq, ug),
                                       -np.log1p(RHO * ug), rcond=None)[0]

            def fit(ft, fa, theta0=None, iters=60, scale_free=False,
                    beta0=None):
                # scale_free: theta = theta0 + c*1 with c estimated. B-splines
                # form a partition of unity, so adding c to every coefficient
                # shifts log q by c. The LTM identifies (alpha, q) only up to
                # alpha -> k*alpha, q -> q/k, and alpha is anchored at
                # alpha(2)=1, so a specified q needs this one free constant --
                # without it the candidate is fitted on the wrong scale.
                beta = ((binit / binit[0])[1:].copy() if beta0 is None
                        else np.asarray(beta0, dtype=float).copy())
                theta = np.zeros(q_q) if theta0 is None else theta0.copy()
                alpha = np.zeros(q_0)
                cc = 0.0
                use_trust = False      # sticky once the fallback is needed
                for _ in range(iters):
                    bp, tp, ap = beta.copy(), theta.copy(), alpha.copy()

                    def f(v):
                        if scale_free:
                            th = theta0 + v[0]
                            al = v[1:]
                        else:
                            th = v[:q_q] if ft else theta
                            al = (v[q_q:] if ft else v) if fa else alpha
                        val, g = OFS(np.concatenate([th, al]), x, t, delta, idv,
                                     beta, knots_0, knots_q, k0, kq)
                        if scale_free:
                            return val, np.concatenate([[g[:q_q].sum()], g[q_q:]])
                        gg = []
                        if ft: gg.append(g[:q_q])
                        if fa: gg.append(g[q_q:])
                        return val, np.concatenate(gg)

                    if scale_free:
                        v0 = np.concatenate([[cc], alpha])
                    else:
                        v0 = np.concatenate(([theta] if ft else []) + ([alpha] if fa else []))
                    cons = []
                    if scale_free:
                        row = np.zeros(v0.size); row[1:] = Aeq
                        cons = [LinearConstraint(row.reshape(1, -1), 0.0, 0.0)]
                    elif ft and fa:
                        # ONLY ODE-Flex. The alpha(anchor)=1 constraint pins the
                        # scale freedom alpha -> k*alpha, q -> q/k, which exists
                        # only when BOTH functional parameters are free. Under
                        # ODE-Cox (q == 1) alpha is already identified, so the
                        # constraint would delete a real degree of freedom and
                        # bias alpha-hat upward. (ODE-LT takes the scale_free
                        # branch above, where a free c makes it legitimate.)
                        row = np.zeros(v0.size); row[q_q:] = Aeq
                        cons = [LinearConstraint(row.reshape(1, -1), 0.0, 0.0)]
                    # ftol=1e-8 makes SLSQP's line search probe far more
                    # aggressively than the library default, which is where the
                    # BIC-study failures were concentrated; fall back to
                    # trust-constr for the rest of this fit when it breaks down.
                    r, use_trust = solve_sieve_step(
                        f, v0, cons, prefer_trust=use_trust,
                        slsqp_options={'maxiter': 120, 'ftol': 1e-8})
                    if scale_free:
                        cc = r.x[0]; theta = theta0 + cc; alpha = r.x[1:]
                    else:
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
                return float(v), beta

            # Fit ODE-Cox first, retrying from alternative starts if it fails,
            # then warm-start every other candidate from its beta. Cox is the
            # cheapest and best-conditioned member of the family (theta = 0
            # removes the q-spline entirely), so it converges where the others
            # struggle, and its beta is a far better starting point than the
            # cox_rec initialiser for the models that do struggle.
            out = {'_m': m}
            beta_cox = None
            _starts = ((None,) if os.environ.get('WARMSTART_DISABLE')
                       else (None, np.zeros(p - 1), (binit / binit[0])[1:] * 0.5))
            for b0 in _starts:
                try:
                    v, beta_cox = fit(False, True, None, beta0=b0)
                    out['Cox'] = (v, (p - 1) + q_0)
                    break
                except Exception:                            # noqa: BLE001
                    beta_cox = None
            if beta_cox is None:
                return setting, N, seed, None                # Cox unrecoverable

            for nm, ft, fa, th0, npar, sf in [
                    ('AM',   True,  False, None,      (p - 1) + q_q,        False),
                    ('LT',   False, True,  theta_lt,  (p - 1) + q_0 + 1,    True),
                    ('Flex', True,  True,  None,      (p - 1) + q_q + q_0,  False)]:
                _b0 = None if os.environ.get('WARMSTART_DISABLE') else beta_cox
                v, _ = fit(ft, fa, th0, scale_free=sf, beta0=_b0)
                out[nm] = (v, npar)
        return setting, N, seed, out
    except Exception:                                        # noqa: BLE001
        if os.environ.get('BIC_RAISE'):                      # diagnostics
            raise
        return setting, N, seed, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--grid', nargs='*', default=['1:1000', '2:1000', '3:1000',
                                                  '4:1000'],
                    help='setting:N pairs')
    ap.add_argument('--q-knots-extra', type=int, default=0,
                    help='extra interior knots for log q(u)')
    ap.add_argument('--alpha-knots-extra', type=int, default=0,
                    help='extra interior knots for log alpha(t) beyond the '
                         'default ceil(#unique times ^ 1/5)')
    ap.add_argument('--out', default=os.path.join(HERE, 'bic_model_selection',
                                                  'results',
                                                  'bic_selection.csv'))
    a = ap.parse_args(argv)
    grid = [tuple(int(v) for v in g.split(':')) for g in a.grid]
    tasks = [(s, seed, N, a.alpha_knots_extra, a.q_knots_extra)
             for s, N in grid for seed in range(1, a.reps + 1)]
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
          f'{"Cox":>8s}{"AM":>8s}{"LT":>8s}{"Flex":>8s}{"correct":>9s}', flush=True)
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
                      for k in ('Cox', 'AM', 'LT', 'Flex')}
                picks.append(min(sc, key=sc.get))
            fr = {k: sum(p == k for p in picks) / len(picks)
                  for k in ('Cox', 'AM', 'LT', 'Flex')}
            corr = fr[TRUTH[s]]
            rows.append(dict(setting=s, n=N, truth=TRUTH[s], crit=crit,
                             n_ok=len(R), **fr, correct=corr))
            if crit == 'BIC':
                print(f'{s:>8d}{N:>7d}{TRUTH[s]:>7s}{len(R):>5d}'
                      f'{fr["Cox"]:>8.1%}{fr["AM"]:>8.1%}{fr["LT"]:>8.1%}'
                      f'{fr["Flex"]:>8.1%}{corr:>9.1%}', flush=True)
    # raw fits, so the likelihood gain can be compared with the penalty
    raw = []
    for (s_, N) in grid:
        for r in got[(s_, N)]:
            raw.append(dict(setting=s_, n=N, m=r['_m'],
                            **{f'{k}_negll': r[k][0] for k in ('Cox','AM','LT','Flex')},
                            **{f'{k}_npar': r[k][1] for k in ('Cox','AM','LT','Flex')}))
    if raw:
        pd.DataFrame(raw).to_csv(a.out.replace('.csv', '_raw.csv'), index=False)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f'\nwrote {a.out}   ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
