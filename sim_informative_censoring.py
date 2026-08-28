"""Simulation study: covariate-dependent (informative) censoring.

Addresses the reviewers' request (Review 2, comment 2; Review 1 & 2 on
informative censoring) to go beyond covariate-independent U(a,b) censoring.

Censoring regimes on the same Cox event process (setting 1, lambda0(t)=t^2+1,
beta=(b,b,b)); each subject's censoring time is

    C_i = C0_i * exp(x_i' gamma_c) * xi_i^{c_xi},   C0_i ~ U(a, b)

  * random   : gamma_c = 0, c_xi = 0   -> C ~ U(a,b)          (paper baseline)
  * cov       : gamma_c != 0, c_xi = 0  -> C depends on covariates X
  * frailty   : c_xi != 0               -> C depends on the *unobserved* frailty

For `random` and `cov`, censoring depends only on the OBSERVED covariates, so
conditional independence given x still holds and the sieve estimator should stay
consistent with ~95% CI coverage. `frailty` makes censoring depend on the
unobserved xi -- genuinely informative censoring that breaks the assumption --
included as a stress test. The `cov`/`random` pair is fit WITHOUT frailty; the
`frailty`/`random_fr` pair is fit WITH frailty, so each comparison uses one
estimator.

Reports per regime & coefficient: mean estimate, bias, empirical SD, mean
estimated SE, and 95% Wald CI coverage of the true beta.

Usage:
    python -m ode_unify.sim_informative_censoring --reps 300 --N 500
    python -m ode_unify.sim_informative_censoring --reps 150 --N 500 --with_frailty
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os

import numpy as np

import ode_unify as U
from ode_unify.paper_dgp import simulate_paper


def one_rep(seed, *, N, beta, setting, censor, cc, cf, re, paper=False):
    with contextlib.redirect_stdout(io.StringIO()):
        if paper:
            # paper-faithful covariates + the setting's own censoring window
            data = simulate_paper(N, seed, setting, beta=beta,
                                  random_effect=re, censor_coef=cc,
                                  censor_frailty_coef=cf)
        else:
            data = U.simulate(N, seed, setting, random_effect=re, beta=beta,
                              censor=censor, censor_coef=cc,
                              censor_frailty_coef=cf)
        est = U.fit(data, estimator='cox', random_effect=re, ci=True,
                    seed=seed, data_setting=setting)
    p = len(beta)
    se = est.se.ravel() if est.se is not None else np.full(p, np.nan)
    n_evt = int((data['delta'].ravel() == 1).sum())
    return est.beta.ravel(), se, n_evt


def run_regime(name, *, reps, N, beta, setting, censor, cc, cf, re, seed0=1,
               paper=False, save_dir=None):
    p = len(beta)
    B = np.full((reps, p), np.nan)
    S = np.full((reps, p), np.nan)
    evts, ok = [], 0
    for r in range(reps):
        try:
            b, s, ne = one_rep(seed0 + r, N=N, beta=beta, setting=setting,
                               censor=censor, cc=cc, cf=cf, re=re, paper=paper)
            B[r], S[r] = b, s
            evts.append(ne)
            ok += 1
        except Exception:                          # noqa: BLE001
            continue
    truth = np.asarray(beta, float)
    mean = np.nanmean(B, axis=0)
    esd = np.nanstd(B, axis=0, ddof=1)
    mse = np.nanmean(S, axis=0)
    cov = np.nanmean(((B - 1.96 * S) <= truth) & (truth <= (B + 1.96 * S)),
                     axis=0)
    out = dict(regime=name, ok=ok, reps=reps,
               mean_events=float(np.mean(evts)) if evts else np.nan,
               mean=mean, bias=mean - truth, esd=esd, mse=mse, cov=cov)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, f'{name}.npz'),
                            beta=B, se=S, truth=truth, events=np.array(evts),
                            ok=ok, reps=reps, N=N, setting=setting,
                            paper=bool(paper))
    return out


def _row(name, vals):
    return f'  {name:9s} ' + ' '.join(f'{v:8.4f}' for v in vals)


def report(res, beta):
    print(f'\n[{res["regime"]}]  ok={res["ok"]}/{res["reps"]}  '
          f'mean events/rep={res["mean_events"]:.0f}')
    print(_row('true', beta))
    print(_row('mean', res['mean']))
    print(_row('bias', res['bias']))
    print(_row('emp.SD', res['esd']))
    print(_row('mean.SE', res['mse']))
    print(_row('cover95', res['cov']))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--N', type=int, default=500)
    ap.add_argument('--setting', type=int, default=1)
    ap.add_argument('--beta', type=float, nargs='+', default=[0.3, 0.3, 0.3])
    ap.add_argument('--censor', type=float, nargs=2, default=[1.0, 2.0])
    ap.add_argument('--gamma_c', type=float, nargs='+', default=[-0.5, -0.5, -0.5],
                    help='covariate-dependent censoring coefficients')
    ap.add_argument('--c_xi', type=float, default=1.0,
                    help='frailty-informative censoring strength')
    ap.add_argument('--with_frailty', action='store_true')
    ap.add_argument('--frailty_only', action='store_true',
                    help='run only the frailty (informative) pair')
    ap.add_argument('--paper', action='store_true',
                    help="use ode_unify.paper_dgp.simulate_paper (the paper's "
                         'covariate design and per-setting censoring window) '
                         'instead of the general dgp defaults')
    ap.add_argument('--save_dir', default=None,
                    help='write per-regime npz (beta/se/events) here')
    args = ap.parse_args(argv)

    beta, censor, gc = args.beta, tuple(args.censor), args.gamma_c
    print(f'Informative-censoring study  (Cox setting {args.setting}, '
          f'beta={beta}, N={args.N}, reps={args.reps}, censor~U{censor})')
    print(f"gamma_c={gc}   C_i = C0_i * exp(x'gamma_c)"
          + (f" * xi^{args.c_xi}" if args.with_frailty else ""))
    print('=' * 70)

    results = []
    # Pair 1 (no frailty): random vs covariate-dependent censoring.
    if not args.frailty_only:
        for name, cc, cf, re in [('random', None, 0.0, False),
                                 ('cov', gc, 0.0, False)]:
            res = run_regime(name, reps=args.reps, N=args.N, beta=beta,
                             setting=args.setting, censor=censor,
                             cc=cc, cf=cf, re=re, paper=args.paper,
                             save_dir=args.save_dir)
            results.append(res); report(res, beta)

    # Pair 2 (frailty estimator): random vs truly-informative frailty censoring.
    if args.with_frailty or args.frailty_only:
        for name, cc, cf in [('random_fr', None, 0.0),
                             ('frailty_inf', gc, args.c_xi)]:
            res = run_regime(name, reps=args.reps, N=args.N, beta=beta,
                             setting=args.setting, censor=censor,
                             cc=cc, cf=cf, re=True, paper=args.paper,
                             save_dir=args.save_dir)
            results.append(res); report(res, beta)

    print('\n' + '=' * 70)
    print('SUMMARY  (max|bias|, mean 95% coverage over the 3 coefficients)')
    for res in results:
        print(f'  {res["regime"]:11s} max|bias|={np.max(np.abs(res["bias"])):.4f}'
              f'   mean cover95={np.mean(res["cov"]):.3f}'
              f'   events/rep={res["mean_events"]:.0f}')
    return results


if __name__ == '__main__':
    main()
