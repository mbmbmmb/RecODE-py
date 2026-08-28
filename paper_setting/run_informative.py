"""Parallel informative-censoring study.

Each subject's censoring time is

    C_i = C0_i * exp(x_i' gamma_c) * xi_i ** c_xi

with C0_i drawn from the setting's own paper censoring window.

  * gamma_c = 0, c_xi = 0     -> random censoring (baseline)
  * gamma_c != 0, c_xi = 0    -> depends on the OBSERVED covariates; conditional
                                 independence given x still holds
  * c_xi != 0                 -> depends on the UNOBSERVED frailty; genuinely
                                 violates the assumption

Usage::

    python -m ode_unify.paper_setting.run_informative --reps 1000 --workers 10
    python -m ode_unify.paper_setting.run_informative --only cov_decr --reps 100
"""
from __future__ import annotations
import argparse, contextlib, io, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ode_unify as U                                     # noqa: E402
from ode_unify.paper_dgp import simulate_paper            # noqa: E402

OUT = os.path.join(HERE, 'results', 'informative')

# name -> (gamma_c, c_xi, random_effect, description)
REGIMES = {
    'random':      (None,               0.0, False, 'C ~ U(a,b)'),
    'cov':         ([-0.5, -0.5, -0.5], 0.0, False, "C = C0*exp(x'gamma_c), gamma_c=-0.5"),
    # censoring strictly monotone DECREASING in the linear predictor f(x)=x'beta:
    # gamma_c = -beta, so C = C0*exp(-f(x)). Subjects with a higher event rate are
    # censored earlier -- the strongest observed-covariate dependence in this family.
    'cov_decr':    ([-1.0, -1.0, -1.0], 0.0, False, "C = C0*exp(-x'beta) (monotone decreasing)"),
    'random_fr':   (None,               0.0, True,  'C ~ U(a,b), gamma frailty'),
    'frailty_inf': ([-0.5, -0.5, -0.5], 1.0, True,  "C = C0*exp(x'gamma_c)*xi (VIOLATES)"),
}


def _one(args):
    name, seed, N, setting, beta = args
    gc, cxi, re, _ = REGIMES[name]
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            d = simulate_paper(N, seed, setting, beta=beta, random_effect=re,
                               censor_coef=gc, censor_frailty_coef=cxi)
            est = U.fit(d, estimator='cox', random_effect=re, ci=True,
                        seed=seed, data_setting=setting)
        p = len(beta)
        se = est.se.ravel() if est.se is not None else np.full(p, np.nan)
        return name, seed, est.beta.ravel(), se, int((d['delta'].ravel() == 1).sum())
    except Exception:                                       # noqa: BLE001
        return name, seed, None, None, 0


def run(name, reps, N, setting, beta, workers, out_dir):
    tasks = [(name, s, N, setting, beta) for s in range(1, reps + 1)]
    p = len(beta)
    B = np.full((reps, p), np.nan); S = np.full((reps, p), np.nan)
    ev = []; ok = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_one, t) for t in tasks]):
            nm, seed, b, s, ne = f.result()
            if b is not None:
                B[seed - 1] = b; S[seed - 1] = s; ev.append(ne); ok += 1
    truth = np.asarray(beta, float)
    good = np.isfinite(B[:, 0])
    bias = np.nanmean(B, 0) - truth
    esd = np.nanstd(B, 0, ddof=1)
    mse = np.nanmean(S, 0)
    cov = np.nanmean(((B - 1.96 * S) <= truth) & (truth <= (B + 1.96 * S)), 0)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f'{name}.npz'),
                        beta=B[good], se=S[good], truth=truth,
                        events=np.array(ev), ok=ok, reps=reps, N=N,
                        setting=setting, gamma_c=np.array(REGIMES[name][0] or [0.0] * p),
                        c_xi=REGIMES[name][1])
    print(f'[{name}] ok={ok}/{reps} in {time.time()-t0:.0f}s  '
          f'events/rep={np.mean(ev):.0f}', flush=True)
    print(f'    bias    = {np.round(bias, 4)}', flush=True)
    print(f'    emp.SD  = {np.round(esd, 4)}', flush=True)
    print(f'    mean.SE = {np.round(mse, 4)}', flush=True)
    print(f'    CP95    = {np.round(cov, 4)}   mean={np.mean(cov):.3f}', flush=True)
    return dict(name=name, ok=ok, bias=bias, cov=cov)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--N', type=int, default=1000)
    ap.add_argument('--setting', type=int, default=1)
    ap.add_argument('--beta', type=float, nargs='+', default=[1.0, 1.0, 1.0])
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args(argv)
    sel = a.only or list(REGIMES)
    bad = [s for s in sel if s not in REGIMES]
    if bad:
        raise SystemExit(f'unknown regime(s): {bad}')
    print(f'informative censoring: setting {a.setting}, N={a.N}, beta={a.beta}, '
          f'{a.reps} reps, {a.workers} workers', flush=True)
    res = []
    for name in sel:
        print(f'\n--- {name}: {REGIMES[name][3]} ---', flush=True)
        res.append(run(name, a.reps, a.N, a.setting, a.beta, a.workers, a.out))
    print('\n=== summary ===', flush=True)
    for r in res:
        print(f'  {r["name"]:12s} max|bias|={np.max(np.abs(r["bias"])):.4f}  '
              f'mean CP95={np.mean(r["cov"]):.3f}', flush=True)


if __name__ == '__main__':
    main()
