"""Exact-parity checks: ode_unify.estimate/inference vs the separate modules.

For each (estimator, random_effect) combination this fits the *same* dataset
through the unified pipeline and through the original per-model
``RecurrentODE_py`` ``main()`` / ``inference()`` (into a scratch dir), then
reports the max absolute difference in the point estimates (``est_r`` = beta +
spline coefficients) and the standard errors. All should be 0.0 to machine
precision (the resampling SEs match because the reference routines use fixed
RNG seeds: 0 for the frailty cox, the data seed for frailty aft/ltm).

Requires ``RecurrentODE_py`` (the reference) to be importable. Run::

    python -m ode_unify.sanity_check                 # all six combos
    python -m ode_unify.sanity_check --only cox re_cox
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ode_unify as U  # noqa: E402

TOL = 1e-8

CASES = {
    'cox': dict(estimator='cox', re=False, ds=1, knots=None, N=1000),
    'aft': dict(estimator='aft', re=False, ds=2, knots='quantile', N=1000),
    'npmle': dict(estimator='npmle', re=False, ds=3, knots='equal', N=2000),
    'ltm': dict(estimator='ltm', re=False, ds=4, knots='K4', N=1000),
    're_cox': dict(estimator='cox', re=True, ds=1, knots=None, N=1000),
    're_aft': dict(estimator='aft', re=True, ds=2, knots='quantile', N=1000),
    're_ltm': dict(estimator='ltm', re=True, ds=1, knots='K4', N=1000),
}


def _reference(name, seed):
    c = CASES[name]
    N, ds, knots = c['N'], c['ds'], c['knots']
    if name == 'cox':
        import RecurrentODE_py.cox.main as m
        with tempfile.TemporaryDirectory() as tmp:
            m.main(N, seed, ds, True, root=tmp)
            d = np.load(os.path.join(
                tmp, 'res', f'res_cox_N{N}_seed{seed}_setting{ds}_se.npz'))
            return {'est_r': d['est_r'].ravel(), 'se_all': d['se_all'].ravel()}
    if name == 'aft':
        import RecurrentODE_py.aft.main as m
        with tempfile.TemporaryDirectory() as tmp:
            m.main(N, seed, ds, knots, True, root=tmp)
            d = np.load(os.path.join(
                tmp, 'res',
                f'res_aft_N{N}_seed{seed}_setting{ds}_knots{knots}_se.npz'))
            return {'est_r': d['est_r'].ravel(), 'se_all': d['se_all'].ravel()}
    if name == 'npmle':
        import RecurrentODE_py.npmle.main as m
        with tempfile.TemporaryDirectory() as tmp:
            m.main(N, seed, ds, knots, True, root=tmp)
            d = np.load(os.path.join(
                tmp, 'res',
                f'res_Gtransform_N{N}_seed{seed}_setting{ds}_knots{knots}_se.npz'))
            return {'est_r': d['est_r'].ravel(), 'se_all': d['se_all'].ravel()}
    if name == 'ltm':
        import RecurrentODE_py.ltm.main as m
        with tempfile.TemporaryDirectory() as tmp:
            m.main(N, seed, ds, knots, True, root=tmp)
            d = np.load(os.path.join(
                tmp, 'res',
                f'res_ltm_N{N}_seed{seed}_setting{ds}_knots{knots}_se.npz'))
            return {'est_r': d['est_r'].ravel(), 'se_all': d['se_all'].ravel()}
    if name == 're_cox':
        import RecurrentODE_py.random_effect.cox.main as m
        from RecurrentODE_py.random_effect.cox.inference import inference as inf
        with tempfile.TemporaryDirectory() as tmp:
            m.main(N, seed, ds, ci=True, root=tmp)
            inf(N, seed, ds, root=tmp)
            se = np.load(os.path.join(
                tmp, 'res', f'res_cox_N{N}_seed{seed}_setting{ds}_se.npz'))
            i = np.load(os.path.join(
                tmp, 'res',
                f'res_cox_N{N}_seed{seed}_setting{ds}_inference.npz'))
            return {'est_r': se['est_r'].ravel(),
                    'se_beta': se['se_beta'].ravel(),
                    'se_all': i['se_all'].ravel()}
    if name == 're_aft':
        import RecurrentODE_py.random_effect.aft_rec.main as m
        with tempfile.TemporaryDirectory() as tmp:
            m.main(N, seed, ds, knots, ci=True, root=tmp)
            d = np.load(os.path.join(
                tmp, 'res',
                f'res_aft_N{N}_seed{seed}_setting{ds}_knots{knots}_se.npz'))
            return {'est_r': d['est_r'].ravel(), 'se_all': d['se_all'].ravel()}
    if name == 're_ltm':
        import RecurrentODE_py.random_effect.ltm.main as m
        with tempfile.TemporaryDirectory() as tmp:
            m.main(N, seed, ds, knots, ci=True, root=tmp)
            d = np.load(os.path.join(
                tmp, 'res', 'cox_rec_rd',
                f'res_cox_N{N}_seed{seed}_setting{ds}_knots{knots}_fish.npz'))
            return {'est_r': d['est_r'].ravel(), 'se_all': d['se_all'].ravel()}
    raise KeyError(name)


def check(name, seed=1):
    c = CASES[name]
    with contextlib.redirect_stdout(io.StringIO()):
        data = U.simulate(c['N'], seed, c['ds'], random_effect=c['re'])
        # layout='legacy' mirrors each old pipeline's memory layout so the
        # comparison is bit-for-bit; the default 'uniform' layout agrees with
        # it to optimizer tolerance (~1e-6), far below the standard errors.
        est = U.estimate(data, estimator=c['estimator'],
                         random_effect=c['re'], knots=c['knots'], seed=seed,
                         layout='legacy')
        est = U.inference(est, data, seed=seed, data_setting=c['ds'],
                          spline_se=True)
        ref = _reference(name, seed)

    diffs = {'est_r': float(np.max(np.abs(
        est.raw['est_r'].ravel() - ref['est_r'])))}
    if 'se_all' in ref:
        diffs['se_all'] = float(np.max(np.abs(
            est.se_all.ravel() - ref['se_all'])))
    if 'se_beta' in ref:
        diffs['se_beta'] = float(np.max(np.abs(
            est.raw['se_beta'].ravel() - ref['se_beta'])))
    ok = all(v <= TOL for v in diffs.values())
    return ok, diffs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', default=None,
                    help='subset of: ' + ', '.join(CASES))
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args(argv)
    names = args.only or list(CASES)

    print(f'ode_unify exact-parity check (seed={args.seed}, tol={TOL:g})')
    print('-' * 66)
    all_ok = True
    for name in names:
        ok, diffs = check(name, seed=args.seed)
        all_ok &= ok
        parts = '  '.join(f'{k}={v:.2e}' for k, v in diffs.items())
        print(f'  {"PASS" if ok else "FAIL":4s}  {name:8s}  {parts}')
    print('-' * 66)
    print('ALL EXACT' if all_ok else 'MISMATCH DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
