"""Tables S1-S2 of the Supplementary Material: sensitivity of the regression estimates in
Tables 2 and 3 of the paper to the number and placement of the knots.

Each study of Tables 2 and 3 (ODE-Cox, ODE-AM and ODE-LT in Settings 1-3, ODE-Flex in
Settings 1-4, n = 1000) is refitted with the number of knots set by the exponent p in
place of the default (1/7 for ODE-Cox and 1/5 for the others), and with the knots of both
log alpha and log q either equally spaced or placed at the sample quantiles (K1 and K4
for ODE-Flex); the spline order and the variance estimator are unchanged. The
replications with the default exponent and placement are those of run_paper.py and are
read from its results directory.

Table S1 holds the rows with the placement of the paper (paper_placement) and Table S2
the other placements.

Usage:  python knot_beta_sensitivity.py run [--reps 1000] [--workers 9]
        python knot_beta_sensitivity.py report        (writes tables/table_S1_S2_knots_beta.csv)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from ode_unify.estimator import estimate                 # noqa: E402
from ode_unify.inference import inference                # noqa: E402
from ode_unify.paper_dgp import simulate_paper           # noqa: E402
import run_paper as RP                                   # noqa: E402

OUT = os.path.join(HERE, 'simulation_study', 'knot_sensitivity', 'results')
SLUGS = ['t1_cox_s1', 't1_am_s2', 't1_lt_s3',
         't2_flex_s1', 't2_flex_s2', 't2_flex_s3', 't2_flex_s4']
P_GRID = [0.10, 1 / 7, 1 / 5, 0.25]
DEFAULT_P = {s: (1 / 7 if s == 't1_cox_s1' else 1 / 5) for s in SLUGS}
PLACEMENTS = {'equal': {'cox': 'equal', 'aft': 'equal', 'npmle': 'equal', 'ltm': 'K1'},
              'quantile': {'cox': 'quantile', 'aft': 'quantile', 'npmle': 'quantile',
                           'ltm': 'K4'}}


def _default_knots(slug):
    return RP.STUDIES[slug]['knots'] or 'equal'


def _root(slug, p, knots):
    """Results directory holding <slug>/seed*.npz for exponent p and knot scheme."""
    if knots == _default_knots(slug):
        if np.isclose(p, DEFAULT_P[slug]):
            return RP.DEFAULT_RESULTS
        return os.path.join(OUT, f'p{p:.2f}')
    return os.path.join(OUT, knots, f'p{p:.2f}')


def _configs():
    """(slug, p, knots) for every exponent and both placements, and for the placement
    of the paper when it is neither (K3 in Setting 3 of ODE-Flex)."""
    out = []
    for s in SLUGS:
        schemes = [PLACEMENTS[pl][RP.STUDIES[s]['estimator']] for pl in PLACEMENTS]
        if _default_knots(s) not in schemes:
            schemes.append(_default_knots(s))
        out += [(s, p, kn) for kn in schemes for p in P_GRID]
    return out


def _one(args):
    slug, p, knots, seed = args
    cfg = RP.STUDIES[slug]
    out_dir = os.path.join(_root(slug, p, knots), slug)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'seed{seed}.npz')
    if os.path.isfile(path):
        return 'skip'
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        data = simulate_paper(cfg['N'], seed, cfg['setting'])
        est = estimate(data, estimator=cfg['estimator'], random_effect=False,
                       knots=knots, seed=seed, layout='legacy', knot_exp=p)
        est = inference(est, data, spline_se=True, data_setting=cfg['setting'], seed=seed)
    payload = {k: np.asarray(v) for k, v in est.raw.items()}
    payload.update(beta=est.beta, se=est.se, _success=np.array(bool(est.success)),
                   _knot_exp=np.array(p),
                   _n_events=np.array(int((data['delta'].ravel() == 1).sum())))
    np.savez_compressed(path, **payload)
    return 'ok'


def run(reps, workers):
    tasks = [(s, p, kn, seed) for s, p, kn in _configs()
             for seed in range(1, reps + 1)
             if not os.path.isfile(os.path.join(_root(s, p, kn), s, f'seed{seed}.npz'))]
    tasks.sort(key=lambda t: -RP.SEED_COST.get(t[0], 1.0) * t[1])
    print(f'{len(tasks)} fits on {workers} workers', flush=True)
    t0, done, fail = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, t): t for t in tasks}
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as exc:                       # noqa: BLE001
                fail += 1
                print(f'  {futs[f]} failed: {type(exc).__name__}: {exc}', flush=True)
            if done % 500 == 0 or done == len(tasks):
                print(f'  {done}/{len(tasks)} ({fail} failed, {time.time() - t0:.0f}s)',
                      flush=True)


def n_knots(slug, root):
    """Median number of intervals of each spline over the replications."""
    d = os.path.join(root, slug)
    k = []
    for f in sorted(os.listdir(d))[:50]:
        z = np.load(os.path.join(d, f), allow_pickle=True)
        if 'knots_0' in z.files:
            k.append((len(np.unique(z['knots_0'])) - 1, len(np.unique(z['knots_q'])) - 1))
        else:
            k.append((len(np.unique(z['knots'])) - 1, np.nan))
    return np.nanmedian(np.array(k, float), axis=0)


def report():
    rows = []
    for s, p, kn in _configs():
        cfg = RP.STUDIES[s]
        root = _root(s, p, kn)
        r = RP.summarize(s, root)
        if r is None:
            continue
        l0, lq = n_knots(s, root)
        for name in ('beta_2', 'beta_3'):
            c = r['coef'][name]
            rows.append(dict(setting=cfg['setting'], estimator=cfg['estimator'],
                             knots=kn, p=round(p, 2),
                             paper_placement=kn == _default_knots(s),
                             default=bool(np.isclose(p, DEFAULT_P[s])
                                          and kn == _default_knots(s)),
                             intervals_alpha_or_single=l0, intervals_q=lq,
                             reps=r['reps'], se_dropped=r['n_se_dropped'], coef=name,
                             bias=c['bias'], SE=c['se'], ESE=c['ese'], CP=c['cp']))
    T = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    T.to_csv(os.path.join(HERE, 'tables', 'table_S1_S2_knots_beta.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(T.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=9)
    a = ap.parse_args()
    run(a.reps, a.workers) if a.cmd == 'run' else report()
