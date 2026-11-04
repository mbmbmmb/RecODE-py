"""Figure S1 of the Supplementary Material: fitted log q in the three settings with an
oscillating q (ODE-AM, settings 13-15, n = 2000, 20 replications) for four values of the
knot exponent p and both knot placements.

The fits are those of knot_single.py, but every replication is evaluated on one common
grid per setting, so that the fits can be compared point by point across replications.
The grid covers [0, 0.75 u], where u is the smallest 95th percentile of mu_hat over the
replications; few observations and few knots lie beyond it.

Usage:  python knot_curves.py [--workers 9]
        (writes bic_model_selection/results/knot_curves_s13_15.npz)
"""
import argparse
import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

from ode_unify._engine.ltm.cox_rec import cox_rec                      # noqa: E402
from ode_unify.paper_dgp import CUSTOM_FUNCS, simulate_paper         # noqa: E402
import knot_single as KS                                              # noqa: E402

SETTINGS, REPS, N = (13, 14, 15), 20, 2000
PS = [0.05, 0.14, 0.25, 0.35]
FRAC = 0.75


def _u95(args):
    s, seed = args
    with contextlib.redirect_stdout(io.StringIO()):
        d = simulate_paper(N, seed, s)
        temp2, _ = cox_rec(np.ascontiguousarray(d['x']), d['time'].ravel(),
                           d['delta'].ravel())
    return float(np.quantile(np.concatenate([[0.0], np.asarray(temp2).ravel()]), 0.95))


def main(workers):
    seeds = range(1, REPS + 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        u = list(ex.map(_u95, [(s, sd) for s in SETTINGS for sd in seeds]))
        grids = {s: np.linspace(0.0, FRAC * min(u[i * REPS:(i + 1) * REPS]), KS.NG)
                 for i, s in enumerate(SETTINGS)}
        tasks = [(s, sd, N, p, pl, grids[s]) for pl in ('quantile', 'equal')
                 for s in SETTINGS for p in PS for sd in seeds]
        res = list(ex.map(KS._one, tasks))
    out = {}
    for s in SETTINGS:
        out[f's{s}_grid'] = grids[s]
        out[f's{s}_true'] = np.log(CUSTOM_FUNCS[s]['q'](grids[s]))
        for pl in ('quantile', 'equal'):
            for p in PS:
                out[f's{s}_{pl}_p{p:g}'] = np.array(
                    [r['fit'] for r in res if r['ok'] and (r['setting'], r['placement'],
                                                           r['p']) == (s, pl, p)])
    path = os.path.join(HERE, 'bic_model_selection', 'results', 'knot_curves_s13_15.npz')
    np.savez_compressed(path, **out)
    print('wrote', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=9)
    main(ap.parse_args().workers)
