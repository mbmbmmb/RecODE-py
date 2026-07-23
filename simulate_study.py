"""Monte-Carlo simulation study driven end-to-end by ode_unify.

For each of the paper's 7 canonical settings this simulates ``reps`` datasets
with the unified generator (:mod:`ode_unify.dgp`), fits them with the unified
estimator + inference (:mod:`ode_unify.estimator` / :mod:`ode_unify.inference`),
persists each replication to ``results/<slug>/seed<k>.npz`` (resumable), and
renders the functional-parameter band plots with the unified visual module
(:mod:`ode_unify.visual`).

By default the study runs with ``layout='legacy'`` so every per-seed result is
bit-identical to the historical per-model pipelines (and hence to the plots in
``recurrent_ode/plots``); pass ``--layout uniform`` for the package's uniform
memory layout (statistically identical).

Usage::

    python -m ode_unify.simulate_study all  --reps 100 --workers 9
    python -m ode_unify.simulate_study run  --only cox_setting1
    python -m ode_unify.simulate_study plot --only aft_setting2
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ode_unify as U  # noqa: E402
from ode_unify.estimator import Estimate  # noqa: E402
from ode_unify import visual  # noqa: E402

DEFAULT_RESULTS = os.path.join(HERE, 'results')
DEFAULT_PLOTS = os.path.join(HERE, 'plots')


def _lin(a, b, n=60):
    return np.linspace(a, b, n)


STUDIES = {
    'cox_setting1': dict(
        estimator='cox', random_effect=False, data_setting=1, knots=None,
        N=1000, kind='single',
        truth=lambda u: u ** 2 + 1.0, grid=_lin(0.1, 2.5),
        title='cox / setting 1  (λ₀(t)=t²+1)',
        ylabel=r'$\lambda_0(t)$', out='cox_s1.png'),
    'aft_setting2': dict(
        estimator='aft', random_effect=False, data_setting=2,
        knots='quantile', N=1000, kind='single',
        truth=lambda u: 2.0 / (1.0 + u), grid=_lin(0.1, 3.5),
        title='aft / setting 2  (q(u)=2/(1+u))',
        ylabel='q(u)', out='aft_s2.png'),
    'npmle_setting3': dict(
        estimator='npmle', random_effect=False, data_setting=3,
        knots='equal', N=2000, kind='single',
        truth=lambda u: 0.2 / (1.0 + u), grid=_lin(0.05, 2.5),
        title='npmle / setting 3  (q(u)=0.2/(1+u))',
        ylabel='q(u)', out='npmle_s3.png'),
    're_cox_setting1': dict(
        estimator='cox', random_effect=True, data_setting=1, knots=None,
        N=1000, kind='single',
        truth=lambda u: u ** 2 + 1.0, grid=_lin(0.1, 2.5),
        title='random_effect.cox / setting 1  (λ₀(t)=t²+1)',
        ylabel=r'$\lambda_0(t)$', out='re_cox_s1.png'),
    're_aft_setting2': dict(
        estimator='aft', random_effect=True, data_setting=2,
        knots='quantile', N=1000, kind='single',
        truth=lambda u: 2.0 / (1.0 + u), grid=_lin(0.1, 3.5),
        title='random_effect.aft / setting 2  (q(u)=2/(1+u))',
        ylabel='q(u)', out='re_aft_s2.png'),
    'ltm_setting4': dict(
        estimator='ltm', random_effect=False, data_setting=4, knots='K4',
        N=1000, kind='ltm',
        truth_alpha=lambda t: t + 1.0, truth_q=lambda u: 2.0 / (1.0 + u),
        grid_t=_lin(0.05, 2.0), grid_u=_lin(0.05, 2.0),
        scale_a=3.0, scale_q=1.0 / 3.0, use_median=False,
        title_alpha='ltm / setting 4  (α(t)=t+1, scaled ×3)',
        title_q='ltm / setting 4  (q(u)=2/(1+u), scaled ×1/3)',
        out_alpha='ltm_s4_alpha.png', out_q='ltm_s4_q.png'),
    're_ltm_setting1': dict(
        estimator='ltm', random_effect=True, data_setting=1, knots='K4',
        N=1000, kind='ltm',
        truth_alpha=lambda t: t ** 2 + 1.0,
        truth_q=lambda u: np.ones_like(u),
        grid_t=_lin(0.01, 2.0), grid_u=_lin(0.01, 10.0),
        scale_a=1.5 ** 2 + 1.0, scale_q=1.0 / (1.5 ** 2 + 1.0),
        use_median=True,
        title_alpha='random_effect.ltm / setting 1  (α(t)=t²+1)',
        title_q='random_effect.ltm / setting 1  (q(u)=1)',
        out_alpha='re_ltm_s1_alpha.png', out_q='re_ltm_s1_q.png'),
}


# --------------------------------------------------------------------------- #
# run one replication (worker process)
# --------------------------------------------------------------------------- #

def _run_one(args):
    slug, seed, out_root, layout = args
    cfg = STUDIES[slug]
    out_dir = os.path.join(out_root, slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'seed{seed}.npz')
    if os.path.isfile(out_path):
        return slug, seed, 'skip', 0.0
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        data = U.simulate(cfg['N'], seed, cfg['data_setting'],
                          random_effect=cfg['random_effect'])
        est = U.fit(data, estimator=cfg['estimator'],
                    random_effect=cfg['random_effect'], knots=cfg['knots'],
                    ci=True, seed=seed, data_setting=cfg['data_setting'],
                    spline_se=True, layout=layout)
    payload = {k: np.asarray(v) for k, v in est.raw.items()}
    payload['beta'] = est.beta
    if est.se is not None:
        payload['se'] = est.se
    payload['_success'] = np.array(bool(est.success))
    payload['_estimator'] = np.array(cfg['estimator'])
    payload['_random_effect'] = np.array(bool(cfg['random_effect']))
    payload['_knots'] = np.array(cfg['knots'] or '')
    np.savez_compressed(out_path, **payload)
    return slug, seed, 'ok', time.time() - t0


def run_study(slug, reps, seed0=1, out_root=DEFAULT_RESULTS, workers=8,
              layout='legacy'):
    if slug not in STUDIES:
        raise KeyError(f'unknown study {slug!r}')
    tasks = [(slug, s, out_root, layout) for s in range(seed0, seed0 + reps)]
    done = ok = skipped = failed = 0
    t_start = time.time()
    print(f'[{slug}] {reps} reps on {workers} workers (layout={layout}) ...')
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            try:
                _, seed, status, _ = fut.result()
                ok += status == 'ok'
                skipped += status == 'skip'
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f'  seed {futs[fut][1]} FAILED: '
                      f'{type(exc).__name__}: {exc}')
            if done % max(1, reps // 10) == 0 or done == reps:
                print(f'  {done}/{reps}  (ok={ok} skip={skipped} '
                      f'fail={failed}, {time.time() - t_start:.0f}s)')
    print(f'[{slug}] done: ok={ok} skip={skipped} fail={failed} '
          f'in {time.time() - t_start:.0f}s')
    return ok, skipped, failed


# --------------------------------------------------------------------------- #
# rebuild Estimates from saved results, plot via ode_unify.visual
# --------------------------------------------------------------------------- #

def _load_estimate(path, cfg):
    f = np.load(path)
    est_r = f['est_r'].ravel()
    p = int(f['p'].ravel()[0])
    beta = est_r[:p].copy()
    is_ltm = cfg['estimator'] == 'ltm'
    if is_ltm:
        beta[0] = 1.0
        q_q = int(f['q_q'].ravel()[0])
        spline = {'knots_0': f['knots_0'].ravel(),
                  'knots_q': f['knots_q'].ravel(),
                  'k0': int(f['k0'].ravel()[0]), 'kq': int(f['kq'].ravel()[0]),
                  'q_0': int(f['q_0'].ravel()[0]), 'q_q': q_q,
                  'coefs_q': est_r[p:p + q_q],
                  'coefs_alpha': est_r[p + q_q:]}
    else:
        spline = {'knots': f['knots'].ravel(), 'k': int(f['k'].ravel()[0]),
                  'coefs': est_r[p:]}
    se_all = f['se_all'].ravel() if 'se_all' in f.files else None
    success = bool(f['_success'].ravel()[0]) if '_success' in f.files else True
    return Estimate(beta=beta, spline=spline, estimator=cfg['estimator'],
                    random_effect=cfg['random_effect'],
                    knots_setting=cfg['knots'], seed=0, runtime=0.0,
                    success=success, se_all=se_all)


def _load_all(slug, out_root, cfg):
    d = os.path.join(out_root, slug)
    if not os.path.isdir(d):
        raise FileNotFoundError(f'no results for {slug} in {out_root}')
    files = sorted(f for f in os.listdir(d)
                   if f.startswith('seed') and f.endswith('.npz'))
    return [_load_estimate(os.path.join(d, f), cfg) for f in files]


def plot_study(slug, out_root=DEFAULT_RESULTS, plot_root=DEFAULT_PLOTS):
    cfg = STUDIES[slug]
    ests = _load_all(slug, out_root, cfg)
    os.makedirs(plot_root, exist_ok=True)
    if cfg['kind'] == 'single':
        out = visual.band_plot(
            ests, os.path.join(plot_root, cfg['out']),
            truth=cfg['truth'], grid=cfg['grid'], title=cfg['title'],
            ylabel=cfg['ylabel'])
        print(f'  wrote {out}')
        return [out]
    pa, pq = visual.ltm_band_plot(
        ests, os.path.join(plot_root, cfg['out_alpha']),
        os.path.join(plot_root, cfg['out_q']),
        truth_alpha=cfg['truth_alpha'], truth_q=cfg['truth_q'],
        grid_t=cfg['grid_t'], grid_u=cfg['grid_u'],
        scale_a=cfg['scale_a'], scale_q=cfg['scale_q'],
        title_alpha=cfg['title_alpha'], title_q=cfg['title_q'],
        use_median=cfg['use_median'])
    print(f'  wrote {pa}\n  wrote {pq}')
    return [pa, pq]


def _selected(only):
    if not only:
        return list(STUDIES)
    missing = [s for s in only if s not in STUDIES]
    if missing:
        raise SystemExit(f'unknown study(ies): {missing}')
    return only


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['run', 'plot', 'all', 'list'])
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--seed0', type=int, default=1)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--layout', choices=['legacy', 'uniform'],
                    default='legacy')
    ap.add_argument('--results', default=DEFAULT_RESULTS)
    ap.add_argument('--plots', default=DEFAULT_PLOTS)
    args = ap.parse_args(argv)

    if args.command == 'list':
        for slug, c in STUDIES.items():
            re = ' RE' if c['random_effect'] else '   '
            print(f'  {slug:20s} estimator={c["estimator"]:5s}{re} '
                  f'setting={c["data_setting"]} knots={c["knots"]} N={c["N"]}')
        return

    selected = _selected(args.only)
    if args.command == 'all':
        for slug in selected:
            run_study(slug, args.reps, seed0=args.seed0,
                      out_root=args.results, workers=args.workers,
                      layout=args.layout)
            print(f'[{slug}] plotting ...')
            try:
                plot_study(slug, out_root=args.results, plot_root=args.plots)
            except Exception as e:  # noqa: BLE001
                print(f'  plot FAILED for {slug}: {type(e).__name__}: {e}')
        return
    if args.command == 'run':
        for slug in selected:
            run_study(slug, args.reps, seed0=args.seed0,
                      out_root=args.results, workers=args.workers,
                      layout=args.layout)
    if args.command == 'plot':
        for slug in selected:
            print(f'[{slug}] plotting ...')
            try:
                plot_study(slug, out_root=args.results, plot_root=args.plots)
            except FileNotFoundError as e:
                print(f'  skip: {e}')


if __name__ == '__main__':
    main()
