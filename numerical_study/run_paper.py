"""Monte-Carlo study reproducing every simulation setting reported in §5 of
``latex/main.tex``, using the paper-faithful generator
:func:`ode_unify.paper_dgp.simulate_paper`.

Unlike ``ode_unify/simulate_study.py`` (which uses the general
:func:`ode_unify.dgp.simulate` defaults -- Setting-3 covariates and ``U(2,4)``
censoring for every setting), every study here draws covariates and censoring
times exactly as the paper specifies them, so the summaries are directly
comparable to Tables 1-3.

Study coverage
--------------
* ``t1_*``  Table 1: ODE-Cox (S1), ODE-AM (S2), ODE-LT (S3), n=1000
* ``t2_*``  Table 2: ODE-Flex, settings 1-4, n=1000
* ``t3_*``  Table 3: settings 5-6 (Gamma frailty), ODE-Cox/ODE-AM and
  ODE-Flex, n=2000 and n=4000

The competitor columns of Tables 1 and 3 (``reReg`` cox.LWYY / am.GL, the
Zeng-Lin NPMLE, ``reda``) are **not** run here -- they are not part of the local
ODE module. Their published values are carried over verbatim in the report.

Usage::

    python -m ode_unify.numerical_study.run_paper list
    python -m ode_unify.numerical_study.run_paper all  --reps 100 --workers 9
    python -m ode_unify.numerical_study.run_paper run  --only t1_cox_s1 --reps 100
    python -m ode_unify.numerical_study.run_paper plot
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
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(PKG)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ode_unify as U                                    # noqa: E402
from ode_unify.estimator import Estimate                 # noqa: E402
from ode_unify.inference import inference                # noqa: E402
from ode_unify.paper_dgp import simulate_paper           # noqa: E402
from ode_unify import visual                             # noqa: E402

DEFAULT_RESULTS = os.path.join(HERE, 'simulation_study', 'random_censoring',
                               'results')
DEFAULT_PLOTS = os.path.join(HERE, 'simulation_study', 'random_censoring')


def _lin(a, b, n=60):
    return np.linspace(a, b, n)


# true functional parameters per paper setting -------------------------------
TRUE_ALPHA = {
    1: lambda t: t ** 2 + 1.0,
    2: lambda t: np.ones_like(np.asarray(t, float)),
    3: lambda t: 0.2 / (1.0 + t),
    4: lambda t: t + 1.0,
    5: lambda t: t ** 2 + 1.0,
    6: lambda t: np.ones_like(np.asarray(t, float)),
}
TRUE_Q = {
    1: lambda u: np.ones_like(np.asarray(u, float)),
    2: lambda u: 2.0 / (1.0 + u),
    3: lambda u: 1.0 / (u / 2.0 + 1.0),
    4: lambda u: 2.0 / (1.0 + u),
    5: lambda u: np.ones_like(np.asarray(u, float)),
    6: lambda u: 2.0 / (1.0 + u),
}
# time support (roughly the censoring range) used for the band plots
TMAX = {1: 2.0, 2: 3.0, 3: 4.0, 4: 3.0, 5: 2.0, 6: 3.0}

# ODE-Flex identifies (alpha, q) only up to alpha -> c*alpha, q -> q/c, so the
# curves must be put on the truth's scale before they can be compared. §5.1
# does this by rescaling "alpha(t) such that alpha(t0) = alpha_0(t0)", with t0
# the median observed event time -- which is why every alpha curve in the
# paper's Figure 3 passes through a single node at t ~ 0.9 for Setting 1.
#
# The solver's own constraint pins alpha_hat(2.0) = 1 (1.5 for the frailty
# solver), NOT alpha_hat(median) = 1. Anchoring the rescaling there instead is
# what the earlier version of this file did, and for Setting 1 it is a disaster:
# the censoring is U(0, 2), so t = 2 is the extreme edge of support where alpha
# is barely identified, and normalising by a value estimated there threw the
# whole curve off by 250%. Anchoring at the median event time -- inside the data
# -- brings the same fits to within a few percent of the truth.
#
# SUPPORT[(setting, N)] = (t_anchor, t_lo, t_hi, u_lo, u_hi), averaged over 5
# pilot replications:
#   t_anchor  median observed event time -- the rescaling anchor of §5.1
#   t_lo/t_hi 2.5th / 95th percentile of observed event times
#   u_lo/u_hi 2.5th / 95th percentile of the true mean function mu evaluated at
#             the observed event times -- i.e. the range q(.) is actually
#             evaluated over
#
# The curves are plotted only on this data-supported region. Outside it the
# sieve is extrapolating: at the extreme upper tail of t almost no subject is
# still at risk, alpha is barely identified there, and both the estimate and its
# band degrade sharply. Trimming to the supported range is what the paper's own
# figures do -- its Setting-1 panel stops at t ~ 1.8 even though censoring runs
# to 2.0.
SUPPORT = {
    (1, 1000): (0.8810, 0.0405, 1.7169, 0.0619, 15.3625),
    (2, 1000): (0.7022, 0.0200, 2.2036, 0.0546, 4.1491),
    (3, 1000): (0.9242, 0.0258, 3.1858, 0.0200, 2.0338),
    (4, 1000): (0.9136, 0.0292, 2.3557, 0.0820, 6.4971),
    (5, 2000): (0.8744, 0.0382, 1.7236, 0.0597, 13.9652),
    (5, 4000): (0.8730, 0.0419, 1.7132, 0.0652, 14.7767),
    (6, 2000): (0.7085, 0.0200, 2.1617, 0.0557, 4.2367),
    (6, 4000): (0.7063, 0.0206, 2.1773, 0.0602, 4.3049),
}


def grid_alpha(setting, N, n=60):
    _, lo, hi, _, _ = SUPPORT[(setting, N)]
    return np.linspace(lo, hi, n)


def grid_q(setting, N, n=60):
    _, _, _, lo, hi = SUPPORT[(setting, N)]
    return np.linspace(lo, hi, n)


def flex_scales(ests, setting, N):
    """Per-replication (scale_alpha, scale_q) putting each fit on the truth's
    scale, following §5.1: alpha_hat(t0) is matched to alpha_0(t0) at the median
    event time t0, and q is divided by the same factor so that the product
    q(mu) * alpha(t) is unchanged.

    The solver's own constraint pins alpha_hat(2.0) = 1 (1.5 for the frailty
    solver), NOT alpha_hat(median) = 1. Anchoring the rescaling at the solver's
    point is a disaster for Setting 1, whose censoring is U(0, 2): t = 2 is the
    extreme edge of support, and normalising by a value estimated there threw the
    whole curve off by 250% (pointwise coverage 0.065). Anchoring at the median
    event time -- inside the data -- brings the same fits to within a few percent.
    """
    from ode_unify.visual import curve
    t0 = SUPPORT[(setting, N)][0]
    a0 = float(np.asarray(TRUE_ALPHA[setting](t0)).ravel()[0])
    ahat = np.array([float(curve(e, [t0], which='alpha')[0][0]) for e in ests])
    ahat = np.where(np.abs(ahat) < 1e-12, np.nan, ahat)
    c = a0 / ahat
    return c, 1.0 / c


def _single(slug, est, setting, N, knots, re, label, truth, grid, ylabel, out,
            group='misc'):
    return dict(slug=slug, estimator=est, setting=setting, N=N, knots=knots,
                random_effect=re, kind='single', label=label, group=group,
                truth=truth, grid=grid, ylabel=ylabel, out=out)


STUDIES = {
    # ---- Table 1: specified-functional-parameter estimators, n=1000 --------
    't1_cox_s1': _single(
        't1_cox_s1', 'cox', 1, 1000, None, False,
        'Table 1 / Setting 1 / ODE-Cox',
        TRUE_ALPHA[1], grid_alpha(1, 1000), r'$\alpha(t)=t^2+1$',
        'setting1.png', group='cox'),
    't1_am_s2': _single(
        't1_am_s2', 'aft', 2, 1000, 'quantile', False,
        'Table 1 / Setting 2 / ODE-AM',
        TRUE_Q[2], grid_q(2, 1000), r'$q(u)=2/(1+u)$', 'setting2.png',
        group='am'),
    't1_lt_s3': _single(
        't1_lt_s3', 'npmle', 3, 1000, 'equal', False,
        'Table 1 / Setting 3 / ODE-LT',
        TRUE_ALPHA[3], grid_alpha(3, 1000), r'$\alpha(t)=0.2/(1+t)$',
        'setting3_lt.png', group='ltm'),
    # ---- Table 2: ODE-Flex on settings 1-4, n=1000 ------------------------
    # Knot placement follows §5.1's per-setting prescription, and each choice
    # was confirmed by a knot sweep (30 seeds, bias on beta_2/beta_3):
    #   Setting 1 -- "equally spaced" -> K1 (+0.008/-0.002); K4 gave -0.015/-0.024
    #   Setting 3 -- quantiles for log alpha only -> K3 (+0.001/-0.049);
    #                K4 gave +0.096 because Setting 3 yields only ~0.7 events per
    #                subject, so the q-quantiles cluster in a narrow range
    #   Setting 4 -- "quantiles" for both -> K4
    # Setting 2's sweep was inconclusive at 30 seeds (MC SE ~0.016); K4 is kept
    # because its full 100-rep run reproduces the published row closely.
    **{f't2_flex_s{s}': dict(
        slug=f't2_flex_s{s}', estimator='ltm', setting=s, N=1000,
        knots={1: 'K1', 2: 'K4', 3: 'K3', 4: 'K4'}[s],
        random_effect=False, kind='ltm',
        label=f'Table 2 / Setting {s} / ODE-Flex',
        truth_alpha=TRUE_ALPHA[s], truth_q=TRUE_Q[s],
        grid_t=grid_alpha(s, 1000), grid_u=grid_q(s, 1000),
        group='ltm',
        out_alpha=f'setting{s}_flex_alpha.png', out_q=f'setting{s}_flex_q.png')
       for s in (1, 2, 3, 4)},
    # ---- Table 3: Gamma frailty, settings 5-6, n=2000 and 4000 ------------
    **{f't3_cox_s5_n{n}': _single(
        f't3_cox_s5_n{n}', 'cox', 5, n, None, True,
        f'Table 3 / Setting 5 / ODE-Cox / n={n}',
        TRUE_ALPHA[5], grid_alpha(5, n), r'$\alpha(t)=t^2+1$',
        f'setting5_n{n}.png', group='random_effect/cox')
       for n in (2000, 4000)},
    **{f't3_am_s6_n{n}': _single(
        f't3_am_s6_n{n}', 'aft', 6, n, 'quantile', True,
        f'Table 3 / Setting 6 / ODE-AM / n={n}',
        TRUE_Q[6], grid_q(6, n), r'$q(u)=2/(1+u)$',
        f'setting6_n{n}.png', group='random_effect/am')
       for n in (2000, 4000)},
    **{f't3_flex_s{s}_n{n}': dict(
        slug=f't3_flex_s{s}_n{n}', estimator='ltm', setting=s, N=n,
        knots='K4', random_effect=True, kind='ltm',
        label=f'Table 3 / Setting {s} / ODE-Flex / n={n}',
        truth_alpha=TRUE_ALPHA[s], truth_q=TRUE_Q[s],
        grid_t=grid_alpha(s, n), grid_u=grid_q(s, n),
        resample_B=(1500 if s == 5 else 2000), group='random_effect/ltm',
        out_alpha=f'setting{s}_flex_n{n}_alpha.png',
        out_q=f'setting{s}_flex_n{n}_q.png')
       for s in (5, 6) for n in (2000, 4000)},
}


# --------------------------------------------------------------------------- #
# one replication
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
        data = simulate_paper(cfg['N'], seed, cfg['setting'])
        est = U.estimate(data, estimator=cfg['estimator'],
                         random_effect=cfg['random_effect'],
                         knots=cfg['knots'], seed=seed, layout=layout)
        # the RE-LTM resampling only knows data_setting 1 (cox-type) and
        # 2 (aft-type); paper settings 5/6 are their frailty counterparts.
        ds = {5: 1, 6: 2}.get(cfg['setting'], cfg['setting'])
        # §5.2 uses B=1500 (Setting 5) and B=2000 (Setting 6) resampling draws
        # for ODE-Flex; the engine default (800/1000) inflates the ESE.
        rb = cfg.get('resample_B')
        est = inference(est, data, spline_se=True, data_setting=ds, seed=seed,
                        resample_B=rb)
    payload = {k: np.asarray(v) for k, v in est.raw.items()}
    payload['beta'] = est.beta
    if est.se is not None:
        payload['se'] = est.se
    payload['_success'] = np.array(bool(est.success))
    payload['_estimator'] = np.array(cfg['estimator'])
    payload['_random_effect'] = np.array(bool(cfg['random_effect']))
    payload['_knots'] = np.array(cfg['knots'] or '')
    payload['_setting'] = np.array(cfg['setting'])
    payload['_N'] = np.array(cfg['N'])
    payload['_n_events'] = np.array(int((data['delta'].ravel() == 1).sum()))
    np.savez_compressed(out_path, **payload)
    return slug, seed, 'ok', time.time() - t0


def run_study(slug, reps, seed0=1, out_root=DEFAULT_RESULTS, workers=10,
              layout='legacy'):
    if slug not in STUDIES:
        raise KeyError(f'unknown study {slug!r}')
    tasks = [(slug, s, out_root, layout) for s in range(seed0, seed0 + reps)]
    done = ok = skipped = failed = 0
    t_start = time.time()
    print(f'[{slug}] {reps} reps on {workers} workers ...', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            try:
                _, seed, status, _ = fut.result()
                ok += status == 'ok'
                skipped += status == 'skip'
            except Exception as exc:                       # noqa: BLE001
                failed += 1
                print(f'  seed {futs[fut][1]} FAILED: '
                      f'{type(exc).__name__}: {exc}', flush=True)
            if done % max(1, reps // 10) == 0 or done == reps:
                print(f'  {done}/{reps}  (ok={ok} skip={skipped} '
                      f'fail={failed}, {time.time() - t_start:.0f}s)', flush=True)
    print(f'[{slug}] done: ok={ok} skip={skipped} fail={failed} '
          f'in {time.time() - t_start:.0f}s', flush=True)
    return ok, skipped, failed


# Measured single-seed cost in seconds (from the 400-replication run, 10
# workers). Used only to schedule longest-job-first in the pooled runner, so the
# expensive frailty studies start immediately and the cheap ones backfill.
SEED_COST = {
    't3_flex_s6_n4000': 113.0, 't3_flex_s6_n2000': 89.0,
    't3_flex_s5_n4000': 78.0, 't3_flex_s5_n2000': 66.0,
    't3_am_s6_n4000': 60.0, 't3_cox_s5_n4000': 39.0,
    't3_am_s6_n2000': 29.0, 't1_am_s2': 5.3, 't3_cox_s5_n2000': 3.3,
    't2_flex_s2': 1.5, 't2_flex_s3': 1.4, 't2_flex_s4': 1.4,
    't2_flex_s1': 1.1, 't1_lt_s3': 1.0, 't1_cox_s1': 0.6,
}


def run_pooled(slugs, reps, seed0=1, out_root=DEFAULT_RESULTS, workers=10,
               layout='legacy'):
    """Run every (study, seed) task through ONE process pool.

    Running each study in its own pool leaves workers idle at every study's
    tail; pooling the whole grid keeps all of them busy to the end. Tasks are
    submitted longest-expected-first so the multi-minute frailty ODE-Flex seeds
    are never left to run alone at the finish.

    Already-computed seeds are filtered out here rather than dispatched and
    skipped, so extending an existing study only queues the new seeds.
    """
    tasks = []
    for slug in slugs:
        d = os.path.join(out_root, slug)
        for seed in range(seed0, seed0 + reps):
            if not os.path.isfile(os.path.join(d, f'seed{seed}.npz')):
                tasks.append((slug, seed, out_root, layout))
    tasks.sort(key=lambda t: -SEED_COST.get(t[0], 1.0))
    est = sum(SEED_COST.get(t[0], 1.0) for t in tasks) / max(workers, 1)
    todo = {}
    for t in tasks:
        todo[t[0]] = todo.get(t[0], 0) + 1
    print(f'pooled run: {len(tasks)} new tasks over {len(todo)} studies, '
          f'{workers} workers', flush=True)
    print(f'  estimated wall time ~{est/3600:.1f}h', flush=True)
    for slug in sorted(todo, key=lambda k: -SEED_COST.get(k, 1.0)):
        print(f'    {slug:20s} {todo[slug]:5d} seeds', flush=True)
    if not tasks:
        print('nothing to do', flush=True)
        return

    done = ok = failed = 0
    per = {}
    cost_total = sum(SEED_COST.get(t[0], 1.0) for t in tasks)
    cost_done = 0.0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            slug = futs[fut][0]
            try:
                fut.result()
                ok += 1
                per[slug] = per.get(slug, 0) + 1
                if per[slug] == todo[slug]:
                    print(f'  [{slug}] complete ({todo[slug]} seeds, '
                          f'{time.time() - t_start:.0f}s elapsed)', flush=True)
            except Exception as exc:                       # noqa: BLE001
                failed += 1
                print(f'  seed {futs[fut][1]} of {slug} FAILED: '
                      f'{type(exc).__name__}: {exc}', flush=True)
            cost_done += SEED_COST.get(slug, 1.0)
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t_start
                # Tasks are dispatched longest-first, so extrapolating from the
                # completed COUNT badly overestimates the remaining time early
                # on (the expensive seeds all finish first). Extrapolate from
                # the completed COST instead.
                frac = cost_done / cost_total if cost_total else 1.0
                eta = el * (1.0 - frac) / frac if frac > 0 else 0.0
                print(f'  {done}/{len(tasks)} (ok={ok} fail={failed}) '
                      f'{frac*100:.1f}% of work, {el/3600:.2f}h elapsed, '
                      f'ETA {eta/3600:.2f}h', flush=True)
    print(f'pooled run done: ok={ok} fail={failed} in '
          f'{(time.time() - t_start)/3600:.2f}h', flush=True)


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #

Z = 1.959963984540054


SE_OUTLIER_FACTOR = 5.0


def summarize(slug, out_root=DEFAULT_RESULTS, truth=1.0,
              se_outlier_factor=SE_OUTLIER_FACTOR):
    """Bias / SE / ESE / CP per coefficient, matching the paper's definitions.

    A few percent of random-effect ODE-Flex replications return a standard
    error that is wildly inflated (up to 200x the median) because the resampled
    derivative matrix comes out near-singular; the point estimate in those
    replications is perfectly ordinary. Replications whose reported SE exceeds
    ``se_outlier_factor`` times the median SE are therefore treated as
    *variance-computation failures*: they still contribute to ``bias`` and to
    the empirical ``SE`` (which use only the point estimates), but are excluded
    from ``ESE`` and ``CP`` (which depend on the failed SE). The count is
    reported as ``n_se_dropped`` so the exclusion is always visible.
    """
    cfg = STUDIES[slug]
    d = os.path.join(out_root, slug)
    files = sorted(f for f in os.listdir(d)
                   if f.startswith('seed') and f.endswith('.npz')) \
        if os.path.isdir(d) else []
    B, S, OK, RT, EV = [], [], [], [], []
    for f in files:
        z = np.load(os.path.join(d, f), allow_pickle=True)
        B.append(z['beta'].ravel())
        S.append(z['se'].ravel() if 'se' in z.files
                 else np.full(z['beta'].size, np.nan))
        OK.append(bool(z['_success']))
        RT.append(float(z['runtime']) if 'runtime' in z.files else np.nan)
        EV.append(int(z['_n_events']) if '_n_events' in z.files else 0)
    if not B:
        return None
    B, S = np.array(B), np.array(S)
    is_ltm = cfg['estimator'] == 'ltm'
    idx = [1, 2] if is_ltm else [0, 1, 2]

    # a replication is a variance failure if ANY reported SE is far above the
    # median for its coefficient
    with np.errstate(invalid='ignore'):
        med = np.nanmedian(S[:, idx], axis=0)
        bad = np.any(~np.isfinite(S[:, idx])
                     | (S[:, idx] > se_outlier_factor * med), axis=1)

    rows = {}
    for j in idx:
        b, se = B[:, j], S[:, j]
        g = ~bad
        rows[f'beta_{j + 1}'] = dict(
            bias=float(b.mean() - truth), se=float(b.std(ddof=1)),
            ese=float(np.nanmean(se[g])),
            cp=float(np.mean(np.abs(b[g] - truth) <= Z * se[g])))
    return dict(slug=slug, label=cfg['label'], reps=len(B),
                success=int(np.sum(OK)), n_se_dropped=int(bad.sum()),
                median_runtime=float(np.nanmedian(RT)),
                mean_events=float(np.mean(EV)), coef=rows)


def report(out_root=DEFAULT_RESULTS, only=None):
    print(f'{"study":22s}{"reps":>5s}{"ok":>5s}{"drop":>5s}{"ev/rep":>9s}'
          f'{"coef":>8s}{"Bias":>9s}{"SE":>8s}{"ESE":>8s}{"CP":>7s}')
    print('-' * 86)
    for slug in (only or STUDIES):
        r = summarize(slug, out_root)
        if r is None:
            print(f'{slug:22s}  (no results)')
            continue
        first = True
        for name, c in r['coef'].items():
            head = f'{slug:22s}{r["reps"]:>5d}{r["success"]:>5d}' \
                   f'{r["n_se_dropped"]:>5d}{r["mean_events"]:>9.0f}' \
                if first else ' ' * 46
            print(f'{head}{name:>8s}{c["bias"]:>+9.4f}{c["se"]:>8.4f}'
                  f'{c["ese"]:>8.4f}{c["cp"]:>7.3f}')
            first = False


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

def _load_estimate(path, cfg):
    f = np.load(path, allow_pickle=True)
    est_r = f['est_r'].ravel()
    p = int(f['p'].ravel()[0])
    beta = est_r[:p].copy()
    if cfg['estimator'] == 'ltm':
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


def plot_study(slug, out_root=DEFAULT_RESULTS, plot_root=DEFAULT_PLOTS):
    cfg = STUDIES[slug]
    d = os.path.join(out_root, slug)
    if not os.path.isdir(d):
        raise FileNotFoundError(f'no results for {slug}')
    files = sorted(f for f in os.listdir(d)
                   if f.startswith('seed') and f.endswith('.npz'))
    ests = [_load_estimate(os.path.join(d, f), cfg) for f in files]
    plot_dir = os.path.join(plot_root, cfg.get('group', 'misc'))
    os.makedirs(plot_dir, exist_ok=True)
    if cfg['kind'] == 'single':
        out = visual.band_plot(
            ests, os.path.join(plot_dir, cfg['out']), truth=cfg['truth'],
            grid=cfg['grid'], title=cfg['label'], ylabel=cfg['ylabel'])
        print(f'  wrote {out}')
        return [out]
    sa, sq = flex_scales(ests, cfg['setting'], cfg['N'])
    pa, pq = visual.ltm_band_plot(
        ests, os.path.join(plot_dir, cfg['out_alpha']),
        os.path.join(plot_dir, cfg['out_q']),
        truth_alpha=cfg['truth_alpha'], truth_q=cfg['truth_q'],
        grid_t=cfg['grid_t'], grid_u=cfg['grid_u'],
        scale_a=sa, scale_q=sq,
        title_alpha=cfg['label'] + r'  $\hat\alpha$',
        title_q=cfg['label'] + r'  $\hat q$', use_median=True)
    print(f'  wrote {pa}\n  wrote {pq}')
    return [pa, pq]


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['run', 'plot', 'all', 'list', 'report'])
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--seed0', type=int, default=1)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--layout', choices=['legacy', 'uniform'], default='legacy')
    ap.add_argument('--sequential', action='store_true',
                    help='run each study in its own pool (default: pool every '
                         '(study, seed) task together, which keeps all workers '
                         'busy through the tail of each study)')
    ap.add_argument('--results', default=DEFAULT_RESULTS)
    ap.add_argument('--plots', default=DEFAULT_PLOTS)
    args = ap.parse_args(argv)

    if args.command == 'list':
        for slug, c in STUDIES.items():
            re_ = ' RE' if c['random_effect'] else '   '
            print(f'  {slug:20s} est={c["estimator"]:5s}{re_} '
                  f'setting={c["setting"]} N={c["N"]:5d} '
                  f'knots={str(c["knots"]):9s} {c["label"]}')
        return
    if args.command == 'report':
        report(args.results, args.only)
        return

    sel = args.only or list(STUDIES)
    bad = [s for s in sel if s not in STUDIES]
    if bad:
        raise SystemExit(f'unknown study(ies): {bad}')

    if args.command in ('run', 'all'):
        if args.sequential:
            for slug in sel:
                run_study(slug, args.reps, seed0=args.seed0,
                          out_root=args.results, workers=args.workers,
                          layout=args.layout)
        else:
            run_pooled(sel, args.reps, seed0=args.seed0,
                       out_root=args.results, workers=args.workers,
                       layout=args.layout)
    if args.command in ('plot', 'all'):
        for slug in sel:
            print(f'[{slug}] plotting ...', flush=True)
            try:
                plot_study(slug, args.results, args.plots)
            except Exception as e:                          # noqa: BLE001
                print(f'  plot FAILED: {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
