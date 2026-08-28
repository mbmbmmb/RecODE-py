"""Render ``review/simulation_settings.md`` from the paper-setting results.

Emits one section per published table, each showing the paper's value and the
locally recomputed value side by side, plus the informative-censoring study and
the figure index. Re-run after extending the replication count::

    python -m ode_unify.paper_setting.make_report --reps 100
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(PKG)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ode_unify.paper_setting.run_paper import STUDIES, summarize   # noqa: E402
from ode_unify.paper_setting import paper_values as PV             # noqa: E402
from ode_unify.paper_dgp import PAPER_DESIGN                       # noqa: E402

RESULTS = os.path.join(HERE, 'results')
PLOTS = os.path.join(HERE, 'plots')
OUT_MD = os.path.join(ROOT, 'ode_unify', 'review', 'simulation_settings.md')


def _f(v, nd=3):
    return '--' if v is None else f'{v:.{nd}f}'


def _cmp_rows(slug):
    """Rows of (coef, paper..., local...) for one study."""
    loc = summarize(slug, RESULTS)
    if loc is None:
        return None, None
    pap = PV.paper_row(slug)
    rows = []
    for coef in ('beta_2', 'beta_3'):
        if coef not in loc['coef']:
            continue
        p = pap.get(coef, (None,) * 4)
        c = loc['coef'][coef]
        rows.append((coef.replace('beta_', 'b'),
                     p[0], p[1], p[2], p[3],
                     c['bias'], c['se'], c['ese'], c['cp']))
    return loc, rows


HDR = ('| Coef | Bias (paper) | Bias (local) | SE (paper) | SE (local) '
       '| ESE (paper) | ESE (local) | CP (paper) | CP (local) |')
SEP = '|---|---|---|---|---|---|---|---|---|'


def _table(slugs, titles):
    out = []
    for slug, title in zip(slugs, titles):
        loc, rows = _cmp_rows(slug)
        if loc is None:
            out.append(f'\n**{title}** — `{slug}` — no results.\n')
            continue
        cfg = STUDIES[slug]
        drop = loc.get('n_se_dropped', 0)
        drop_txt = (f', {drop} SE outlier(s) excluded from ESE/CP'
                    if drop else '')
        out.append(f'\n**{title}** — `{slug}`, n={cfg["N"]}, '
                   f'{loc["reps"]} reps ({loc["success"]} converged), '
                   f'{loc["mean_events"]:.0f} events/rep, '
                   f'knots=`{cfg["knots"]}`{drop_txt}\n')
        out.append(HDR); out.append(SEP)
        for r in rows:
            out.append(f'| {r[0]} | {_f(r[1])} | **{_f(r[5])}** | {_f(r[2])} '
                       f'| **{_f(r[6])}** | {_f(r[3])} | **{_f(r[7])}** '
                       f'| {_f(r[4])} | **{_f(r[8])}** |')
    return '\n'.join(out)


def _agreement():
    """Compact paper-vs-local agreement summary across every study."""
    rows = ['| Study | Paper row | max abs d(Bias) | ESE local/paper | CP (local) | SE outliers dropped | Verdict |',
            '|---|---|---|---|---|---|---|']
    for slug in STUDIES:
        loc, cmp_rows = _cmp_rows(slug)
        if loc is None:
            continue
        which, key = PV.STUDY_TO_PAPER[slug]
        name = f'{which}: ' + ' / '.join(str(x) for x in
                                        (key if isinstance(key, tuple) else (key,)))
        dbias, ratios, cps = [], [], []
        for r in cmp_rows:
            if r[1] is not None:
                dbias.append(abs(r[5] - r[1]))
            if r[3]:
                ratios.append(r[7] / r[3])
            cps.append(r[8])
        db = max(dbias) if dbias else float('nan')
        rt = sum(ratios) / len(ratios) if ratios else float('nan')
        # 100 reps => Monte-Carlo SE of a bias estimate is ~SE/10
        tol = 3 * max(r[6] for r in cmp_rows) / 10
        if db <= tol and 0.8 <= rt <= 1.25:
            v = 'matches'
        elif db <= tol:
            v = 'bias matches; ESE differs'
        else:
            v = '**check**'
        dr = loc.get('n_se_dropped', 0)
        rows.append(f'| `{slug}` | {name} | {db:.4f} | {rt:.2f} '
                    f'| {min(cps):.2f}-{max(cps):.2f} '
                    f'| {dr}/{loc["reps"]} | {v} |')
    return '\n'.join(rows)


def informative_section(path):
    """Render the informative-censoring regimes from saved npz."""
    files = sorted(glob.glob(os.path.join(path, '*.npz')))
    if not files:
        return '\n_(no informative-censoring results found)_\n'
    order = ['random', 'cov', 'random_fr', 'frailty_inf']
    LABEL = {
        'random': 'A. Random censoring (baseline)',
        'cov': "B. Covariate-dependent: C = C0·exp(x'γc)",
        'random_fr': 'C. Random censoring, frailty model',
        'frailty_inf': "D. Frailty-informative: C = C0·exp(x'γc)·ξ^c",
    }
    ASSUM = {'random': 'holds', 'cov': 'holds (given x)',
             'random_fr': 'holds', 'frailty_inf': '**violated**'}
    by = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    out = ['| Regime | Assumption | Coef | Bias | Emp. SD | Mean SE | CP95 |',
           '|---|---|---|---|---|---|---|']
    summ = []
    for name in order:
        if name not in by:
            continue
        z = np.load(by[name], allow_pickle=True)
        B, S, truth = z['beta'], z['se'], z['truth']
        bias = np.nanmean(B, 0) - truth
        esd = np.nanstd(B, 0, ddof=1)
        mse = np.nanmean(S, 0)
        cov = np.nanmean(((B - 1.96 * S) <= truth) & (truth <= B + 1.96 * S), 0)
        first = True
        for j in range(len(truth)):
            lab = LABEL[name] if first else ''
            asm = ASSUM[name] if first else ''
            out.append(f'| {lab} | {asm} | b{j+1} | {bias[j]:+.4f} '
                       f'| {esd[j]:.4f} | {mse[j]:.4f} | {cov[j]:.3f} |')
            first = False
        summ.append((name, float(np.max(np.abs(bias))), float(np.mean(cov)),
                     float(np.mean(z['events']))))
    out.append('')
    out.append('| Regime | max abs bias | mean CP95 | events/rep | verdict |')
    out.append('|---|---|---|---|---|')
    for name, mb, mc, ev in summ:
        ok = 'consistent' if mc >= 0.90 else '**breaks down**'
        out.append(f'| `{name}` | {mb:.4f} | {mc:.3f} | {ev:.0f} | {ok} |')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=RESULTS)
    ap.add_argument('--plots', default=PLOTS)
    ap.add_argument('--informative',
                    default=os.path.join(RESULTS, 'informative'))
    ap.add_argument('--out', default=OUT_MD)
    args = ap.parse_args(argv)

    reps = set()
    for slug in STUDIES:
        r = summarize(slug, args.results)
        if r:
            reps.add(r['reps'])
    reps_txt = str(sorted(reps)[0]) if len(reps) == 1 else \
        f'{min(reps)}-{max(reps)}' if reps else 'n/a'

    doc = TEMPLATE.format(
        reps=reps_txt,
        design=_design_table(),
        t1=_table(['t1_cox_s1', 't1_am_s2', 't1_lt_s3'],
                  ['Setting 1 — ODE-Cox', 'Setting 2 — ODE-AM',
                   'Setting 3 — ODE-LT']),
        t1_comp=_competitor_table(PV.TABLE1,
                                  [(1, 'reReg-Cox'), (2, 'reReg-AFT'),
                                   (3, 'NPMLE')]),
        t2=_table([f't2_flex_s{s}' for s in (1, 2, 3, 4)],
                  [f'Setting {s} — ODE-Flex' for s in (1, 2, 3, 4)]),
        t3=_table(['t3_cox_s5_n2000', 't3_cox_s5_n4000',
                   't3_flex_s5_n2000', 't3_flex_s5_n4000',
                   't3_am_s6_n2000', 't3_am_s6_n4000',
                   't3_flex_s6_n2000', 't3_flex_s6_n4000'],
                  ['Setting 5 — ODE-Cox, n=2000', 'Setting 5 — ODE-Cox, n=4000',
                   'Setting 5 — ODE-Flex, n=2000', 'Setting 5 — ODE-Flex, n=4000',
                   'Setting 6 — ODE-AM, n=2000', 'Setting 6 — ODE-AM, n=4000',
                   'Setting 6 — ODE-Flex, n=2000',
                   'Setting 6 — ODE-Flex, n=4000']),
        t3_comp=_competitor_table(PV.TABLE3,
                                  [(5, 'Reda', 2000), (6, 'Reda', 2000),
                                   (5, 'Reda', 4000), (6, 'Reda', 4000)]),
        agreement=_agreement(),
        informative=informative_section(args.informative),
        plots=_plot_index(args.plots),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(doc)
    print(f'wrote {args.out}  ({len(doc.splitlines())} lines)')


def _design_table():
    rows = ['| # | alpha(t) | q(u) | Covariates | Censoring | Frailty | n (paper) |',
            '|---|---|---|---|---|---|---|']
    A = {1: 't^2+1', 2: '1', 3: '0.2/(1+t)', 4: 't+1', 5: 't^2+1', 6: '1'}
    Q = {1: '1', 2: '2/(1+u)', 3: '1/(u/2+1)', 4: '2/(1+u)', 5: '1',
         6: '2/(1+u)'}
    for s, d in PAPER_DESIGN.items():
        cov = ('N(0, 0.5) trunc +-4' if d['cov'] == 'normal'
               else 'x1,x2 ~ N(0,1) trunc +-1; x3 ~ Bern(0.5)')
        c = d['censor']
        cen = (f'U({c[1]:g}, {c[2]:g})' if c[0] == 'unif'
               else f'min{{U({c[1]:g}, {c[2]:g}), {c[3]:g}}}')
        fr = 'Gamma(mean 1, var 0.5)' if d['random_effect'] else '--'
        rows.append(f'| {s} | `{A[s]}` | `{Q[s]}` | {cov} | {cen} | {fr} '
                    f'| {d["n_paper"]} |')
    return '\n'.join(rows)


def _competitor_table(tbl, keys):
    rows = ['| Row | Coef | Bias | SE | ESE | CP |', '|---|---|---|---|---|---|']
    for k in keys:
        ent = tbl[k]
        name = ' / '.join(str(x) for x in (k if isinstance(k, tuple) else (k,)))
        first = True
        for coef, v in ent.items():
            rows.append(f'| {name if first else ""} | {coef.replace("beta_","b")} '
                        f'| {_f(v[0])} | {_f(v[1])} | {_f(v[2])} | {_f(v[3])} |')
            first = False
    return '\n'.join(rows)


def _plot_index(plot_dir):
    files = sorted(os.path.relpath(p, plot_dir) for p in
                   glob.glob(os.path.join(plot_dir, '**', '*.png'),
                             recursive=True))
    if not files:
        return '_(no plots generated yet)_'
    rows = ['| Group | Figure | File |', '|---|---|---|']
    for f in files:
        grp = os.path.dirname(f) or '-'
        name = os.path.basename(f)[:-4].replace('_', ' ')
        rows.append(f'| `{grp}` | {name} | `paper_setting/plots/{f}` |')
    return '\n'.join(rows)


TEMPLATE = '''# Paper Simulation Settings — Reproduction with the Paper-Faithful DGP

**Paper:** JMLR-25-1706, *Recurrent Event Analysis with Ordinary Differential
Equations* — §5 of `latex/main.tex`
**Replications:** {reps} per setting
**Generator:** `ode_unify/paper_dgp.py` (`simulate_paper`, `true_rate_paper`)
**Runner:** `ode_unify/paper_setting/run_paper.py`
**Results:** `ode_unify/paper_setting/results/` · **Plots:** `ode_unify/paper_setting/plots/`

Every number in the **local** columns below was recomputed from scratch with a
generator that matches the paper's stated covariate design and censoring
distribution setting by setting. Numbers in the **paper** columns are
transcribed verbatim from Tables 1-3 of `main.tex` (`paper_setting/paper_values.py`).

> **Why a new generator.** `ode_unify/dgp.py` hard-codes one covariate design
> (two N(0,1) clipped at +-1 plus a Bernoulli(0.5)) and defaults to `U(2,4)`
> censoring for *every* setting. That combination is the paper's Setting 3
> only, so summaries produced with it are not comparable to Tables 1-3 — the
> Setting 1 standard error came out ~4.5x too small purely because `U(2,4)`
> yields far more events than the paper's `U(0,2)`. `paper_dgp.py` pins both
> per setting and leaves `dgp.py` untouched.

## 1. The six settings as specified in §5

All settings share
`mu'_x(t) = q(mu_x(t)) * exp(b1*x1 + b2*x2 + b3*x3) * alpha(t)` with
`b1 = b2 = b3 = 1`; settings 5-6 multiply the intensity by a Gamma frailty.

{design}

Settings 5 and 6 reuse Settings 1 and 2's intensity and covariates; the paper
does not restate their censoring windows, so each inherits the matching
non-frailty setting's window.

## 2. Table 1 — specified functional parameters (n=1000)

{t1}

### Competitor rows (paper values, not re-run)

These come from R packages outside the local ODE module and are reproduced here
for context only.

{t1_comp}

## 3. Table 2 — ODE-Flex, both `alpha` and `q` unspecified (n=1000)

{t2}

## 4. Table 3 — Gamma frailty models (Settings 5-6)

{t3}

### Competitor rows (paper values, not re-run)

{t3_comp}

## 5. Agreement summary

`max abs d(Bias)` is the largest absolute difference between the local and
published bias across the two reported coefficients. With {reps} replications the
Monte-Carlo standard error of a bias estimate is roughly `SE/10`, so differences
below about `3*SE/10` are noise. `ESE local/paper` is the mean ratio of the
estimated standard errors.

{agreement}

## 6. Informative censoring (beyond the paper)

Reviewer-requested extension, not present in the current `main.tex`. Each
subject's censoring time is

```
C_i = C0_i * exp(x_i' gamma_c) * xi_i^(c_xi)
```

with `C0_i` drawn from the setting's own paper censoring window. Regime B keeps
censoring dependent on the **observed** covariates (conditional independence
given `x` still holds); regime D makes it depend on the **unobserved** frailty,
which genuinely violates the assumption.

{informative}

## 7. Figures

{plots}

Each `*_alpha.png` / `*_q.png` pair corresponds to one column of the paper's
Figure 3 (settings 1-4) or Figure 4 (settings 5-6); the single-curve plots
correspond to Figure 2. Titles carry the pointwise coverage of the 95% band
against the true curve.

## 9. Findings and deviations

**1. The generator was the whole problem.** Switching from `dgp.py` to
`paper_dgp.py` moved Setting 1's ODE-Cox ESE from 0.0085 (about 5x too small) to
0.0431 against the paper's 0.043, with bias -0.002 matching exactly. Every
specified-functional-parameter estimator (ODE-Cox, ODE-AM, ODE-LT) now agrees
with its published row to within Monte-Carlo error.

**2. Setting 3 ODE-Flex needed the paper's knot placement (`K3`, not `K4`).**
§5.1 places interior knots at the quantiles for `log alpha(t)` only in Setting 3.
Fitting with quantile knots on `q` as well (`K4`) produced a systematic `+0.109`
bias on `b2`; the whole distribution was shifted, not a few outliers (median
1.101). A knot sweep isolated the cause -- `K1` `-0.005`, `K3` `+0.001`,
`K2` `+0.072`, `K4` `+0.096` -- because Setting 3 yields only ~0.7 events per
subject, so the `q`-quantiles cluster in a narrow range. With `K3` the bias falls
to `+0.017` and SE/ESE land on 0.100/0.098 against the paper's 0.097/0.092.

**3. A latent engine bug surfaced under Setting 3's censoring.**
`C = min{{U(2,6), 4}}` puts an atom at 4, so ~52% of subjects share a censoring
time. `_engine/ltm/cox_rec.py` and `_engine/aft/cox_rec.py` passed those tied
times straight to `solve_ivp`, which requires strictly increasing `t_eval`, and
every Setting-3 ODE-Flex fit aborted with `ValueError: Values in t_eval are not
properly sorted`. Both now deduplicate before integrating and expand afterwards
via `unique_sort_index` -- the idiom already used in
`_engine/ltm/objective_func.py` and `_engine/random_effect/ltm/cox_rec.py`. The
fix is exact (the cumulative hazard at a repeated time is the same value) and a
no-op when times are distinct: `sanity_check` still reports **ALL EXACT** across
all seven (estimator, random_effect) combinations at `tol=1e-8`.

**4. The frailty ODE-Flex ESE gap was a variance-computation failure in a few
percent of replications, not a bias.** Two things were wrong.

*(a) Resampling draws were below spec.* `ode_unify.inference` picked B from
`data_setting` (800 / 1000) where §5.2 specifies **B=1500** (Setting 5) and
**B=2000** (Setting 6). `inference()` now takes an optional `resample_B`; the
default is unchanged, so `sanity_check` still reports **ALL EXACT**.

*(b) A few replications returned a wildly inflated SE.* After fixing B the mean
ESE was still ~1.4-1.75x the published value. The distribution showed why: the
**median** SE already matched the paper closely, while the mean was dragged up
by a thin tail reaching 200x the median (`t3_flex_s6_n2000` had median 0.102
against a published 0.110, but a maximum of 20.5).

Diagnosis, on a single replication with the resampling machinery instrumented:

* Raising B does not help -- the SE is stable in B where it is stable at all
  (Setting 5, n=4000: 0.0584 / 0.0583 / 0.0588 at B=1000 / 2000 / 8000).
* A direct finite-difference derivative matrix is always well behaved
  (`cond(A_fd)` 470-4300) and gives sensible SEs on exactly the replications
  where resampling blows up.
* The resampled `A` is far worse conditioned than the truth --
  `cond(A_resample)=6.6e4` against `cond(A_fd)=653` on the same data.
* The blow-ups are *not* caused by outlier draws: `||Y_i||` is well behaved
  (median 1.0, max 1.76, none beyond 10x the median), and trimming them changes
  nothing.

The cause is that `A` has a wide singular-value spectrum -- some spline-coefficient
directions are barely identified -- and the resampling noise floor swamps its
smallest singular values, so `inv(A) V inv(A)` occasionally explodes. `A`
estimates minus the pseudo-likelihood Hessian and is therefore symmetric in the
population, while the OLS estimate is not; projecting it onto the symmetric cone
averages the two independent noisy estimates of each off-diagonal entry and
cures the blow-ups (on the worst case above, SE 0.4705 -> 0.0863, against 0.0911
from finite differences). This is available as
`inference(..., resample_symmetrize=True)`, off by default to preserve exact
reference parity.

*What the reported numbers do.* Rather than change the variance method, the
summariser treats a replication whose reported SE exceeds **5x the median** as a
variance-computation failure. Such replications still contribute to `Bias` and
to the empirical `SE`, which depend only on the point estimates -- those are
perfectly ordinary in the affected replications, and `succ_ind` is 1 for every
one of them -- but are excluded from `ESE` and `CP`, which depend on the failed
standard error. The count is printed in every table (`SE outliers dropped`); it
is 0 for all eleven non-frailty-Flex studies and 3-18 out of 400 (0.75-4.5%) for
the four frailty ODE-Flex studies. With that guard every row of Tables 1-3 lands
within Monte-Carlo error of the published value, ESE ratios included
(0.90-1.07 for the four frailty Flex rows, down from 1.19-1.75).

**5. The ODE-Flex curves were being rescaled at the wrong anchor.** ODE-Flex
identifies `(alpha, q)` only up to `alpha -> c*alpha`, `q -> q/c`, so the fitted
curves must be put on the truth's scale before they can be compared. The solver
pins `alpha_hat(2.0) = 1` (1.5 for the frailty solver), and the first version of
this study normalised there. For Setting 1 that is the worst possible choice:
the censoring is `U(0, 2)`, so `t = 2` sits at the extreme edge of support where
`alpha` is barely identified. Normalising by a value estimated there wrecked the
whole curve -- pointwise coverage **0.065** for `alpha` and **0.000** for `q`,
with a 250% relative bias -- even though the regression coefficients for the very
same fits matched the paper (bias +0.003).

§5.1 instead rescales so that `alpha_hat(t0) = alpha_0(t0)` with `t0` the median
observed event time, which is why every alpha curve in the paper's Figure 3
passes through a single node at `t ~ 0.9` for Setting 1. Rescaling per
replication at the median (and capping the plotting grid at the 95th percentile
of event times, so the figures stop where the data does -- the paper's Setting-1
panel likewise ends at `t ~ 1.8`, not 2.0) fixes it completely:

| Study | curve | ptwise cov before -> after | max rel bias before -> after |
|---|---|---|---|
| `t2_flex_s1` | alpha | 0.065 -> **1.000** | 250% -> **9.6%** |
| `t2_flex_s1` | q | 0.000 -> **1.000** | 68% -> **2.6%** |
| `t2_flex_s2` | alpha | 1.000 -> 1.000 | 27% -> **5.1%** |
| `t3_flex_s5_n2000` | alpha | 0.990 -> 0.998 | 30% -> **5.0%** |
| `t3_flex_s6_n2000` | alpha | 0.998 -> 0.999 | 9.9% -> **5.4%** |

`ode_unify.visual.band_plot` now accepts one scale per estimate (a scalar still
works), since the factor `c` differs from replication to replication.

Every curve -- ODE-Cox and ODE-AM included -- is additionally plotted only on
its **data-supported region**, `SUPPORT[(setting, N)]`: the 2.5th-95th
percentile of observed event times for `alpha(t)`, and the 2.5th-95th percentile
of the true mean function `mu` for `q(u)`. Outside that range the sieve is
extrapolating into a region where almost nobody is still at risk, and both the
estimate and its band degrade sharply; the paper's figures trim the same way.
This alone took `t1_lt_s3` from a 20.0% to a 3.2% maximum relative bias. The
range is fixed in advance from the design, not chosen by looking at which
segments happened to fit well.

Both are display fixes only -- no point estimate or standard error changes.

**6. Coverage sits slightly low across the board** (typically 0.91-0.95 against
the published 0.93-0.96). With 100 replications the standard error of a coverage
estimate is about 0.022, so most gaps are within ~1.5 SE. This is the main reason
to extend to 1000 replications.

**7. Competitor methods are not re-run.** `reReg` (cox.LWYY, am.GL), the
Zeng-Lin NPMLE and `reda` live outside the local ODE module; their published
values are carried through verbatim in the competitor tables above and are
clearly labelled as such. The timing study (paper Figure 1) is likewise not
reproduced, since it is a comparison against those same external methods.

## 10. Reproducing this report

```bash
cd /Users/bomeng/Desktop/research/review/jmlr/code

python3 -m ode_unify.paper_setting.run_paper list
python3 -m ode_unify.paper_setting.run_paper run  --reps 100 --workers 10
python3 -m ode_unify.paper_setting.run_paper plot
python3 -m ode_unify.paper_setting.run_paper report

python3 -m ode_unify.sim_informative_censoring --paper --setting 1 \
    --N 1000 --reps 100 --beta 1 1 1 --gamma_c -0.5 -0.5 -0.5 \
    --c_xi 1.0 --with_frailty \
    --save_dir ode_unify/paper_setting/results/informative

python3 -m ode_unify.paper_setting.make_report
```

Runs are resumable: `_run_one` skips a seed whose `.npz` already exists, so
extending 100 -> 1000 replications only computes the new seeds.
'''


if __name__ == '__main__':
    main()
