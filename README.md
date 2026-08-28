# RecODE-py

**A unified ODE toolkit for recurrent-event survival models.**

`ode_unify` provides *one* data generator, *one* estimator, and *one* inference
routine covering the whole RecurrentODE model family — Cox, AFT, NPMLE, and
linear-transformation (LTM) models, each with an optional gamma frailty
(random effect). Every model shares the same mean-function ODE

```
mu'_x(t) = xi * q(mu(t)) * exp(x'beta) * alpha(t),      mu(0) = 0,
```

where recurrent events form a non-homogeneous Poisson process with intensity
`mu'_x(t)`, `alpha(t)` is the baseline rate, `q(u)` the transformation, `beta`
the regression coefficients, and `xi` a subject-level frailty.

The functional parameters `alpha(t)` / `q(u)` are estimated by B-spline sieves;
standard errors come either in closed form (Fisher inversion / sandwich) or by
resampling, depending on the model.

---

## Contents

| Path | What it is |
|------|------------|
| `dgp.py` | Data-generating process — the single `simulate()` generator |
| `estimator.py` | Point estimation (`estimate`) returning an `Estimate` object |
| `inference.py` | Standard errors / CIs (`inference`) and the `fit` convenience wrapper |
| `visual.py` | Functional-parameter curves and 95%-band plots |
| `simulate_study.py` | End-to-end Monte-Carlo study CLI (run → save → plot) |
| `sanity_check.py` | Exact-parity checks against the standalone reference modules |
| `_engine/` | Vendored numerical kernels (objectives, MLE, inference); internal |
| `plots/` | Rendered simulation figures (`.png`) |

---

## Installation

```bash
git clone https://github.com/mbmbmmb/RecODE-py.git
```

The package is named `ode_unify`; import it from the directory that *contains*
the clone (i.e. clone as `.../<parent>/ode_unify`), or add that parent to
`PYTHONPATH`.

**Requirements:** Python ≥ 3.9, `numpy`, `scipy`, `matplotlib`.

```bash
pip install numpy scipy matplotlib
```

---

## Quick start

```python
import numpy as np
import ode_unify as U

# 1. Simulate 1000 subjects from preset Cox setting 1  (lambda0(t) = t^2 + 1)
data = U.simulate(N=1000, seed=1, setting=1)

# 2. Estimate + inference in one call
est = U.fit(data, estimator='cox')

print(est.beta)                     # regression coefficients
print(est.se)                       # standard errors
print(est.ci_lower, est.ci_upper)   # 95% Wald CIs

# 3. Plot the fitted baseline rate with its 95% band vs the truth
U.plot_fit(est, 'cox_fit.png',
           grid=np.linspace(0.1, 2.5, 100),
           truth=lambda t: t ** 2 + 1.0)
```

The public API re-exported from `ode_unify/__init__.py` is:

```python
simulate, true_rate, frailty,                # dgp
Estimate, estimate, inference, fit,          # estimation / inference
curve, plot_fit, band_plot, ltm_band_plot    # visual
```

---

## Module reference

### `dgp.py` — data generation

One flexible generator covers every case: closed-form presets, custom
functionals integrated through the ODE, and optional frailty.

#### `simulate(N, seed, setting=None, *, random_effect=False, beta=(1,1,1), rho1=0.5, r1=1.0, alpha=None, q=None, rate=None, frailty_dist='gamma', frailty_params=(2.0, 0.5), censor=(2.0, 4.0), n_grid=200, ode_kw=None)`

Simulate `N` recurrent-event trajectories in long format.

- `setting ∈ {1,2,3,4}` — use a paper preset rate (fast, closed-form).
- Or supply a custom `rate(t, m)` closed form (`m = exp(x'beta)`), **or** custom
  `alpha(t)` / `q(u)` callables (the intensity is obtained by integrating the ODE).
- `random_effect=True` draws `xi` from `frailty_dist`; `False` fixes `xi = 1`.
- `censor=(a, b)` — administrative censoring `~ U(a, b)`.

Returns a `dict` with `x` (n_rows × p), `time`, `delta` (1 = event, 0 = censor),
`id`, and `rho1` / `r1`.

```python
# preset
data = U.simulate(1000, seed=7, setting=2)

# custom functionals with a lognormal frailty
data = U.simulate(500, seed=7,
                  alpha=lambda t: t + 1.0,
                  q=lambda u: 2.0 / (1.0 + u),
                  random_effect=True,
                  frailty_dist='lognormal', frailty_params=(0.5,))
```

#### `true_rate(setting, rho1=0.5)`

Return the analytic intensity `rate(t, m)` for one of the four canonical presets
(Cox, AFT, Box-Cox transformation, general transformation). Useful as the
`truth=` argument when plotting.

#### `frailty(N, rng, random_effect, dist='gamma', params=(2.0, 0.5))`

Draw the length-`N` subject multipliers `xi`. Built-in `dist`: `'gamma'`
`(shape, scale)`, `'lognormal'` `(sigma,)` (mean pinned to 1), `'invgauss'`
`(scale,)`, or any callable `dist(N, rng) -> array`. With `random_effect=False`
returns all-ones **without consuming the RNG stream**.

---

### `estimator.py` — point estimation

#### `estimate(data, *, estimator, random_effect=False, knots=None, seed=0, layout='uniform') -> Estimate`

Fit the model and return **point estimates only** (fast — seconds; no SEs).

- `estimator ∈ {'cox', 'aft', 'npmle', 'ltm'}`.
- `random_effect=True` fits the gamma-frailty version (not available for `npmle`).
- `knots` — sieve knot scheme: `'quantile'` / `'equal'` for AFT/NPMLE,
  `'K1'..'K4'` for LTM. Defaults: AFT `'quantile'`, NPMLE `'equal'`, LTM `'K4'`;
  ignored for Cox.
- `layout` — `'uniform'` (default, C-order everywhere; all results mutually
  consistent) or `'legacy'` (mirrors the standalone per-model pipelines
  bit-for-bit). The two agree to optimizer tolerance (~1e-6).

```python
est = U.estimate(data, estimator='ltm', knots='K4')
print(est.beta)                 # beta[0] pinned to 1.0 for the LTM family
print(est.spline['coefs_alpha'])
```

#### `class Estimate`

Dataclass holding the fit: `beta`, `spline` (knots / orders / spline
coefficients), `estimator`, `random_effect`, `seed`, `runtime`, `success`, and —
once `inference()` runs — `se`, `ci_lower`, `ci_upper`, `se_all`. `raw` mirrors
the per-model result-file schema.

---

### `inference.py` — standard errors & convenience `fit`

#### `inference(est, data, *, seed=None, data_setting=None, spline_se=True) -> Estimate`

Fill in SEs and 95% Wald CIs on a fitted `Estimate` (modified in place). The
method is chosen automatically:

| Model | Method |
|-------|--------|
| Cox / AFT / NPMLE / LTM (no frailty) | closed-form empirical Fisher inversion |
| frailty **Cox** | closed-form sandwich for `beta`; resampling spline SEs if `spline_se=True` |
| frailty **AFT / LTM** | resampling (Zeng & Lin 2008) — computes `beta` and spline SEs together |

`seed` seeds the frailty resampling (defaults to `est.seed`); `data_setting`
only affects the frailty-LTM resampling count. Inference can be *much* slower
than estimation, which is why it is a separate step.

```python
est = U.estimate(data, estimator='cox', random_effect=True)
est = U.inference(est, data, spline_se=True)   # spline_se needed for bands
```

#### `fit(data, *, estimator, random_effect=False, knots=None, ci=True, seed=0, data_setting=None, spline_se=True, layout='uniform') -> Estimate`

One-call convenience: `estimate()` then (if `ci=True`) `inference()`.

```python
est = U.fit(data, estimator='aft', knots='quantile', ci=True)
```

---

### `visual.py` — curves & bands

Works directly on `Estimate` objects (no result files needed).

- **`curve(est, grid, which='auto', scale=1.0)`** — reconstruct a fitted
  functional `exp(B(grid) @ theta)` with its pointwise 95% Wald band. Returns
  `(y, lower, upper)`; bands are `None` if no spline SEs are present. For LTM
  pass `which='alpha'` or `which='q'`.
- **`plot_fit(est, out, *, grid=None, grid_u=None, truth=None, truth_q=None, scale=1.0, scale_q=1.0, title=None)`** —
  one fitted replication vs an optional true curve. One panel for Cox/AFT; two
  panels (`alpha(t)` and `q(u)`) for LTM. Writes `out`, returns the path.
- **`band_plot(estimates, out, *, truth, grid, which='auto', scale=1.0, ...)`** —
  aggregate many replications of the same model: truth, mean/median estimate,
  mean 95% band, plus pointwise & simultaneous coverage. Requires each
  `Estimate` to carry spline SEs.
- **`ltm_band_plot(estimates, out_alpha, out_q, *, truth_alpha, truth_q, grid_t, grid_u, ...)`** —
  the two-panel LTM version of `band_plot`.

Which functional a fit exposes: **Cox** → `alpha(t)`, **AFT** → `q(u)`,
**LTM** → both.

```python
grid = np.linspace(0.1, 2.5, 100)
ests = [U.fit(U.simulate(1000, s, setting=1), estimator='cox') for s in range(1, 21)]
U.band_plot(ests, 'plots/cox_s1.png',
            truth=lambda t: t ** 2 + 1.0, grid=grid,
            ylabel=r'$\lambda_0(t)$')
```

---

### `simulate_study.py` — Monte-Carlo study CLI

Runs the paper's 7 canonical settings end-to-end: simulate → fit + inference →
persist each replication to `results/<slug>/seed<k>.npz` (resumable) → render
band plots into `plots/`.

```bash
# run all settings (100 reps each, 9 worker processes) and plot
python -m ode_unify.simulate_study all  --reps 100 --workers 9

# just (re-)run one setting
python -m ode_unify.simulate_study run  --only cox_setting1

# just (re-)plot from existing results
python -m ode_unify.simulate_study plot --only aft_setting2

# list the available studies
python -m ode_unify.simulate_study list
```

Settings: `cox_setting1`, `aft_setting2`, `npmle_setting3`, `ltm_setting4`,
`re_cox_setting1`, `re_aft_setting2`, `re_ltm_setting1`. Add `--layout uniform`
for the package's uniform layout (default `legacy` reproduces the historical
per-model pipelines bit-for-bit).

The rendered figures live in [`plots/`](plots) — e.g. `cox_s1.png`,
`aft_s2.png`, `ltm_s4_alpha.png`, `re_ltm_s1_q.png`.

---

### `sanity_check.py` — exact-parity check

Verifies that the unified `estimate`/`inference` reproduce the standalone
per-model reference (`RecurrentODE_py`) to machine precision.

```bash
python -m ode_unify.sanity_check                 # all combos
python -m ode_unify.sanity_check --only cox re_cox
```

> **Note:** this script requires the reference package `RecurrentODE_py` on the
> import path. It is a developer verification tool, not needed for normal use.

---

## Project layout

```
ode_unify/
├── __init__.py          # public API re-exports
├── dgp.py               # simulate / true_rate / frailty
├── estimator.py         # estimate / Estimate
├── inference.py         # inference / fit
├── visual.py            # curve / plot_fit / band_plot / ltm_band_plot
├── paper_dgp.py         # paper-faithful generator (true_rate_paper / simulate_paper)
├── simulate_study.py    # Monte-Carlo study CLI
├── sim_informative_censoring.py   # informative-censoring study
├── sanity_check.py      # parity checks
├── _engine/             # internal numerical kernels
│   ├── cox/  aft/  ltm/  npmle/
│   └── random_effect/   # frailty variants
└── numerical_study/     # all numerical work; scripts at the top level
    ├── run_paper.py             # 15-study registry, pooled runner, plots
    ├── run_informative.py       # informative-censoring study (parallel)
    ├── bic_selection.py         # BIC model-selection study
    ├── paper_values.py          # published values, transcribed
    ├── make_report.py           # side-by-side report generator
    ├── plot_informative.py      # informative-censoring figure
    ├── plot_bic_selection.py    # model-selection figure
    ├── simulation_study/        # reproduction of the paper's Section 5,
    │   │                        # split by censoring type
    │   ├── random_censoring/        # C ~ U(a,b), the paper's own settings
    │   │   ├── results/                 # per-replication .npz (git-ignored)
    │   │   ├── cox/  am/  ltm/          # figures, by model family
    │   │   └── random_effect/{cox,am,ltm}/
    │   └── informative_censoring/   # C depending on x, and on the frailty
    │       ├── results/                 # per-regime .npz (git-ignored)
    │       └── informative_censoring.png
    └── bic_model_selection/
        ├── results/                 # selection rates (.csv)
        └── bic_model_selection.png
```

Scripts live directly in `numerical_study/`; each study keeps its own results and
figures beside them, with the simulation study split by censoring type.

`results/` and `numerical_study/results/` (per-replication `.npz` output) are
regenerable and git-ignored, as is `plots/` from `simulate_study.py` (superseded
by `numerical_study/plots/`).

## Reproducing the paper's simulation section

`ode_unify/dgp.py` is the *general* generator: its covariate design is fixed and it
defaults to `U(2,4)` censoring for every setting. `paper_dgp.py` instead pins the
covariate distribution, the censoring window and the frailty per setting, exactly as
specified in §5 of the paper, so the Monte-Carlo summaries are directly comparable to
the published tables.

```bash
python -m ode_unify.numerical_study.run_paper list                     # 15 studies
python -m ode_unify.numerical_study.run_paper run  --reps 1000 --workers 10
python -m ode_unify.numerical_study.run_paper plot
python -m ode_unify.numerical_study.run_paper report                   # Bias/SE/ESE/CP
python -m ode_unify.numerical_study.make_report                        # full markdown report
```

Runs are resumable: a seed whose `.npz` already exists is skipped, so extending
100 → 1000 replications only computes the new seeds. The runner pools every
`(study, seed)` task into one process pool, dispatched longest-job-first, so workers
stay busy through the tail of each study; pass `--sequential` for the older
one-pool-per-study behaviour.

### Informative censoring

```bash
python -m ode_unify.sim_informative_censoring --paper --setting 1 \
    --N 1000 --reps 100 --beta 1 1 1 --gamma_c -0.5 -0.5 -0.5 \
    --c_xi 1.0 --with_frailty --save_dir ode_unify/numerical_study/results/informative
python -m ode_unify.numerical_study.plot_informative
```

Each subject's censoring time is `C_i = C0_i · exp(x_i'γ_c) · ξ_i^{c_ξ}`. With
`γ_c ≠ 0` and `c_ξ = 0` the censoring depends on the *observed* covariates, so
conditional independence given `x` still holds and the estimator is unaffected. With
`c_ξ ≠ 0` it depends on the *unobserved* frailty, which genuinely violates the
assumption. `plot_informative.py` renders the former group only.
