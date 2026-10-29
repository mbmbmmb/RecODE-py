# RecODE-py

**A unified ODE toolkit for recurrent-event survival models.**

This repository contains the Python implementation of the methods in

> Bo Meng, Weijing Tang, Gongjun Xu and Ji Zhu. *Recurrent Event Analysis with Ordinary
> Differential Equations.* Submitted to the Journal of Machine Learning Research.

together with the code and saved results that reproduce every simulation table and
figure of the paper and its Supplementary Material, and the real-data analysis of
MIMIC-III (Section 6). See [Reproducing the paper](#reproducing-the-paper).

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
| `numerical_study/` | Simulation study, informative censoring and BIC model selection, with their figures |
| `tests/` | Regression tests for the solver cascade (`python tests/test_solve_sieve_step.py`) |

---

## Installation

```bash
git clone https://github.com/mbmbmmb/RecODE-py.git
```

The package is named `ode_unify`; import it from the directory that *contains*
the clone (i.e. clone as `.../<parent>/ode_unify`), or add that parent to
`PYTHONPATH`.

**Requirements:** Python ≥ 3.9 and the packages in `requirements.txt`
(`numpy`, `scipy`, `pandas`, `matplotlib`).

```bash
pip install -r requirements.txt
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

`simulate_study.py` writes its figures to a local `plots/` directory, which is
git-ignored: it predates and is superseded by
[`numerical_study/`](numerical_study), whose figures are tracked.

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
├── paper_dgp.py         # generator for the settings of the paper (simulate_paper)
├── simulate_study.py    # Monte-Carlo study CLI
├── sim_informative_censoring.py   # informative-censoring generator
├── sanity_check.py      # parity checks
├── _engine/             # numerical kernels (objectives, ODE solves, inference)
│   ├── cox/  aft/  ltm/  npmle/
│   └── random_effect/   # frailty variants
├── tests/               # solver-cascade regression tests (no pytest required)
├── numerical_study/     # simulation studies of Section 5 and Supplement A
│   ├── tables/                  # every simulation table, rebuilt by make_tables.py
│   ├── simulation_study/        # Sections 5.1-5.3 (figures; .npz results git-ignored)
│   └── bic_model_selection/     # Sections 5.4-5.5 and Supplement A
│       ├── results/                 # saved results behind the tables (.csv, .npz)
│       ├── knot_curves_all_settings.png   # Figure S1
│       └── knot_placement.png             # Figure S2
└── mimic_study/         # real-data analysis of Section 6 (see mimic_study/README.md)
    └── results/                 # aggregate results and Figure 5
```

## Reproducing the paper

All commands are run from the folder that contains the script. Saved results are
included for every table, so `make_tables.py` rebuilds the tables without refitting;
the commands below regenerate the saved results themselves.

### Simulation studies (`numerical_study/`)

| Paper | Script | Saved results |
|---|---|---|
| Tables 2-4, Figures 1-4 (Sections 5.1-5.2) | `run_paper.py run --reps 1000`, then `run_paper.py report` / `plot` | `simulation_study/random_censoring/results/` (git-ignored, regenerable) |
| Table 5 (Section 5.3) | `run_informative.py --setting 1 2 3 4 --only random cov cov_decr --reps 1000` | `simulation_study/informative_censoring/results/` (git-ignored); `tables/table_5_*.csv` |
| Table 6 (Section 5.4), Supplement A.4 | `lr_bootstrap.py --reps 300 --grid 1:1000 3:1000 7:1000 --out bic_model_selection/results/lr_bootstrap.csv`; `lr_bootstrap.py --reps 200 --grid 1:2000 7:2000 1:4000 7:4000 --out bic_model_selection/results/lr_settings_table.csv` | `lr_bootstrap.csv`, `lr_settings_table.csv` |
| Table 7 (Section 5.5) | `bic_selection_quantile.py --reps 100 --grid 1:2000 2:2000 3:2000 7:2000 --out bic_model_selection/results/bic_selection_q17_n2000.csv` | `bic_selection_q17_n2000*.csv` |
| Supplement A.3 (equally spaced knots) | `bic_selection.py --reps 100 --grid 1:2000 2:2000 3:2000 7:2000 --out bic_model_selection/results/bic_selection_n2000.csv` | `bic_selection_n2000*.csv` |
| Table S1, Figure S1 | `knot_single.py --reps 20 --n 2000 --settings 9 10 11 12 --tag single`; `knot_single.py --reps 20 --n 2000 --settings 11 12 --tag single_noanchor`; `knot_setting4.py --reps 30 --n 2000 --tag s4_fixed`; then `plot_knot_sensitivity.py` | `knot_single*.{csv,npz}`, `knot_s4_fixed*.{csv,npz}` |
| Table S2 | `knot_single.py --reps 5 --n 1000 --settings 9 10 11 12 --tag n1000`; `knot_single.py --reps 20 --n 2000 --settings 13 --tag s13` (likewise 14, 15); then `aic_bic_knots.py` | `knot_n1000.csv`, `knot_s13.csv`, `knot_s14.csv`, `knot_s15.csv` |
| Tables S3-S4, Figure S2 | `knot_placement.py --reps 40 --n 1000 --p P --out bic_model_selection/results/place_pP.csv` for P in 0.10 0.14 0.20 0.25 0.30; then `bic_model_selection/plot_knot_placement.py` | `place_p*.csv` |

```bash
cd numerical_study
python make_tables.py       # Tables 5-7, S1, S3-S4 and the Supplement A.3/A.4 numbers -> tables/
python aic_bic_knots.py     # Table S2 -> tables/
```

The settings are defined in `paper_dgp.py`: Settings 1-6 are those of Section 5,
Setting 7 is the general model with `alpha(t) = t + 1` and `q(u) = 1 + 0.7 sin(1.5u)`
used for the power study and model selection, and Settings 9-15 are the single-function
settings of the knot study. All scripts accept `--workers` for parallel runs.

### Real-data analysis (`mimic_study/`)

The MIMIC-III data require credentialed access through PhysioNet, so the repository
contains the full processing and fitting pipeline and the aggregate results, but no
patient-level data. See [`mimic_study/README.md`](mimic_study/README.md).

## Citation

If you use this code, please cite the paper above.
