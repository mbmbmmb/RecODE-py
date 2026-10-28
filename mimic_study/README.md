# MIMIC-III analysis (Section 6 of the paper)

This folder reproduces the real-data analysis of the paper: the recurrence of ICU
readmissions in MIMIC-III, modelled by the Cox-type model (ODE-Cox, `q ≡ 1`) and the
general model (ODE-Flex, `α` and `q` unspecified), with the likelihood ratio test of
`H0: q ≡ 1` calibrated by the parametric bootstrap.

## Data access

MIMIC-III is available only to credentialed users who have signed the PhysioNet data
use agreement (<https://physionet.org/content/mimiciii/1.4/>). For this reason the
repository contains **no patient-level data**: all intermediate files are written to
`output/`, which is excluded from version control, and `results/` contains only
aggregate results.

The scripts read two inputs from the folder given by the environment variable
`MIMIC_DIR` (default: `mimic_study/data/`, also excluded from version control):

```
$MIMIC_DIR/
├── processed-data/
│   ├── input.csv        # 26 summary statistics of the 17 SAPS-II features, first 24 h
│   └── valid_aids.csv   # HADM_ID of each row of input.csv
└── raw-data/
    └── ADMISSIONS.csv   # ADMISSIONS table of MIMIC-III v1.4
```

`input.csv` and `valid_aids.csv` are produced by the preprocessing of Purushotham et
al. (2018), <https://github.com/USC-Melady/Benchmarking_DL_MIMICIII>
(`preprocessing/steps/get_avg_17_features_raw.py`); they contain the 35,643 adult
patients alive 24 hours after their first ICU admission.

## Pipeline

```bash
export MIMIC_DIR=/path/to/mimic/data
bash run_all.sh            # or run the steps below one at a time
```

| Step | Script | Output (in `output/` unless stated) |
|---|---|---|
| 1. Cohort cleaning: first ICU admission, age cap, clinical bounds, missingness indicators | `clean_cohort.py` | `cohort_clean.csv` |
| 2. Long format with the two-year administrative window, `C = min(730 days, death)` | `build_long_format.py --horizon-days 730 --tag 2y` | `cohort_long_2y.{csv,npz}` |
| 3. Standardization (median/IQR for skewed variables, mean/SD otherwise), median imputation | `scale_features.py --tag 2y` | `cohort_long_scaled_2y.npz`, `scaler_params_2y.csv` |
| 4. ODE-Cox and ODE-Flex fits, likelihood ratio statistic, coefficient table | `fit_final_horizon.py 2y` | `final_2y.npz`, `final_2y_coef.csv` |
| 5. Resampling standard errors and pointwise bands (B = 8000, seed 1) | `resample_bands.py 2y 8000 1` | updates `final_2y.npz`, `final_2y_coef.csv` |
| 6. Parametric bootstrap of the likelihood ratio test (B = 200) | `bootstrap_final_lr.py 2y 200 10` | `final_2y_bootstrap.csv` |
| 7. Figure 5 | `plot_final_horizon.py 2y` | `results/mimic_2y_functional.png` |
| 8. Aggregate results | `export_results.py 2y` | `results/*.csv` |

On a 10-core workstation, steps 1-3 take about a minute, step 4 about 40 minutes and
step 6 about two hours.

Model specification (Section 6.2 of the paper): 20 covariates (the averages of the
maximum and minimum of each vital sign and laboratory value, single-valued variables,
comorbidities, admission-type indicators, and missingness indicators for PaO2/FiO2 and
bilirubin); quadratic B-splines for `log α` and `log q` with `⌈m^{1/7}⌉ = 5` intervals
and knots at the sample quantiles; ODE-Cox with all coefficients free and sandwich
standard errors; ODE-Flex with the BUN covariate rescaled by its ODE-Cox estimate and
`log α` fixed at its ODE-Cox value at the median readmission time, so that ODE-Cox is
nested in ODE-Flex and the likelihood ratio statistic is non-negative.

## Results (`results/`)

| File | Content |
|---|---|
| `coefficients_2y.csv` | Table 9: estimates, standard errors, 95% intervals and p-values under both models |
| `functional_alpha_2y.csv`, `functional_q_2y.csv` | Figure 5: `log α(t)` and `log q(u)` with pointwise standard errors |
| `lr_test_2y.csv`, `bootstrap_lr_2y.csv` | LR = 55.41; bootstrap null median 5.62, maximum 18.28; p ≤ 1/201 |
| `cohort_2y.csv` | 35,626 patients, 6,949 readmissions, spline and anchor settings |
| `scaler_params_2y.csv` | centring and scaling of each covariate |
| `mimic_2y_functional.png` | Figure 5 |
