"""Export the aggregate results of the MIMIC-III analysis to results/.

Only summaries are written (no patient-level records), so that the files can be shared
under the PhysioNet data use agreement:

    coefficients_<tag>.csv        Table 9: ODE-Cox and ODE-Flex coefficients, SEs, p-values
    functional_<tag>.csv          Figure 5: log alpha(t) and log q(u) with pointwise SEs
    lr_test_<tag>.csv             likelihood ratio statistic and its bootstrap calibration
    bootstrap_lr_<tag>.csv        the bootstrap statistics
    cohort_<tag>.csv              cohort size, events and knot settings
    scaler_params_<tag>.csv       centring and scaling of each covariate

Usage:  python export_results.py [tag]
"""
import os
import sys

import numpy as np
import pandas as pd

from config import OUT, RESULTS

LABELS = {
    'bun_mean': 'BUN', 'age': 'Age', 'heartrate_mean': 'Heart rate',
    'sysbp_mean': 'Systolic blood pressure', 'tempc_mean': 'Body temperature',
    'PaO2FiO2_vent_min': 'PaO2/FiO2', 'urineoutput': 'Urine output', 'wbc_mean': 'WBC',
    'potassium_mean': 'Potassium', 'sodium_mean': 'Sodium', 'bicarbonate_mean': 'Bicarbonate',
    'bilirubin_mean': 'Bilirubin', 'gcs_min': 'GCS', 'AIDS': 'AIDS',
    'HEM': 'Haematologic malignancy', 'METS': 'Metastatic cancer',
    'M_PaO2FiO2_vent_min': 'PaO2/FiO2 not measured', 'M_bilirubin_mean': 'Bilirubin not measured',
    'Adm_type_scheduled': 'Scheduled surgical', 'Adm_type_unscheduled': 'Unscheduled surgical',
}


def main(tag='2y'):
    os.makedirs(RESULTS, exist_ok=True)
    z = np.load(os.path.join(OUT, f'final_{tag}.npz'), allow_pickle=True)

    coef = pd.read_csv(os.path.join(OUT, f'final_{tag}_coef.csv'))
    coef.insert(1, 'covariate', coef.feature.map(LABELS))
    coef['cox_ci_lower'] = coef.cox_est - 1.959964 * coef.cox_se
    coef['cox_ci_upper'] = coef.cox_est + 1.959964 * coef.cox_se
    coef = coef[['feature', 'covariate', 'cox_est', 'cox_se', 'cox_ci_lower', 'cox_ci_upper',
                 'cox_p', 'flex_est', 'flex_se', 'flex_p']].sort_values('cox_p')
    coef.to_csv(os.path.join(RESULTS, f'coefficients_{tag}.csv'), index=False)

    pd.DataFrame(dict(t=z['gt'], log_alpha_flex=z['log_alpha'], se_log_alpha_flex=z['se_alpha'],
                      log_alpha_cox=z['log_alpha_cox'])).to_csv(
        os.path.join(RESULTS, f'functional_alpha_{tag}.csv'), index=False)
    pd.DataFrame(dict(u=z['gu'], log_q_flex=z['log_q'], se_log_q_flex=z['se_q'])).to_csv(
        os.path.join(RESULTS, f'functional_q_{tag}.csv'), index=False)

    lr = float(z['LR'])
    row = dict(LR=lr, negll_cox=float(z['negll_cox']), negll_flex=float(z['negll_flex']))
    boot_path = os.path.join(OUT, f'final_{tag}_bootstrap.csv')
    if os.path.exists(boot_path):
        b = pd.read_csv(boot_path)
        ok = b.LR.notna()
        b.loc[ok, ['b', 'LR']].to_csv(os.path.join(RESULTS, f'bootstrap_lr_{tag}.csv'), index=False)
        row.update(B=int(ok.sum()), boot_median=b.LR[ok].median(), boot_q99=b.LR[ok].quantile(0.99),
                   boot_max=b.LR[ok].max(),
                   bootstrap_p=(1 + (b.LR[ok] >= lr).sum()) / (1 + ok.sum()))
    pd.DataFrame([row]).to_csv(os.path.join(RESULTS, f'lr_test_{tag}.csv'), index=False)

    mu_c = np.asarray(z['mu_c'], float)
    pd.DataFrame([dict(patients=int(z['m']), readmissions=int(z['events']),
                       horizon_days=float(z['horizon']), spline_order=int(z['k']),
                       intervals=int(np.ceil(int(z['m']) ** (1 / 7))),
                       anchor_rescaling_c0=float(z['c0']), alpha_anchor_day=float(z['t_anchor']),
                       band_draws=int(z['B']), band_seed=int(z['band_seed']) if 'band_seed' in z else 0,
                       mu_c_q95=float(np.quantile(mu_c, 0.95)))]).to_csv(
        os.path.join(RESULTS, f'cohort_{tag}.csv'), index=False)

    sp = os.path.join(OUT, f'scaler_params_{tag}.csv')
    if os.path.exists(sp):
        pd.read_csv(sp).to_csv(os.path.join(RESULTS, f'scaler_params_{tag}.csv'), index=False)
    print('wrote', sorted(f for f in os.listdir(RESULTS) if tag in f))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '2y')
