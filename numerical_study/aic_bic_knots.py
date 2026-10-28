"""Table S2 of the Supplementary Material: AIC and BIC as rules for choosing the number
of knots.

For each setting, each criterion selects the number of knots separately on every
simulated dataset. The table reports the oracle number of knots k* (smallest mean RMSE,
which requires the truth), the most frequently selected number of knots, and the
relative RMSE: the mean RMSE of the selected fits divided by the RMSE at k*.

The sweeps store BIC = 2 n L + npar log n together with npar, so
AIC = BIC - npar log n + 2 npar is recovered without refitting.

Inputs (bic_model_selection/results/), produced by knot_single.py:
    knot_n1000.csv            settings 9-12, n = 1000, 5 replications
    knot_s13.csv, knot_s14.csv, knot_s15.csv   settings 13-15, n = 2000, 20 replications
Output: tables/table_S2_aic_bic_knots.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'bic_model_selection', 'results')
OUT = os.path.join(HERE, 'tables')
DESC = {9: 'Cox, alpha(t)=1+sqrt(t)+t^2', 10: 'Cox, alpha(t)=1+t',
        11: 'AM, q(u)=1/(1+sqrt(u))', 12: 'AM, q(u)=1/(1+u)',
        13: 'AM, q(u)=1+0.7sin(1.5u)', 14: 'AM, q(u)=1/(1+u)+0.1sin(1.5u)',
        15: 'AM, q(u)={1+0.7sin(1.5u)}/(1+u)'}
SOURCES = [('knot_n1000.csv', 1000, s) for s in (9, 10, 11, 12)] + \
          [('knot_s13.csv', 2000, 13), ('knot_s14.csv', 2000, 14), ('knot_s15.csv', 2000, 15)]


def main():
    rows = []
    for f, n, s in SOURCES:
        g = pd.read_csv(os.path.join(R, f))
        g = g[(g.setting == s) & g.ok].reset_index(drop=True)
        g['AIC'] = g.BIC - g.npar * np.log(n) + 2 * g.npar
        rmse = g.groupby('knots').rmse.mean()
        k_star = int(rmse.idxmin())
        row = dict(setting=s, model=DESC[s], n=n, reps=g.seed.nunique(), k_star=k_star)
        for c in ('AIC', 'BIC'):
            sel = g.loc[g.groupby('seed')[c].idxmin()]
            row[f'k_{c}'] = int(sel.knots.value_counts().idxmax())
            row[f'rel_rmse_{c}'] = sel.rmse.mean() / rmse.min()
        rows.append(row)
    T = pd.DataFrame(rows)
    print(T.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    for c in ('AIC', 'BIC'):
        print(f'{c}: mean relative RMSE {T[f"rel_rmse_{c}"].mean():.2f}, '
              f'worst case {T[f"rel_rmse_{c}"].max():.2f}')
    os.makedirs(OUT, exist_ok=True)
    T.to_csv(os.path.join(OUT, 'table_S2_aic_bic_knots.csv'), index=False)


if __name__ == '__main__':
    main()
