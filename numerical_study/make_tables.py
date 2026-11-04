"""Rebuild the simulation tables of the revised paper and its Supplementary Material from
the saved results, and write them to tables/.

    Table 5   size and power of the LR test          bic_model_selection/results/lr_bootstrap.csv,
                                                     lr_settings_table.csv
    Table S3  number and placement of the knots      knot_s13.csv, knot_s14.csv, knot_s15.csv,
              (RMSE of log q), and the BIC           knot_s13_15_equal.csv
              comparison of placements in Sec. A.1
    Table S5  model selection by BIC                 bic_model_selection/results/bic_selection_q17_n2000.csv
    Sec. A.3  BIC selection with equally spaced knots  bic_selection_n2000.csv
    Sec. A.4  power under Setting 3                  lr_bootstrap.csv
    Table S6  covariate-dependent censoring          simulation_study/informative_censoring/results/*.npz
    Tables S7-S8  censoring driven by a frailty      simulation_study/informative_censoring/results/*.npz

Tables 2-4 are produced by run_paper.py report, Tables S1-S2 by knot_beta_sensitivity.py
and Table S4 by aic_bic_knots.py.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'bic_model_selection', 'results')
INF = os.path.join(HERE, 'simulation_study', 'informative_censoring', 'results')
OUT = os.path.join(HERE, 'tables')
LEVELS = (0.10, 0.05, 0.01)


def save(df, name):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f'\n== {name}\n' + df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


def censoring_row(path):
    z = np.load(path)
    B, S, t = z['beta'], z['se'], z['truth']
    free = np.isfinite(B).any(0)                        # ODE-Flex fixes beta_1 = 1
    B, S, t = B[:, free], S[:, free], t[free]
    ok = np.isfinite(B).all(1) & np.isfinite(S).all(1)
    B, S = B[ok], S[ok]
    return dict(bias=np.abs(B.mean(0) - t).max(), SE=B.std(0, ddof=1).mean(),
                ESE=S.mean(), CP=(np.abs(B - t) <= 1.959964 * S).mean(), reps=int(ok.sum()))


def covariate_censoring():
    if not os.path.isdir(INF):
        print('\n(skipping Table S6: run run_informative.py to create', INF, ')')
        return
    rows = []
    for s, model in [(1, 'ODE-Cox'), (2, 'ODE-AM'), (3, 'ODE-LT'), (4, 'ODE-Flex')]:
        pre = '' if s == 1 else f's{s}_'
        for k, regime in [(0, 'random'), (0.5, 'cov')]:
            rows.append(dict(setting=s, model=model, k=k,
                             **censoring_row(os.path.join(INF, f'{pre}{regime}.npz'))))
    save(pd.DataFrame(rows), 'table_S6_covariate_censoring.csv')


def frailty_censoring():
    """Tables S7-S8: Setting 1 with censoring C = C0*exp(-k x'beta)*xi^c (Supplement A.6)."""
    if not os.path.isdir(INF):
        print('\n(skipping the frailty-censoring tables: run run_informative.py to create', INF, ')')
        return
    rows = [dict(frailty='No', k=0, c=0, **censoring_row(os.path.join(INF, 'random.npz'))),
            dict(frailty='No', k=0.5, c=0, **censoring_row(os.path.join(INF, 'cov.npz'))),
            dict(frailty='Yes', k=0, c=0, **censoring_row(os.path.join(INF, 'random_fr.npz')))]
    for f in sorted(glob.glob(os.path.join(INF, 's1_fr_k*_c*.npz'))):
        k, c = os.path.basename(f)[len('s1_fr_k'):-len('.npz')].split('_c')
        rows.append(dict(frailty='Yes', k=float(k), c=float(c), **censoring_row(f)))
    T = pd.DataFrame(rows).sort_values(['frailty', 'k', 'c'], kind='stable')
    save(T, 'table_S7_S8_frailty_censoring.csv')


def rejection_rates(df):
    rows = []
    for (s, n), sub in df.dropna().groupby(['setting', 'n']):
        null = sub.lr_boot.values                       # pooled warp-speed null draws
        pv = np.array([(1 + (null >= o).sum()) / (len(null) + 1) for o in sub.lr_obs])
        row = dict(setting=int(s), n=int(n), reps=len(sub))
        for lv in LEVELS:
            x, m = int((pv <= lv).sum()), len(pv)
            lo = beta_dist.ppf(0.025, x, m - x + 1) if x > 0 else 0.0
            hi = beta_dist.ppf(0.975, x + 1, m - x) if x < m else 1.0
            row.update({f'rej_{lv}': x / m, f'lo_{lv}': lo, f'hi_{lv}': hi})
        rows.append(row)
    return pd.DataFrame(rows)


def lr_test():
    d = pd.concat([pd.read_csv(os.path.join(R, 'lr_bootstrap.csv')),
                   pd.read_csv(os.path.join(R, 'lr_settings_table.csv'))], ignore_index=True)
    save(rejection_rates(d), 'table_5_lr_test_and_A4.csv')


def bic_selection():
    for f, name in [('bic_selection_q17_n2000.csv', 'table_S5_bic_selection.csv'),
                    ('bic_selection_n2000.csv', 'sec_A3_bic_selection_equal_knots.csv')]:
        d = pd.read_csv(os.path.join(R, f))
        d = d[d.crit == 'BIC'][['setting', 'n', 'truth', 'n_ok', 'Cox', 'AM', 'LT', 'Flex', 'correct']]
        save(d, name)
        print(f'mean accuracy: {d.correct.mean():.3f}')


def knot_functional():
    """Table S3: RMSE of log q-hat by number and placement of the knots in the three
    settings with an oscillating q (ODE-AM, n = 2000, 20 replications), and BIC over all
    combinations of placement and number of knots on each dataset."""
    q = pd.concat([pd.read_csv(os.path.join(R, f'knot_s{s}.csv')) for s in (13, 14, 15)])
    q['placement'] = 'quantile'
    e = pd.read_csv(os.path.join(R, 'knot_s13_15_equal.csv'))
    d = pd.concat([q, e], ignore_index=True)
    d = d[d.ok].reset_index(drop=True)
    T = d.groupby(['setting', 'p', 'placement']).agg(knots=('knots', 'median'),
                                                     rmse=('rmse', 'mean')).unstack()
    T.columns = [f'{a}_{b}' for a, b in T.columns]
    T = T.reset_index().rename(columns={'knots_quantile': 'knots'}).drop(columns='knots_equal')
    save(T, 'table_S3_knots_functional.csv')
    rows = []
    for s, g in d.groupby('setting'):
        sel = g.loc[g.groupby('seed').BIC.idxmin()]
        w = g.pivot_table(index=['seed', 'p'], columns='placement', values='BIC')
        rows.append(dict(setting=s, datasets=g.seed.nunique(),
                         bic_selects_quantile=int((sel.placement == 'quantile').sum()),
                         mean_bic_equal_minus_quantile=float((w['equal'] - w['quantile']).mean()),
                         min_over_p=float((w['equal'] - w['quantile']).groupby('p').mean().min())))
    save(pd.DataFrame(rows), 'sec_A1_bic_placement.csv')


if __name__ == '__main__':
    covariate_censoring()
    frailty_censoring()
    lr_test()
    bic_selection()
    knot_functional()
