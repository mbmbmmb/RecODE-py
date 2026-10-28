"""Rebuild the simulation tables of the revised paper and its Supplementary Material from
the saved results, and write them to tables/.

    Table 5   covariate-dependent censoring          simulation_study/informative_censoring/results/*.npz
    Table 6   size and power of the LR test          bic_model_selection/results/lr_bootstrap.csv,
                                                     lr_settings_table.csv
    Table 7   model selection by BIC                 bic_model_selection/results/bic_selection_q17_n2000.csv
    Table S1  sensitivity to the knot exponent p     knot_single.csv, knot_single_noanchor.csv,
                                                     knot_s4_fixed.csv
    Table S3  knot placement (median RMSE)           place_p*.csv
    Table S4  BIC across placement and knot number   place_p*.csv
    Table S5  censoring driven by a frailty          informative_censoring/results/*.npz
    Sec. D.3  BIC selection with equally spaced knots  bic_selection_n2000.csv
    Sec. D.4  power under Setting 3                  lr_bootstrap.csv

Table S2 is produced by aic_bic_knots.py. The informative-censoring .npz files are not
stored in the repository (regenerate them with run_informative.py); their summaries are
kept in tables/.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, wilcoxon

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


def table5_and_S5():
    if not os.path.isdir(INF):
        print('\n(skipping Tables 5 and S5: run run_informative.py to create', INF, ')')
        return
    rows = []
    for s, model in [(1, 'ODE-Cox'), (2, 'ODE-AM'), (3, 'ODE-LT'), (4, 'ODE-Flex')]:
        pre = '' if s == 1 else f's{s}_'
        for k, regime in [(0, 'random'), (0.5, 'cov'), (1, 'cov_decr')]:
            rows.append(dict(setting=s, model=model, k=k,
                             **censoring_row(os.path.join(INF, f'{pre}{regime}.npz'))))
    save(pd.DataFrame(rows), 'table_5_covariate_censoring.csv')
    rows = []
    for regime, frail, k, cxi in [('random', 'No', 0, 0), ('cov', 'No', 0.5, 0),
                                  ('random_fr', 'Yes', 0, 0), ('frailty_inf', 'Yes', 0.5, 1)]:
        rows.append(dict(frailty_in_event_process=frail, k=k, c_xi=cxi,
                         **censoring_row(os.path.join(INF, f'{regime}.npz'))))
    save(pd.DataFrame(rows), 'table_S5_frailty_censoring.csv')


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


def table6():
    d = pd.concat([pd.read_csv(os.path.join(R, 'lr_bootstrap.csv')),
                   pd.read_csv(os.path.join(R, 'lr_settings_table.csv'))], ignore_index=True)
    save(rejection_rates(d), 'table_6_lr_test_and_D4.csv')


def table7():
    for f, name in [('bic_selection_q17_n2000.csv', 'table_7_bic_selection.csv'),
                    ('bic_selection_n2000.csv', 'sec_D3_bic_selection_equal_knots.csv')]:
        d = pd.read_csv(os.path.join(R, f))
        d = d[d.crit == 'BIC'][['setting', 'n', 'truth', 'n_ok', 'Cox', 'AM', 'LT', 'Flex', 'correct']]
        save(d, name)
        print(f'mean accuracy: {d.correct.mean():.3f}')


def tableS1():
    single = pd.read_csv(os.path.join(R, 'knot_single.csv'))
    noanch = pd.read_csv(os.path.join(R, 'knot_single_noanchor.csv'))
    flex = pd.read_csv(os.path.join(R, 'knot_s4_fixed.csv'))
    cols = {}
    for s, name, d in [(9, 'Cox_alpha_complex', single), (10, 'Cox_alpha_simple', single),
                       (11, 'AM_q_complex', noanch), (12, 'AM_q_simple', noanch)]:
        m = d[(d.setting == s) & d.ok].groupby('p').rmse.mean()
        cols[name] = m / m.min()
    f = flex[flex.ok & (flex.p >= 0.10)]
    for c, name in [('rmse_alpha', 'Flex_alpha'), ('rmse_q', 'Flex_q')]:
        m = f.groupby('p')[c].mean()
        cols[name] = m / m.min()
    T = pd.DataFrame(cols).reset_index()
    save(T, 'table_S1_knot_exponent.csv')


def placement():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, 'place_p*.csv')))],
                  ignore_index=True)
    d = d[d.kind.isin(['equal', 'quantile']) & d.ok].reset_index(drop=True)
    n = 1000
    d['BIC'] = 2 * n * d.negll + np.log(n) * (d.lq + 2)
    save(d.groupby(['kind', 'lq']).rmse_q.median().unstack().reset_index(),
         'table_S3_knot_placement_rmse.csv')
    d['dBIC'] = d.BIC - d.seed.map(d.groupby('seed').BIC.min())
    sel = d.loc[d.groupby('seed').BIC.idxmin()].groupby(['kind', 'lq']).size()
    g = d.groupby(['kind', 'lq']).dBIC.agg(['mean', 'std', 'count']).reset_index()
    g['se'] = g['std'] / np.sqrt(g['count'])
    g['selected'] = [int(sel.get((k, l), 0)) for k, l in zip(g.kind, g.lq)]
    save(g[['kind', 'lq', 'mean', 'se', 'selected']], 'table_S4_knot_placement_bic.csv')
    a = d[(d.kind == 'quantile') & (d.lq == 14)].set_index('seed')
    b = d[(d.kind == 'quantile') & (d.lq == 6)].set_index('seed')
    diff = b.BIC - a.BIC
    print(f'BIC(6 knots) - BIC(14 knots): {diff.mean():.1f} (se {diff.std() / np.sqrt(len(diff)):.1f}); '
          f'Wilcoxon test of RMSE, 14 vs 6 knots: p = {wilcoxon(a.rmse_q, b.rmse_q).pvalue:.2f}')


if __name__ == '__main__':
    table5_and_S5()
    table6()
    table7()
    tableS1()
    placement()
