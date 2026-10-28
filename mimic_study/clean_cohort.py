"""Cohort cleaning for the recurrent-event readmission model.

Steps applied (each prints what it changed):
  1. Cap age at 90 (MIMIC adds ~300 yr to DOB for patients >=89).
  2. Clip X to clinically plausible bounds (sentinels / data-entry errors).
  3. Add missingness indicators for the high-missingness features
     (PaO2FiO2_vent_min, bilirubin_min, bilirubin_max).
  4. Build per-patient recurrent-event outcome from ALL admissions
     in ADMISSIONS.csv (n_events = readmissions, follow-up days,
     death-at-end flag).
  5. Drop the 4 patients with negative patient-level follow-up
     (last DISCHTIME / DEATHTIME earlier than first ADMITTIME).
  6. Flag patients whose indexed admit is NOT their first hospital
     admit (column ``is_first_hospital_admit``); kept in the cohort
     but exposed for downstream filtering.
  7. Write <OUT>/cohort_clean.csv (one row per patient).

Inputs and output folders are set in config.py.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import OUT, RAW, SRC

V_NAMES = [
    'age', 'heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
    'tempc_max', 'tempc_min', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_min', 'bun_max', 'wbc_min', 'wbc_max', 'potassium_min',
    'potassium_max', 'sodium_min', 'sodium_max', 'bicarbonate_min',
    'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'gcs_min',
    'AIDS', 'HEM', 'METS', 'Adm_type',
]

CLIP = {
    'PaO2FiO2_vent_min': (50, 600),
    'urineoutput':       (0, 20000),
    'tempc_min':         (25, 45),
    'tempc_max':         (25, 45),
    'sysbp_min':         (30, 300),
    'sysbp_max':         (30, 300),
    'heartrate_min':     (20, 250),
    'heartrate_max':     (20, 250),
    'bun_min':           (0, 300),
    'bun_max':           (0, 300),
    'wbc_min':           (0, 500),
    'wbc_max':           (0, 500),
    'potassium_min':     (1, 10),
    'potassium_max':     (1, 10),
    'sodium_min':        (100, 180),
    'sodium_max':        (100, 180),
    'bicarbonate_min':   (2, 60),
    'bicarbonate_max':   (2, 60),
    'bilirubin_min':     (0, 100),
    'bilirubin_max':     (0, 100),
    'gcs_min':           (3, 15),
}
# Lab pairs whose post-clip min/max correlation > 0.8: collapse each
# to (mean, spread) = ((max+min)/2, max-min) to remove redundancy.
MEAN_SPREAD_PAIRS = ['tempc', 'bilirubin', 'bun', 'sodium',
                     'bicarbonate', 'wbc', 'heartrate', 'sysbp',
                     'potassium']
# After the mean/spread collapse, only one missingness indicator is
# needed per lab pair (min and max share the same NaN pattern).
MISS_IND = ['PaO2FiO2_vent_min', 'bilirubin_mean']

# Final continuous + categorical features written to cohort_clean.csv.
V_NAMES_FINAL = [
    'age', 'heartrate_mean', 'heartrate_spread',
    'sysbp_mean', 'sysbp_spread',
    'tempc_mean', 'tempc_spread', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_mean', 'bun_spread', 'wbc_mean', 'wbc_spread',
    'potassium_mean', 'potassium_spread', 'sodium_mean', 'sodium_spread',
    'bicarbonate_mean', 'bicarbonate_spread', 'bilirubin_mean',
    'bilirubin_spread', 'gcs_min', 'AIDS', 'HEM', 'METS', 'Adm_type',
]


def main():
    os.makedirs(OUT, exist_ok=True)
    print('=== Loading ===')
    inp = pd.read_csv(os.path.join(SRC, 'input.csv'), header=None,
                      names=V_NAMES)
    aids = pd.read_csv(os.path.join(SRC, 'valid_aids.csv'), header=None,
                       names=['HADM_ID']).astype({'HADM_ID': 'int64'})
    adm = pd.read_csv(os.path.join(RAW, 'ADMISSIONS.csv'),
                      parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'],
                      usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                               'DISCHTIME', 'DEATHTIME',
                               'HOSPITAL_EXPIRE_FLAG'])

    df = pd.concat([aids, inp], axis=1)
    df = df.merge(adm, on='HADM_ID', how='left')
    n0 = len(df)
    print(f'rows / patients: {n0}')

    # ---- 1. Age cap --------------------------------------------------
    # Marginal hazard above age 90 is small in this cohort, so we
    # collapse the de-identified ~300 yr offset to a hard 90 cap and do
    # NOT carry a separate `age_capped` indicator (the cap info is
    # implicit in age itself).
    print('\n=== 1. Cap age at 90 ===')
    n_high = int((df['age'] > 90).sum())
    print(f'rows with age > 90 (MIMIC de-identified ~300 yr offset): {n_high}')
    df.loc[df['age'] > 90, 'age'] = 90
    print(f'after cap: age range = [{df["age"].min():.2f}, '
          f'{df["age"].max():.2f}]')

    # ---- 2. Clip X to clinical bounds --------------------------------
    print('\n=== 2. Clip X to clinical bounds ===')
    clip_report = []
    for col, (lo, hi) in CLIP.items():
        s = df[col]
        n_lo = int(((s < lo) & s.notna()).sum())
        n_hi = int(((s > hi) & s.notna()).sum())
        df[col] = s.clip(lower=lo, upper=hi)
        if n_lo or n_hi:
            clip_report.append({'col': col, 'lo': lo, 'hi': hi,
                                'n_below_lo': n_lo, 'n_above_hi': n_hi})
    rep = pd.DataFrame(clip_report)
    if rep.empty:
        print('no values needed clipping (all within bounds).')
    else:
        print(rep.to_string(index=False))

    # ---- 2a. Collapse high-corr min/max pairs to mean / spread ------
    print('\n=== 2a. Mean/spread re-parameterization ===')
    drop_cols = []
    for base in MEAN_SPREAD_PAIRS:
        lo_col, hi_col = f'{base}_min', f'{base}_max'
        df[f'{base}_mean']   = (df[hi_col] + df[lo_col]) / 2.0
        df[f'{base}_spread'] = (df[hi_col] - df[lo_col]).abs()
        drop_cols.extend([lo_col, hi_col])
        print(f'  {base:12s}: built {base}_mean, {base}_spread '
              f'(dropped {lo_col}, {hi_col})')
    df = df.drop(columns=drop_cols)

    # ---- 2b. Combined 3-sigma + 99% center quantile clip ------------
    # For each continuous feature, take the TIGHTER bound on each side:
    #   lo = max(mean - 3*sd, q0.005)
    #   hi = min(mean + 3*sd, q0.995)
    # Whichever rule is more restrictive wins per side, so heavy-tailed
    # features get pulled in by the quantile rule and well-behaved ones
    # by the sigma rule. Symmetric, no shape assumption.
    print('\n=== 2b. Combined 3-sigma + 99% center quantile clip ===')
    cont = [c for c in V_NAMES_FINAL if c not in {'AIDS', 'HEM', 'METS',
                                                  'Adm_type'}]
    bound_report = []
    for col in cont:
        s = df[col]
        m = s.mean(skipna=True)
        sd = s.std(skipna=True, ddof=1)
        s_lo, s_hi = m - 3 * sd, m + 3 * sd
        q_lo, q_hi = s.quantile(0.005), s.quantile(0.995)
        lo = max(s_lo, q_lo)
        hi = min(s_hi, q_hi)
        rule_lo = 'sigma' if s_lo >= q_lo else 'quantile'
        rule_hi = 'sigma' if s_hi <= q_hi else 'quantile'
        n_lo = int(((s < lo) & s.notna()).sum())
        n_hi = int(((s > hi) & s.notna()).sum())
        df[col] = s.clip(lower=lo, upper=hi)
        bound_report.append({'col': col,
                             'lo': round(lo, 2), 'hi': round(hi, 2),
                             'rule_lo': rule_lo, 'rule_hi': rule_hi,
                             'n_below': n_lo, 'n_above': n_hi})
    print(pd.DataFrame(bound_report).to_string(index=False))

    # ---- 3. Missingness indicators ----------------------------------
    print('\n=== 3. Missingness indicators ===')
    for col in MISS_IND:
        ind = df[col].isna().astype(int)
        df[f'M_{col}'] = ind
        print(f'M_{col:22s} : {int(ind.sum())} / {len(df)} '
              f'({100 * ind.mean():.1f}%) missing')

    # ---- 4. Drop patients with any DISCHTIME < ADMITTIME ------------
    print('\n=== 4. Drop patients with bad timestamps (DISCHTIME < ADMITTIME) ===')
    bad_adm = adm[adm['DISCHTIME'] < adm['ADMITTIME']]
    bad_subj = set(bad_adm['SUBJECT_ID'].dropna().unique())
    n_bad_in_cohort = int(df['SUBJECT_ID'].isin(bad_subj).sum())
    print(f'patients with at least one DISCHTIME<ADMITTIME admission: '
          f'{n_bad_in_cohort}')
    df = df[~df['SUBJECT_ID'].isin(bad_subj)].reset_index(drop=True)
    print(f'cohort after drop: {len(df)}')

    # ---- 5. Build recurrent-event outcome ---------------------------
    print('\n=== 5. Per-patient recurrent-event outcome ===')
    cohort_subj = df['SUBJECT_ID'].dropna().unique()
    all_adm = adm[adm['SUBJECT_ID'].isin(cohort_subj)].sort_values(
        ['SUBJECT_ID', 'ADMITTIME'])
    g = all_adm.groupby('SUBJECT_ID', sort=False)
    n_adm = g.size()
    first_admit = g['ADMITTIME'].min()
    last_dis = g['DISCHTIME'].max()
    last_death = g['DEATHTIME'].max()
    censor_dt = last_death.fillna(last_dis)
    fu_days = (censor_dt - first_admit).dt.total_seconds() / 86400
    last_died = g['HOSPITAL_EXPIRE_FLAG'].apply(lambda s: int(s.iloc[-1]))
    first_hadm = (adm.sort_values('ADMITTIME')
                    .drop_duplicates('SUBJECT_ID', keep='first')
                    [['SUBJECT_ID', 'HADM_ID']]
                    .rename(columns={'HADM_ID': 'first_HADM_ID'}))

    out = (df.merge(pd.DataFrame({
                'SUBJECT_ID': n_adm.index,
                'n_admissions': n_adm.values,
                'n_events': (n_adm - 1).values,
                'follow_up_days': fu_days.values,
                'died_at_end': last_died.values,
            }), on='SUBJECT_ID', how='left')
            .merge(first_hadm, on='SUBJECT_ID', how='left'))
    out['is_first_hospital_admit'] = (
        out['HADM_ID'] == out['first_HADM_ID']).astype(int)

    print(f'patients               : {len(out)}')
    print(f'mean events / patient  : {out["n_events"].mean():.3f}')
    print(f'patients with 0 events : '
          f'{int((out["n_events"] == 0).sum())} '
          f'({100 * (out["n_events"] == 0).mean():.1f}%)')
    print(f'patients died at end   : '
          f'{int(out["died_at_end"].sum())} '
          f'({100 * out["died_at_end"].mean():.1f}%)')
    print(f'is_first_hospital_admit: '
          f'{int(out["is_first_hospital_admit"].sum())} '
          f'({100 * out["is_first_hospital_admit"].mean():.1f}%)')

    # ---- 6. Drop residual negative follow-up ------------------------
    print('\n=== 6. Drop residual negative follow-up ===')
    bad = out['follow_up_days'] < 0
    print(f'patients with negative follow-up: {int(bad.sum())} (dropped)')
    out = out[~bad].reset_index(drop=True)
    print(f'cohort size after drop: {len(out)}')

    # ---- 7. Final follow-up summary ---------------------------------
    print('\n=== 7. Follow-up summary (after drops) ===')
    print(out['follow_up_days'].describe().to_string())
    print('\nQuantiles:')
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        print(f'  {int(q*100)}%: {out["follow_up_days"].quantile(q):8.1f} d')

    # ---- 8. Write ----------------------------------------------------
    final_cols = (['HADM_ID', 'SUBJECT_ID']
                  + V_NAMES_FINAL
                  + [f'M_{c}' for c in MISS_IND]
                  + ['n_admissions', 'n_events', 'follow_up_days',
                     'died_at_end', 'is_first_hospital_admit'])
    path = os.path.join(OUT, 'cohort_clean.csv')
    out[final_cols].to_csv(path, index=False)
    print(f'\nwrote {path}: {out[final_cols].shape}')


if __name__ == '__main__':
    main()
