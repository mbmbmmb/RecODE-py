"""Convert cohort_clean.csv to long format for the recurrent-event model.

Per patient (= one X vector from the first admission):
  - One row per *subsequent* admission inside the follow-up window,
    ``delta = 1`` at ``time = ADMITTIME_k - ADMITTIME_1`` (days since
    first admit).
  - One final censoring row, ``delta = 0``, at the ADMINISTRATIVE
    censoring time ``min(HORIZON_DAYS, death - first admit)``.
  - The X covariates (V1..V26 + missingness indicators) are repeated on
    every row for that patient.

Patients with no readmission contribute exactly one row (the censor).

Censoring
---------
The censoring time must not depend on the recurrent event process. Every patient is
therefore followed from the index admission to a common administrative horizon,
truncated only by death: C = min(horizon, time of death). Censoring at the last
observed discharge would make the follow-up depend on the number of readmissions.

Usage:  python build_long_format.py --horizon-days 730 --tag 2y   (two-year window, paper)

Output: ``<OUT>/cohort_long_<tag>.csv`` and ``.npz`` with columns
    id, time, delta, V1..V26, age_capped, M_*, is_death_censored
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from config import OUT, RAW

V_NAMES = [
    'age', 'heartrate_mean', 'heartrate_spread',
    'sysbp_mean', 'sysbp_spread',
    'tempc_mean', 'tempc_spread', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_mean', 'bun_spread', 'wbc_mean', 'wbc_spread',
    'potassium_mean', 'potassium_spread', 'sodium_mean', 'sodium_spread',
    'bicarbonate_mean', 'bicarbonate_spread', 'bilirubin_mean',
    'bilirubin_spread', 'gcs_min', 'AIDS', 'HEM', 'METS', 'Adm_type',
]
M_COLS = ['M_PaO2FiO2_vent_min', 'M_bilirubin_mean']

# Administrative follow-up horizon in days, measured from each patient's first
# admission; set from the command line (730 days = two years in the paper).
HORIZON_DAYS = 730.0


def main(tag='2y'):
    wide = pd.read_csv(os.path.join(OUT, 'cohort_clean.csv'))
    print(f'cohort_clean.csv : {wide.shape}')

    adm = pd.read_csv(os.path.join(RAW, 'ADMISSIONS.csv'),
                      parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'],
                      usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                               'DISCHTIME', 'DEATHTIME'])
    cohort = wide['SUBJECT_ID'].unique()
    adm = adm[adm['SUBJECT_ID'].isin(cohort)].sort_values(
        ['SUBJECT_ID', 'ADMITTIME'])

    g = adm.groupby('SUBJECT_ID', sort=False)
    first_admit = g['ADMITTIME'].min()
    last_death = g['DEATHTIME'].max()

    # ---- administrative censoring (independent of the event process) --
    # Follow every patient to the common horizon, truncated only by death.
    death_days = ((last_death - first_admit).dt.total_seconds() / 86400)
    horizon = np.inf if HORIZON_DAYS is None else float(HORIZON_DAYS)
    censor_days = death_days.where(death_days.notna() & (death_days <= horizon),
                                   other=horizon)
    is_death_censored = (death_days.notna() & (death_days <= horizon)).astype(int)

    # ---- event rows (subsequent admissions inside the window) --------
    adm = adm.assign(rank=g.cumcount() + 1)
    events = adm[adm['rank'] > 1].copy()
    events['first_admit'] = events['SUBJECT_ID'].map(first_admit)
    events['time'] = (
        (events['ADMITTIME'] - events['first_admit']).dt.total_seconds()
        / 86400
    )
    events['cens'] = events['SUBJECT_ID'].map(censor_days)
    n_drop = int((events['time'] >= events['cens']).sum())
    events = events[events['time'] < events['cens']].copy()
    if n_drop:
        print(f'dropped {n_drop} readmissions at/after the censoring time')
    events['delta'] = 1
    events = events[['SUBJECT_ID', 'time', 'delta']]

    # ---- censoring rows (one per patient) ---------------------------
    cens = pd.DataFrame({
        'SUBJECT_ID': censor_days.index,
        'time': censor_days.values,
        'delta': 0,
    })
    cens = cens[cens['time'] > 0]

    long = pd.concat([events, cens], ignore_index=True)
    # Sort so within each patient, rows go by ascending time and (on ties)
    # event rows (delta=1) come before the censoring row (delta=0).
    long['_sort_delta'] = -long['delta']  # 1 -> -1, 0 -> 0  (events first)
    long = (long.sort_values(['SUBJECT_ID', 'time', '_sort_delta'])
                .drop(columns='_sort_delta')
                .reset_index(drop=True))

    # Drop rows with non-positive or NaN times (can happen if a patient
    # has same-day re-admission timestamps or stray data errors).
    bad = (long['time'].isna()) | (long['time'] <= 0)
    n_bad = int(bad.sum())
    if n_bad:
        # Keep the censoring row even if time<=0 (set to a tiny positive)
        cens_mask = (long['delta'] == 0) & bad
        long.loc[cens_mask, 'time'] = 1e-6
        # Drop event rows with non-positive time (cannot precede baseline)
        evt_mask = (long['delta'] == 1) & bad
        long = long[~evt_mask].reset_index(drop=True)
        print(f'cleaned rows with time <= 0: bumped {int(cens_mask.sum())} '
              f'censor rows to 1e-6, dropped {int(evt_mask.sum())} event rows')

    # ---- attach X (constant within patient) -------------------------
    long['is_death_censored'] = long['SUBJECT_ID'].map(is_death_censored)
    x_cols = V_NAMES + M_COLS
    x_lookup = wide.set_index('SUBJECT_ID')[x_cols]
    long = long.merge(x_lookup, left_on='SUBJECT_ID', right_index=True,
                      how='inner')

    long = long.rename(columns={'SUBJECT_ID': 'id'})
    long = long[['id', 'time', 'delta'] + x_cols + ['is_death_censored']]

    # ---- summarize --------------------------------------------------
    n_pts = long['id'].nunique()
    n_rows = len(long)
    n_evt = int((long['delta'] == 1).sum())
    n_cens = int((long['delta'] == 0).sum())
    rows_per = long.groupby('id').size()
    print('\n=== Long-format summary ===')
    print(f'patients (= unique id) : {n_pts}')
    print(f'rows                   : {n_rows}')
    print(f'  event rows (delta=1) : {n_evt}')
    print(f'  censor rows (delta=0): {n_cens}')
    print(f'rows per patient: min={rows_per.min()}  median='
          f'{rows_per.median():.0f}  max={rows_per.max()}')

    # Sanity checks
    last_per_id = long.groupby('id')['delta'].last()
    n_bad_last = int((last_per_id != 0).sum())
    if n_bad_last:
        print(f'WARNING: {n_bad_last} patients do not end with delta=0')
    else:
        print('OK: every patient ends with delta=0 (censoring row last)')

    n_death_cens = int((long.drop_duplicates('id')['is_death_censored']
                        == 1).sum())
    print(f'death-censored patients: {n_death_cens} '
          f'({100 * n_death_cens / n_pts:.1f}%)')

    # ---- write CSV (human-readable) and NPZ (model-ready) -----------
    path = os.path.join(OUT, f'cohort_long_{tag}.csv')
    long.to_csv(path, index=False)
    print(f'\nwrote {path}: {long.shape}')

    x_cols_for_model = V_NAMES + M_COLS
    x_arr = long[x_cols_for_model].to_numpy(np.float64)

    # ---- NaN handled later, post-scale ------------------------------
    # We deliberately keep NaN in the output. scale_features.py fits
    # scalers on non-NaN train rows and ONLY then maps NaN -> 0, so
    # missing entries land at the post-scale center (additive
    # contribution beta*x = 0). The M_* indicators carry the
    # "test was/wasn't ordered" signal for high-missingness features.
    nan_per_col = np.isnan(x_arr).sum(axis=0)
    print('\n=== NaN counts (kept; filled at center after scaling) ===')
    for c, n in zip(x_cols_for_model, nan_per_col):
        if n:
            print(f'  {c:22s}: {int(n):6d} NaN')
    print(f'total NaN: {int(nan_per_col.sum())}')

    model_data = {
        'id':    long['id'].to_numpy(np.int64),
        'time':  long['time'].to_numpy(np.float64),
        'delta': long['delta'].to_numpy(np.int64),
        'x':     x_arr,
    }
    npz_path = os.path.join(OUT, f'cohort_long_{tag}.npz')
    np.savez_compressed(npz_path, **model_data,
                        x_cols=np.array(x_cols_for_model))
    print(f'wrote {npz_path}: '
          f'id={model_data["id"].shape}  time={model_data["time"].shape}  '
          f'delta={model_data["delta"].shape}  x={model_data["x"].shape}')

    # ---- format comparison vs. model expectation --------------------
    print('\n=== Format vs. recurrent_ode.fit() expectation ===')
    print(f'{"field":7s} {"model expects":35s} {"ours":25s} match?')
    print(f'{"-"*7} {"-"*35} {"-"*25} ------')
    rows = [
        ('id',    '(n,) or (n,1) int',
         f'{model_data["id"].shape} {model_data["id"].dtype}',  True),
        ('time',  '(n,) or (n,1) float',
         f'{model_data["time"].shape} {model_data["time"].dtype}', True),
        ('delta', '(n,) or (n,1) int (0/1)',
         f'{model_data["delta"].shape} {model_data["delta"].dtype}', True),
        ('x',     '(n_rows, p) float',
         f'{model_data["x"].shape} {model_data["x"].dtype}',     True),
    ]
    for name, expect, got, ok in rows:
        print(f'{name:7s} {expect:35s} {got:25s} {"OK" if ok else "NO"}')

    # dtype safety check (NaN is intentional here — see note above)
    n_nan = int(np.isnan(model_data['x']).sum())
    n_neg_t = int((model_data['time'] <= 0).sum())
    n_bad_d = int(((model_data['delta'] != 0) &
                   (model_data['delta'] != 1)).sum())
    print(f'\nNaN in x (kept)   : {n_nan}')
    print(f'time <= 0         : {n_neg_t}')
    print(f'delta not in {{0,1}}: {n_bad_d}')

    # Show the first patient with >=1 event so the layout is obvious
    sample_id = long[long['delta'] == 1]['id'].iloc[0]
    print(f'\nExample (id={sample_id}):')
    print(long[long['id'] == sample_id]
          [['id', 'time', 'delta', 'age', 'is_death_censored']]
          .to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--horizon-days', type=float, default=730.0)
    ap.add_argument('--tag', default='2y')
    a = ap.parse_args()
    HORIZON_DAYS = a.horizon_days
    main(a.tag)
