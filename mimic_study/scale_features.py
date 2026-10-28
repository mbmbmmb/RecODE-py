"""Scale the post-clip features.

Per-column dispatch:
  - RobustScaler (median / IQR) for right-skewed features.
  - StandardScaler (mean / std) for roughly symmetric features.
  - Pass-through for binary indicators and Adm_type.

Fits each scaler on the non-NaN TRAIN rows of patients (70%, seed=42)
only — val/test rows are transformed with the same parameters. NaN
values are preserved through the scaling step and ONLY then filled
with 0, so each missing entry lands exactly at the post-scale center
(additive contribution beta*x = 0). The M_* indicator columns absorb
the "test was/wasn't ordered" signal for the high-missingness
features (PaO2FiO2_vent_min, bilirubin_*).

Outputs:
  - <OUT>/cohort_long_scaled_<tag>.npz
  - <OUT>/scaler_params_<tag>.csv
  - <OUT>/plots/<feature>.png  (only with --plots)

Usage:  python scale_features.py --tag 2y [--plots]
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT
PLOTS = os.path.join(OUT, 'plots')

ROBUST_COLS = [
    'bilirubin_mean', 'bilirubin_spread',
    'bun_mean', 'bun_spread',
    'wbc_mean', 'wbc_spread',
    'urineoutput', 'PaO2FiO2_vent_min',
]
STANDARD_COLS = [
    'age',
    'heartrate_mean', 'heartrate_spread',
    'sysbp_mean', 'sysbp_spread',
    'tempc_mean', 'tempc_spread',
    'sodium_mean', 'sodium_spread',
    'bicarbonate_mean', 'bicarbonate_spread',
    'potassium_mean', 'potassium_spread',
    'gcs_min',
]
PASSTHROUGH_COLS = [
    'AIDS', 'HEM', 'METS', 'Adm_type',
    'M_PaO2FiO2_vent_min', 'M_bilirubin_mean',
]
BINARY = {'AIDS', 'HEM', 'METS',
          'M_PaO2FiO2_vent_min', 'M_bilirubin_mean'}
CATEGORICAL = {'Adm_type'}


def patient_split(ids, seed=42, ratios=(0.70, 0.15, 0.15)):
    rng = np.random.RandomState(seed)
    uniq = np.unique(ids)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    train = set(uniq[:n_tr].tolist())
    val   = set(uniq[n_tr:n_tr + n_va].tolist())
    test  = set(uniq[n_tr + n_va:].tolist())
    return train, val, test


def fit_scalers(train_x, cols):
    """Return (centers, scales, kind) per column; kind in {robust,standard,passthrough}.

    Fit statistics IGNORE NaN — missing entries are kept out of the
    train moments and quantiles. They are filled with 0 only AFTER
    the scaling transform is applied (see apply_scalers).
    """
    centers = np.zeros(len(cols), dtype=np.float64)
    scales  = np.ones(len(cols), dtype=np.float64)
    kind    = []
    for i, col in enumerate(cols):
        x = train_x[:, i]
        if col in ROBUST_COLS:
            c = float(np.nanmedian(x))
            s = float(np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25))
            centers[i], scales[i] = c, s if s > 0 else 1.0
            kind.append('robust')
        elif col in STANDARD_COLS:
            c = float(np.nanmean(x))
            s = float(np.nanstd(x, ddof=1))
            centers[i], scales[i] = c, s if s > 0 else 1.0
            kind.append('standard')
        elif col in PASSTHROUGH_COLS:
            kind.append('passthrough')
        else:
            raise ValueError(f'unknown feature {col}')
    return centers, scales, kind


def apply_scalers(x, centers, scales, kind):
    """Center/scale with NaN preserved, then fill NaN -> 0.

    Filling AFTER the linear transform places every missing entry at
    the post-scale center (=0), so beta*x contributes nothing for
    missing inputs while the M_* indicators carry the missingness
    signal where one was attached.
    """
    out = x.copy().astype(np.float64)
    for i, k in enumerate(kind):
        if k == 'passthrough':
            continue
        out[:, i] = (out[:, i] - centers[i]) / scales[i]
    out = np.nan_to_num(out, nan=0.0)
    return out


def plot_scaled_continuous(col, x, kind, center, scale):
    """Hist of post-scale values for a continuous feature."""
    s = pd.Series(x).astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(s, bins=80, color='seagreen', alpha=0.85)
    ax.axvline(0, color='black', ls='-', lw=0.8, alpha=0.6)
    handles = [
        plt.Line2D([], [], color='none', label=f'scaler   = {kind}'),
        plt.Line2D([], [], color='none',
                   label=('center   = {:.3g}  ({})'.format(
                       center, 'median' if kind == 'robust' else 'mean'))),
        plt.Line2D([], [], color='none',
                   label=('scale    = {:.3g}  ({})'.format(
                       scale, 'IQR' if kind == 'robust' else 'std'))),
        plt.Line2D([], [], color='none', label=f'n        = {len(s)}'),
        plt.Line2D([], [], color='none',
                   label=f'min      = {s.min():.2f}'),
        plt.Line2D([], [], color='none',
                   label=f'max      = {s.max():.2f}'),
        plt.Line2D([], [], color='none',
                   label=f'mean     = {s.mean():.2f}'),
        plt.Line2D([], [], color='none',
                   label=f'std      = {s.std(ddof=1):.2f}'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=9,
              framealpha=0.9, handlelength=0)
    ax.set_title(f'{col}   post-{kind} scale (train-set distribution)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(f'scaled {col}')
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'{col}.png'), dpi=110)
    plt.close(fig)


def plot_corr_heatmap(df, cols, out_path, title):
    """Pearson correlation heatmap; saves PNG and returns the corr DataFrame."""
    corr = df[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(0.55 * n + 2, 0.55 * n + 1))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=60, ha='right', fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6,
                    color='white' if abs(v) > 0.55 else 'black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return corr


def plot_categorical(col, x):
    s = pd.Series(x).astype(int)
    n = len(s)
    counts = s.value_counts().sort_index()
    cats = counts.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([str(c) for c in cats], counts.values,
                  color='seagreen', alpha=0.85)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f'{int(v)}\n({100 * v / n:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    ax.set_title(f'{col}   passthrough (train-set distribution, n = {n})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('count')
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'{col}.png'), dpi=110)
    plt.close(fig)


def main(tag='2y', plots=False):
    npz = np.load(os.path.join(OUT, f'cohort_long_{tag}.npz'), allow_pickle=True)
    cols = [c.decode() if isinstance(c, bytes) else str(c)
            for c in npz['x_cols']]
    x = npz['x']
    ids = npz['id']
    print(f'cohort_long.npz : x={x.shape}  ids={len(np.unique(ids))} patients')

    # ---- Patient-level 70/15/15 split (seed=42) ---------------------
    train_pids, val_pids, test_pids = patient_split(ids)
    train_mask = np.array([int(i) in train_pids for i in ids])
    val_mask   = np.array([int(i) in val_pids   for i in ids])
    test_mask  = np.array([int(i) in test_pids  for i in ids])
    print(f'split  rows: train={train_mask.sum()}  '
          f'val={val_mask.sum()}  test={test_mask.sum()}')
    print(f'split   pts: train={len(train_pids)}  '
          f'val={len(val_pids)}  test={len(test_pids)}')

    # ---- Fit on train rows ------------------------------------------
    centers, scales, kind = fit_scalers(x[train_mask], cols)

    # ---- Apply to all -----------------------------------------------
    x_scaled = apply_scalers(x, centers, scales, kind)

    # ---- Save scaler params -----------------------------------------
    params_df = pd.DataFrame({
        'feature': cols, 'scaler': kind,
        'center': centers, 'scale': scales,
    })
    params_path = os.path.join(OUT, f'scaler_params_{tag}.csv')
    params_df.to_csv(params_path, index=False)
    print(f'wrote {params_path}')

    # ---- Save scaled npz --------------------------------------------
    out_npz = os.path.join(OUT, f'cohort_long_scaled_{tag}.npz')
    np.savez_compressed(out_npz,
                        id=npz['id'], time=npz['time'], delta=npz['delta'],
                        x=x_scaled, x_cols=npz['x_cols'],
                        train_mask=train_mask, val_mask=val_mask,
                        test_mask=test_mask,
                        scaler_kind=np.array(kind),
                        scaler_center=centers, scaler_scale=scales)
    print(f'wrote {out_npz}: x={x_scaled.shape} (scaled)')

    if not plots:
        return
    os.makedirs(PLOTS, exist_ok=True)

    # ---- Plot per-patient unique post-scale hists from TRAIN --------
    train_ids_arr = ids[train_mask]
    train_x_scaled = x_scaled[train_mask]
    _, first_idx = np.unique(train_ids_arr, return_index=True)
    px = train_x_scaled[first_idx]
    print(f'plotting train-set per-patient distributions: {px.shape}')

    for i, col in enumerate(cols):
        if col in BINARY or col in CATEGORICAL:
            plot_categorical(col, px[:, i])
        else:
            plot_scaled_continuous(col, px[:, i],
                                   kind[i], centers[i], scales[i])
    print(f'wrote {len(cols)} scaled-feature PNGs to {PLOTS}')

    # ---- Post-scale correlation heatmap (model frame) ---------------
    # Uses the SAME train per-patient frame as the histograms. Adm_type
    # gets one-hot encoded into 2 dummies (reference = medical = 0).
    df_corr = pd.DataFrame(px, columns=cols)
    adm_idx = cols.index('Adm_type')
    df_corr['Adm_type_scheduled']   = (df_corr['Adm_type'] == 1).astype(float)
    df_corr['Adm_type_unscheduled'] = (df_corr['Adm_type'] == 2).astype(float)
    df_corr = df_corr.drop(columns=['Adm_type'])
    cols_corr = [c for c in cols if c != 'Adm_type'] + [
        'Adm_type_scheduled', 'Adm_type_unscheduled']
    corr_path_png = os.path.join(PLOTS, 'correlation_postscale.png')
    corr = plot_corr_heatmap(df_corr, cols_corr, corr_path_png,
        'Pearson correlation: post-scale model inputs (train, per-patient)')
    corr_csv = os.path.join(OUT, 'correlation_postscale.csv')
    corr.to_csv(corr_csv)
    print(f'wrote {corr_path_png}')
    print(f'wrote {corr_csv}: {corr.shape}')

    cm = corr.values.copy()
    np.fill_diagonal(cm, 0.0)
    pairs = sorted(
        ((cols_corr[i], cols_corr[j], cm[i, j])
         for i in range(len(cols_corr))
         for j in range(i + 1, len(cols_corr))),
        key=lambda t: abs(t[2]), reverse=True)
    print('\ntop 15 |corr| pairs (post-scale, model frame):')
    for a, b, v in pairs[:15]:
        print(f'  {a:24s}  {b:24s}  {v:+.3f}')

    # ---- Sanity report ----------------------------------------------
    print('\n=== Post-scale per-feature stats (train, per-patient) ===')
    rows = []
    for i, col in enumerate(cols):
        col_vals = px[:, i]
        rows.append({
            'feature': col, 'scaler': kind[i],
            'center': round(centers[i], 4),
            'scale':  round(scales[i], 4),
            'min':    round(float(np.min(col_vals)), 3),
            'p1':     round(float(np.quantile(col_vals, 0.01)), 3),
            'median': round(float(np.median(col_vals)), 3),
            'p99':    round(float(np.quantile(col_vals, 0.99)), 3),
            'max':    round(float(np.max(col_vals)), 3),
            'mean':   round(float(np.mean(col_vals)), 3),
            'std':    round(float(np.std(col_vals, ddof=1)), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Scale the post-clip features.')
    ap.add_argument('--tag', default='2y')
    ap.add_argument('--plots', action='store_true')
    a = ap.parse_args()
    main(a.tag, a.plots)
