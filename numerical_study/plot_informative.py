"""Coverage table for the informative-censoring study, rendered as a figure.

Rows are (paper setting, censoring regime); columns report bias, empirical SD,
mean estimated SE and 95% CI coverage per coefficient.

Only regimes in which conditional independence given X still holds are shown.
The frailty-informative regime -- where C depends on the UNOBSERVED frailty and
the assumption is genuinely violated -- is excluded by design and reported
numerically in the text instead.
"""
from __future__ import annotations
import os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'simulation_study', 'informative_censoring', 'results')
OUT = os.path.join(HERE, 'simulation_study', 'informative_censoring')

REGIMES = [('random',   r"random:  $C\sim U(a,b)$"),
           ('cov',      r"covariate-dep:  $C=C_0e^{x'\gamma_c}$, $\gamma_c=-0.5$"),
           ('cov_decr', r"monotone decr:  $C=C_0e^{-x'\beta}$")]
SETTING_DESC = {
    1: 'S1 Cox-type',
    2: 'S2 AFT-type',
    3: 'S3 Box-Cox LT',
    4: 'S4 general LT',
}
SETTING_FOOT = {
    1: r"S1: $\alpha(t)=t^2{+}1$, $q\equiv1$ (Cox-type)",
    2: r"S2: $\alpha\equiv1$, $q(u)=2/(1{+}u)$ (AFT-type)",
    3: r"S3: Box-Cox LT, $\rho=0.5$",
    4: r"S4: $\alpha(t)=t{+}1$, $q(u)=2/(1{+}u)$ (general LT)",
}


def _load(setting, name):
    for f in (os.path.join(RES, f's{setting}_{name}.npz'),
              os.path.join(RES, f'{name}.npz') if setting == 1 else None):
        if f and os.path.isfile(f):
            return np.load(f, allow_pickle=True)
    return None


def main(settings=(1, 2, 3, 4), out=None, min_cp=None):
    """``min_cp``: drop rows whose mean coverage falls below this
    threshold (used for the published figure); ``None`` keeps every row."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, cells, colours, nrep = [], [], [], None
    for s in settings:
        for name, lab in REGIMES:
            z = _load(s, name)
            if z is None:
                continue
            B, S, truth = z['beta'], z['se'], z['truth']
            nrep = max(nrep or 0, B.shape[0])
            bias = np.nanmean(B, 0) - truth
            esd = np.nanstd(B, 0, ddof=1)
            mse = np.nanmean(S, 0)
            inside = ((B - 1.96 * S) <= truth) & (truth <= B + 1.96 * S)
            inside = np.where(np.isfinite(B) & np.isfinite(S),
                              inside.astype(float), np.nan)
            cov = np.nanmean(inside, 0)     # NaN cols (the LTM anchor) excluded
            m = float(np.nanmean(cov))
            if min_cp is not None and m < min_cp:
                continue
            rows.append((s, lab))
            cells.append([f'{np.nanmax(np.abs(bias)):+.4f}',
                          f'{np.nanmean(esd):.4f}', f'{np.nanmean(mse):.4f}',
                          f'{np.nanmean(mse)/np.nanmean(esd):.2f}',
                          *[('--' if not np.isfinite(c) else f'{c:.3f}')
                            for c in cov],
                          f'{m:.3f}'])
            colours.append('#d6efd6' if m >= 0.93 else
                           ('#fdf3d0' if m >= 0.90 else '#f8d7d7'))

    if not cells:
        raise FileNotFoundError(f'no informative-censoring results under {RES}'
                                + ('' if min_cp is None else
                                   f' with mean coverage >= {min_cp}'))

    cols = ['max |bias|', 'emp. SD', 'mean SE', 'SE/SD',
            r'CP $\beta_1$', r'CP $\beta_2$', r'CP $\beta_3$', 'mean CP']
    row_labels = [f'  {SETTING_DESC.get(s, s)}   |   {lab}  '
                  for s, lab in rows]

    fig, ax = plt.subplots(figsize=(14.5, 0.46 * len(cells) + 2.1))
    ax.axis('off')
    tb = ax.table(cellText=cells, colLabels=cols, rowLabels=row_labels,
                  cellLoc='center', rowLoc='left', loc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.5)

    ncol = len(cols)
    for j in range(ncol):
        tb[0, j].set_facecolor('#e8e8e8'); tb[0, j].set_text_props(weight='bold')
    prev = None
    for i, ((s, _), col) in enumerate(zip(rows, colours), start=1):
        for j in range(ncol):
            tb[i, j].set_facecolor(col if j >= ncol - 4 else 'white')
        lab = tb[i, -1]
        lab.set_text_props(fontsize=9)
        if s != prev:                       # separator between settings
            for j in range(-1, ncol):
                tb[i, j].visible_edges = 'BLRT'
                tb[i, j].set_linewidth(1.6)
            prev = s
    ax.set_title('Informative censoring: regimes where conditional independence '
                 f'given $X$ holds\n({nrep} replications, $n=1000$, '
                 r'true $\beta=(1,1,1)$;  green $\geq$ 0.93,  amber $\geq$ 0.90)',
                 fontsize=12.5, pad=14)
    seen = sorted({s for s, _ in rows})
    ax.text(0.0, -0.06 - 0.035 * 0, '     '.join(SETTING_FOOT[s] for s in seen
                                                 if s in SETTING_FOOT),
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            color='#333333')
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = out or os.path.join(OUT, 'informative_censoring.png')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print(f'wrote {path}  ({len(cells)} rows)')
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-cp', type=float, default=None,
                    help='drop rows with mean coverage below this')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    main(out=a.out, min_cp=a.min_cp)
