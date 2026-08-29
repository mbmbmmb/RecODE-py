"""Model-selection table for the ODE family, rendered as a figure.

ODE-Cox (q == 1), ODE-AM (alpha == 1) and ODE-LT (q specified up to scale) are
exact nested restrictions of the ODE-Flex sieve parameterisation, so fitting all
four under ONE objective makes their information criteria comparable and the
nesting exact. Rows are the generating model, columns the model BIC selects.

Generating settings: S1 (Cox-type), S2 (AFT-type), S3 (Box-Cox) and S7. Setting 4
is deliberately NOT used as the ODE-Flex truth -- its q(u)=2/(1+u) IS the Box-Cox
form at rho=1, so ODE-LT contains it with one parameter and is correctly selected.
Setting 7 uses q(u)=1+0.7 sin(1.5u), non-monotone and hence outside the family at
every rho, so it genuinely requires the nonparametric fit.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'bic_model_selection', 'results')
OUT = os.path.join(HERE, 'bic_model_selection')

ROWS = [(1, 'Cox',  'ODE-Cox',  r'S1  $\alpha=t^2{+}1$,  $q=1$'),
        (2, 'AM',   'ODE-AM',   r'S2  $\alpha=1$,  $q=2/(1{+}u)$'),
        (3, 'LT',   'ODE-LT',   r'S3  $\alpha=0.2/(1{+}t)$,  Box-Cox $\rho=0.5$'),
        (7, 'Flex', 'ODE-Flex', r'S7  $\alpha=t{+}1$,  $q=1{+}0.7\sin(1.5u)$')]
COLS = ['ODE-Cox', 'ODE-AM', 'ODE-LT', 'ODE-Flex']
KEY = {'ODE-Cox': 'Cox', 'ODE-AM': 'AM', 'ODE-LT': 'LT', 'ODE-Flex': 'Flex'}


def main(csv=None, out=None, ns=(1000, 4000)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = pd.read_csv(csv or os.path.join(RES, 'bic_selection_wide.csv'))
    fig, axes = plt.subplots(len(ns), 1, figsize=(11.5, 2.6 * len(ns) + 1.4))
    axes = np.atleast_1d(axes)

    for ax, n in zip(axes, ns):
        ax.axis('off')
        cells, colours, rlab, nok = [], [], [], []
        for s, key, name, spec in ROWS:
            r = d[(d.setting == s) & (d.n == n)]
            if not len(r):
                continue
            r = r.iloc[0]
            rlab.append(f'  {spec}   ')
            cells.append([f'{r[KEY[c]]:.1f}' for c in COLS])
            colours.append([KEY[c] == key for c in COLS])
            nok.append(int(r['ok']))
        if not cells:
            continue
        tb = ax.table(cellText=cells, colLabels=COLS,
                      rowLabels=rlab, cellLoc='center', rowLoc='left',
                      loc='center')
        tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
        for j in range(len(COLS)):
            tb[0, j].set_facecolor('#e8e8e8')
            tb[0, j].set_text_props(weight='bold')
        for i, diag in enumerate(colours, start=1):
            for j, on in enumerate(diag):
                v = float(cells[i - 1][j])
                if on:
                    tb[i, j].set_facecolor('#c9e7c9' if v >= 95 else '#e6f2d9')
                    tb[i, j].set_text_props(weight='bold')
                elif v > 0:
                    tb[i, j].set_facecolor('#fdf0d5')
        ax.set_title(f'n = {n}', fontsize=11.5, weight='bold', pad=6)
        # the convergence range moves to the caption rather than being dropped:
        # the percentages are conditional on all four models fitting
        ax._nok = (min(nok), max(nok))

    fig.suptitle('BIC model selection in the ODE family\n'
                 'rows = generating model,  columns = model selected  '
                 r'(%, 100 replications, $\alpha$ sieve $\lceil N^{1/5}\rceil+4$)',
                 fontsize=12.5)
    rng = [a._nok for a in axes if hasattr(a, '_nok')]
    lo = min(r[0] for r in rng) if rng else 0
    hi = max(r[1] for r in rng) if rng else 0
    span = f'all {hi}' if lo == hi else f'{lo}-{hi}'
    fig.text(0.5, 0.012,
             'Shaded diagonal = correct selection. Percentages are conditional on all four '
             f'models fitting, which held for {span} of the 100 replications per cell.\n'
             r'Setting 4 ($q=2/(1{+}u)$) is excluded as the ODE-Flex truth: it IS Box-Cox at '
             r'$\rho=1$, so ODE-LT contains it and is correctly chosen.',
             ha='center', fontsize=8.5, color='#333333')
    fig.tight_layout(rect=[0, 0.055, 1, 0.93])
    os.makedirs(OUT, exist_ok=True)
    path = out or os.path.join(OUT, 'bic_model_selection.png')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print(f'wrote {path}')
    return path


if __name__ == '__main__':
    main()
