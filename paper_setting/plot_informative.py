"""Figure for the informative-censoring study.

Shows the regimes in which conditional independence given X still holds, i.e.
those the estimator is expected to (and does) handle: random censoring, and
covariate-dependent censoring C = C0*exp(x'gamma_c), with and without frailty.

The frailty-informative regime (C depending on the UNOBSERVED xi) is excluded
here by design -- it is the assumption-violating stress test, reported in the
text as a numerical result rather than plotted alongside the valid regimes.
"""
from __future__ import annotations
import os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'informative')
OUT = os.path.join(HERE, 'plots', 'informative')

# regimes in which conditional independence given X holds. The
# frailty-informative regime (C depending on the unobserved xi) is excluded --
# it is the assumption-violating stress test, reported numerically in the text.
REGIMES = [('random', 'Random\n$C \\sim U(a,b)$'),
           ('cov', "Covariate-dependent\n$C = C_0 e^{x'\\gamma_c}$, $\\gamma_c=-0.5$"),
           ('cov_decr', "Monotone decreasing\n$C = C_0 e^{-x'\\beta}$"),
           ('random_fr', 'Random\n+ gamma frailty')]


def main(out=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {}
    for k, _ in REGIMES:
        f = os.path.join(RES, f'{k}.npz')
        if not os.path.isfile(f):
            raise FileNotFoundError(f)
        data[k] = np.load(f, allow_pickle=True)

    p = data[REGIMES[0][0]]['truth'].size
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    xs = np.arange(len(REGIMES))
    w = 0.8 / p
    cols = plt.cm.tab10(np.linspace(0, 0.3, p))

    ax = axes[0]
    for j in range(p):
        b = [float(np.nanmean(data[k]['beta'][:, j]) - data[k]['truth'][j])
             for k, _ in REGIMES]
        e = [float(np.nanstd(data[k]['beta'][:, j], ddof=1)
                   / np.sqrt(np.isfinite(data[k]['beta'][:, j]).sum()))
             for k, _ in REGIMES]
        ax.bar(xs + (j - (p - 1) / 2) * w, b, w, yerr=e, capsize=3,
               color=cols[j], label=rf'$\beta_{j+1}$')
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in REGIMES], fontsize=9)
    ax.set_ylabel('bias'); ax.set_title('Bias (bars = MC standard error)')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    for j in range(p):
        B, S = data[REGIMES[0][0]]['beta'], data[REGIMES[0][0]]['se']
        cov = []
        for k, _ in REGIMES:
            B, S, tr = data[k]['beta'], data[k]['se'], data[k]['truth']
            c = np.nanmean((B[:, j] - 1.96 * S[:, j] <= tr[j])
                           & (tr[j] <= B[:, j] + 1.96 * S[:, j]))
            cov.append(float(c))
        ax.bar(xs + (j - (p - 1) / 2) * w, cov, w, color=cols[j],
               label=rf'$\beta_{j+1}$')
    ax.axhline(0.95, color='r', ls='--', lw=1.2, label='nominal 0.95')
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in REGIMES], fontsize=9)
    ax.set_ylim(0.80, 1.0); ax.set_ylabel('empirical coverage')
    ax.axhline(0.95, color='r', ls='--', lw=1.2)
    ax.set_title('95% CI coverage')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, axis='y', alpha=0.3)

    nrep = data[REGIMES[0][0]]['beta'].shape[0]
    fig.suptitle('Informative censoring: regimes where conditional independence '
                 f'given $X$ holds ({nrep} replications)', fontsize=12)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = out or os.path.join(OUT, 'informative_censoring.png')
    fig.savefig(path, dpi=140)
    print(f'wrote {path}')
    return path


if __name__ == '__main__':
    main()
