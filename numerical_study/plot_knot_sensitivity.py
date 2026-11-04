"""Figure S1 of the Supplementary Material: fitted log q against the truth for several
numbers of knots and both knot placements, in the three settings with an oscillating q
(ODE-AM, median over 20 replications, n = 2000), over the left 75% of the range of mu_hat
used in Table S3.

Reads bic_model_selection/results/knot_curves_s13_15.npz, produced by knot_curves.py.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bic_model_selection')
C = np.load(os.path.join(R, 'results', 'knot_curves_s13_15.npz'))
SHOW = [0.05, 0.14, 0.25, 0.35]
TITLE = {13: r'$q(u)=1+0.7\sin(1.5u)$', 14: r'$q(u)=1/(1+u)+0.1\sin(1.5u)$',
         15: r'$q(u)=\{1+0.7\sin(1.5u)\}/(1+u)$'}
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(SHOW)))

fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.6), sharex='col', sharey='col')
for j, s in enumerate((13, 14, 15)):
    g = C[f's{s}_grid']
    for i, (pl, lab) in enumerate([('quantile', 'sample quantiles'), ('equal', 'equally spaced')]):
        a = ax[i, j]
        for c, p in zip(cols, SHOW):
            a.plot(g, np.median(C[f's{s}_{pl}_p{p:g}'], 0), color=c, lw=1.7, label=f'$p$={p:g}')
        a.plot(g, C[f's{s}_true'], 'k--', lw=2.4, label='truth')
        a.set_title(f'({"abc"[j] if i == 0 else "def"[j]}) {TITLE[s]}, {lab}', fontsize=10.5)
        a.grid(alpha=.25)
        if i == 1:
            a.set_xlabel(r'$u$')
        if j == 0:
            a.set_ylabel(r'$\log\hat q(u)$')
ax[0, 0].legend(fontsize=8.5)
fig.tight_layout()
out = os.path.join(R, 'knot_curves_all_settings.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
