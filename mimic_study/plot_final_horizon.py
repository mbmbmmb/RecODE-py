"""Figure 5 of the paper (reads <OUT>/final_<tag>.npz; writes results/mimic_<tag>_functional.png).

Panel (a): log alpha-hat(t) under ODE-Flex with its 95% pointwise band, and the
ODE-Cox baseline for reference. Panel (b): log q-hat(u) with its 95% band over
the region holding the bulk of the fitted mean functions, and the ODE-Cox line
log q == 0.

Usage:  python plot_final_horizon.py [tag] [mass]   (mass: share of mu-hat(C)
        covered by the displayed u-range; default 0.95)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from config import OUT, RESULTS as FIG

tag = sys.argv[1] if len(sys.argv) > 1 else '2y'
mass = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
z = np.load(os.path.join(OUT, f'final_{tag}.npz'), allow_pickle=True)
zc = norm.ppf(0.975)

gu, lq, sq = z['gu'], z['log_q'], z['se_q']
umax = float(np.quantile(z['mu_c'], mass))
keep = gu <= umax
excl = ((lq - zc * sq > 0) | (lq + zc * sq < 0))[keep].mean()
print(f'{tag}: u <= {umax:.3f} holds {100 * mass:.0f}% of mu-hat(C); '
      f'band excludes 0 on {100 * excl:.1f}% of it; LR={float(z["LR"]):.2f} '
      f'(calibrated by bootstrap_final_lr.py)')

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

a = ax[0]
gt, la, sa = z['gt'], z['log_alpha'], z['se_alpha']
a.fill_between(gt, la - zc * sa, la + zc * sa, color='#c0392b', alpha=.18, lw=0,
               label='95% pointwise band')
a.plot(gt, la, color='#c0392b', lw=2.2, label=r'ODE-Flex  $\log\hat\alpha(t)$')
a.plot(gt, z['log_alpha_cox'], color='0.25', lw=1.4, ls=':',
       label=r'ODE-Cox  $\log\hat\alpha(t)$')
a.set_xscale('symlog', linthresh=1.0)
a.set_xlim(0, float(z['horizon']))
a.set_xticks([0, 1, 7, 30, 90, 365, 730][:7 if z['horizon'] > 400 else 5])
a.set_xticklabels(['0', '1', '7', '30', '90', '365', '730'][:7 if z['horizon'] > 400 else 5])
a.set_xlabel('days since index admission')
a.set_ylabel(r'$\log\hat\alpha(t)$')
a.set_title(r'(a) baseline rate $\log\hat\alpha(t)$')
a.grid(alpha=.3)
a.legend(fontsize=8.5, loc='upper left')

b = ax[1]
b.fill_between(gu[keep], (lq - zc * sq)[keep], (lq + zc * sq)[keep], color='#2e86c1',
               alpha=.2, lw=0, label='95% pointwise band')
b.plot(gu[keep], lq[keep], color='#2e86c1', lw=2.2, label=r'ODE-Flex  $\log\hat q(u)$')
b.axhline(0, color='0.25', ls='--', lw=1.4, label=r'ODE-Cox  $\log q\equiv0$')
b.set_xlim(0, umax)
b.set_xlabel(r'accumulated mean events  $u=\mu(t)$')
b.set_ylabel(r'$\log\hat q(u)$')
b.set_title(r'(b) state dependence $\log\hat q(u)$')
b.grid(alpha=.3)
b.legend(fontsize=8.5, loc='upper right')

fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, f'mimic_{tag}_functional.png')
fig.savefig(out, dpi=200)
print('wrote', out)
