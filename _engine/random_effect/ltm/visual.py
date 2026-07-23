"""Port of random effect/ltm/visual.m.

MATLAB's copy had a stub-signature bug (it referenced ``seed``, ``pxt``,
``pxq``, ``ci`` without declaring them) — we keep the behaviour implied
by the surrounding ``summary_{cox,aft}.m`` calls:
``visual(N, seed, data_setting, knots_setting, pxt, pxq, show)``.

Truth curves per setting:
- setting 1 (Cox):   true_a = 1 + t^2,  true_q = 1
- setting 2 (AFT):   true_a = 1,        true_q = 2 / (1 + u)
"""
from __future__ import annotations

import os
import numpy as np

from ...common import spcol


def _paths(root, data_setting):
    if data_setting == 1:
        return 'cox_rec_rd', 'res_cox_N'
    if data_setting == 2:
        return 'aft_rd', 'res_aft_N'
    raise ValueError(f'unsupported data_setting={data_setting}')


def visual(N, seed, data_setting, knots_setting, pxt, pxq,
           show=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    sub, res_prefix = _paths(root, data_setting)
    res_dir = os.path.join(root, 'res', sub)

    fish_file = os.path.join(
        res_dir,
        f'{res_prefix}{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}_fish.npz',
    )
    out = np.load(fish_file)
    est_r = out['est_r'].ravel()
    se_all = out['se_all'].ravel()
    p = int(out['p'].ravel()[0])
    q_q = int(out['q_q'].ravel()[0])
    q_0 = int(out['q_0'].ravel()[0])
    knots_0 = out['knots_0'].ravel()
    knots_q = out['knots_q'].ravel()
    k0 = int(out['k0'].ravel()[0])
    kq = int(out['kq'].ravel()[0])

    pxt = np.asarray(pxt, dtype=float).ravel()
    pxq = np.asarray(pxq, dtype=float).ravel()

    est_theta = est_r[p:p + q_q]
    est_alpha = est_r[p + q_q:p + q_q + q_0]
    # se_all has length p-1 + q_q + q_0 (first beta fixed at 1)
    se_theta = se_all[p - 1:p - 1 + q_q]
    se_alpha = se_all[p - 1 + q_q:p - 1 + q_q + q_0]

    Bq = spcol(knots_q, kq, pxq)
    est_q = np.exp(Bq @ est_theta)
    est_q_upper = np.exp(Bq @ (est_theta + 1.96 * se_theta))
    est_q_lower = np.exp(Bq @ (est_theta - 1.96 * se_theta))

    B0 = spcol(knots_0, k0, pxt)
    est_a = np.exp(B0 @ est_alpha)
    est_a_upper = np.exp(B0 @ (est_alpha + 1.96 * se_alpha))
    est_a_lower = np.exp(B0 @ (est_alpha - 1.96 * se_alpha))

    if show:
        import matplotlib.pyplot as plt
        if data_setting == 1:
            true_a = pxt ** 2 + 1.0
            true_q = np.ones_like(pxq)
        else:
            true_a = np.ones_like(pxt)
            true_q = 2.0 / (1.0 + pxq)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(pxt, np.log(true_a), 'b', lw=2, label='True')
        axes[0].plot(pxt, np.log(est_a / est_a[0]), 'r', lw=2, label='Estimate')
        axes[0].plot(pxt, np.log(est_a_upper / est_a[0]),
                     '--', color=(0.929, 0.694, 0.125), lw=1)
        axes[0].plot(pxt, np.log(est_a_lower / est_a[0]),
                     '--', color=(0.929, 0.694, 0.125), lw=1)
        axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$\log(\alpha(t))$')
        axes[1].plot(pxq, np.log(true_q), 'b', lw=2)
        axes[1].plot(pxq, np.log(est_q * est_a[0]), 'r', lw=2)
        axes[1].plot(pxq, np.log(est_q_upper * est_a[0]),
                     '--', color=(0.929, 0.694, 0.125), lw=1)
        axes[1].plot(pxq, np.log(est_q_lower * est_a[0]),
                     '--', color=(0.929, 0.694, 0.125), lw=1)
        axes[1].set_xlabel('u'); axes[1].set_ylabel('log(q(u))')

    return est_a, est_a_upper, est_a_lower, est_q, est_q_upper, est_q_lower
