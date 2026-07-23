"""Port of random effect/cox/baseline_hazard_func_v.m.

Returns the subject-specific ``v^*(t) * lambda_0(t)`` evaluated at ``time``,
used by the closed-form sandwich-variance computation in ``inference_beta``.
"""
from __future__ import annotations

import numpy as np

from ...common import spcol


def baseline_hazard_func_v(time, tau, x_tau, beta, theta, knots, k):
    time = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    x_tau = np.asarray(x_tau, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()

    multi_coef = np.exp(x_tau @ beta)
    B = spcol(knots, k, time)
    lam0 = np.exp(B @ theta)

    # For each evaluation time t, compute weighted average of x_tau
    # across subjects still at risk (tau_i >= t), weighted by multi_coef.
    out = np.zeros((time.size, x_tau.shape[1]))
    for idx, t in enumerate(time):
        at_risk = (tau >= t).astype(float)
        denom = (at_risk * multi_coef).sum()
        if denom == 0.0:
            continue
        num = (at_risk * multi_coef)[:, None] * x_tau
        out[idx] = lam0[idx] * num.sum(axis=0) / denom
    return out
