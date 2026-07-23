"""Port of npmle/objective_func.m.

Negative log likelihood under the G-transform recurrent-event model
``Lambda(t|x, beta) = G(Lambda_0(t) * exp(x @ beta))``, with the baseline
hazard ``lambda_0(t) = exp(B(t) @ theta)``.  Returns either the aggregate
loss + gradient (for optimization) or a per-censored-subject score matrix
(for Fisher-information SE calculations).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..common import spcol
from .baseline_hazard_func import baseline_hazard_func
from .baseline_hazard_grad_func import baseline_hazard_grad_func
from .Gtransform import Gtransform


def _unique_with_bins(time):
    u = np.unique(time)
    bin_idx = np.searchsorted(u, time)
    return u, bin_idx


def objective_func(r, x, time, delta, id_vec, q, knots, k, rho1, r1, ci=False):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()

    N, p = x.shape
    m = int(np.sum(1.0 - delta))
    beta = np.asarray(r[:p], dtype=float)
    theta = np.asarray(r[p:p + q], dtype=float)

    multi_coef = np.exp(x @ beta)

    u, bin_idx = _unique_with_bins(time)

    tspan = np.concatenate([[1e-12], u])
    sol = solve_ivp(
        lambda t, y: baseline_hazard_func(t, theta, knots, k),
        (tspan[0], tspan[-1]), [0.0], t_eval=tspan, method='RK45',
        rtol=1e-6, atol=1e-9,
    )
    cum_baseline_hazard = sol.y[0, 1:][bin_idx]

    temp = cum_baseline_hazard * multi_coef
    cumhaz, dcumhaz = Gtransform(temp, rho1, r1)

    B = spcol(knots, k, u)[bin_idx]

    l1 = -float(((np.log(dcumhaz) + B @ theta + x @ beta) * delta).sum())
    l2 = float((cumhaz * (1.0 - delta)).sum())
    loss = (l1 + l2) / m

    # dtheta: integral_0^t lambda_0(s) * B(s) ds via augmented ODE
    def _dtheta_rhs(t, y):
        return baseline_hazard_grad_func(t, theta, knots, k).ravel()

    tspan_g = np.concatenate([[1e-8], u])
    sol_g = solve_ivp(
        _dtheta_rhs, (tspan_g[0], tspan_g[-1]), np.zeros(q),
        t_eval=tspan_g, method='RK45', rtol=1e-6, atol=1e-9,
    )
    dtheta = sol_g.y[:, 1:].T[bin_idx]

    if rho1 > 0:
        ss = (rho1 - 1.0) / (rho1 * cumhaz + 1.0) * delta + (delta - 1.0)
    else:  # rho1 == 0
        ss = -r1 * delta + (delta - 1.0)

    dd_beta = x * (dcumhaz * temp)[:, None]
    dd_theta = dtheta * (dcumhaz * multi_coef)[:, None]

    if ci:
        # MATLAB: for i=1:m, mat(i, id==i) = 1.  m equals the number of
        # censored rows, which matches the number of subjects since each
        # subject contributes exactly one censoring row.  Group rows that
        # belong to the same subject id.
        mat = np.zeros((m, x.shape[0]))
        for i in range(1, m + 1):
            mat[i - 1, id_vec == i] = 1.0
        grad_beta = -mat @ (x * delta[:, None] + ss[:, None] * dd_beta)
        grad_theta = -mat @ (B * delta[:, None] + ss[:, None] * dd_theta)
        grad = np.concatenate([grad_beta, grad_theta], axis=1)
        return loss, grad

    grad_beta = -(delta @ x + ss @ dd_beta)
    grad_theta = -(delta @ B + ss @ dd_theta)
    grad = np.concatenate([grad_beta, grad_theta]) / m
    return loss, grad
