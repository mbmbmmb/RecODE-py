"""Port of random effect/aft_rec/objective_func_grad.m.

Combined gradient w.r.t. ``(beta, theta)`` used by the resampling-based
inference step.  ``ci=True`` returns an ``(m, p+q)`` per-subject score
matrix; ``ci=False`` returns an aggregated ``(p+q,)`` gradient.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ...common import spcol_deriv
from .forward_odesystem_func import forward_odesystem_func


def objective_func_grad(r, x, time, delta, id_vec, beta, knots, k, ci=False):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()
    beta = np.asarray(beta, dtype=float).ravel()

    N = x.shape[0]
    q = knots.size - k
    m = int(np.sum(1.0 - delta))
    theta = np.asarray(r, dtype=float).ravel()

    x_coef = np.exp(x @ beta)
    temp = time * x_coef

    u, bin_idx = np.unique(temp, return_inverse=True)
    tspan = np.concatenate([[0.0], u])
    y0 = np.zeros(q + 1)
    sol = solve_ivp(
        lambda t, y: forward_odesystem_func(y, theta, knots, k),
        (tspan[0], tspan[-1]), y0, t_eval=tspan, method='RK45',
        rtol=1e-6, atol=1e-7,
    )
    res = sol.y[:, 1:].T
    dd_theta = res[bin_idx, 1:]
    cum_hazard = res[bin_idx, 0]

    u2, bin2 = np.unique(cum_hazard, return_inverse=True)
    Bq_u, dBq_u = spcol_deriv(knots, k, u2)
    Bq = Bq_u[bin2]
    dBq = dBq_u[bin2]

    ss = dBq @ theta
    ss[delta == 0] = -1.0

    dd = np.exp(Bq @ theta)[:, None] * (temp[:, None] * x)

    if ci:
        mat = np.zeros((m, N))
        for i in range(1, m + 1):
            mat[i - 1, id_vec == i] = 1.0
        grad_beta = -mat @ (x * delta[:, None] + ss[:, None] * dd)
        grad_sieve = -mat @ (Bq * delta[:, None] + ss[:, None] * dd_theta)
    else:
        grad_beta = -(delta @ x + ss @ dd)
        grad_sieve = -(delta @ Bq + ss @ dd_theta)
    return np.concatenate([grad_beta, grad_sieve], axis=-1)
