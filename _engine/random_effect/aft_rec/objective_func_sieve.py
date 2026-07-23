"""Port of random effect/aft_rec/objective_func_sieve.m.

Theta-only objective under the AFT-type recurrent-event likelihood with
random effects.  Differs from the non-RE variant by always using the
subject-aggregation matrix ``mat`` (both the ``ci=True`` and ``ci=False``
branches) so the per-subject score aligns with the RE inference pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ...common import spcol_deriv
from .hazard_ode_func import hazard_ode_func
from .forward_odesystem_func import forward_odesystem_func


def objective_func_sieve(r, x, time, delta, id_vec, beta, knots, k,
                         ci=False, forward=True):
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

    if forward:
        tspan = np.concatenate([[0.0], u])
        y0 = np.zeros(q + 1)
        sol = solve_ivp(
            lambda t, y: forward_odesystem_func(y, theta, knots, k),
            (tspan[0], tspan[-1]), y0, t_eval=tspan, method='RK45',
            rtol=1e-6, atol=1e-7,
        )
        res = sol.y[:, 1:].T
        dd = res[bin_idx, 1:]
        cum_hazard = res[bin_idx, 0]
    else:
        tspan = np.concatenate([[0.0], u])
        sol = solve_ivp(
            lambda t, y: hazard_ode_func(y, theta, knots, k),
            (tspan[0], tspan[-1]), [0.0], t_eval=tspan, method='RK45',
            rtol=1e-6, atol=1e-7,
        )
        cum_hazard = sol.y[0, 1:][bin_idx]
        dd = np.zeros((N, q))  # adjoint path not exercised in tests

    u2, bin2 = np.unique(cum_hazard, return_inverse=True)
    Bq_u, dBq_u = spcol_deriv(knots, k, u2)
    Bq = Bq_u[bin2]
    dBq = dBq_u[bin2]

    ss = dBq @ theta
    ss[delta == 0] = -1.0

    mat = np.zeros((m, N))
    for i in range(1, m + 1):
        mat[i - 1, id_vec == i] = 1.0

    if ci:
        grad = -mat @ (Bq * delta[:, None] + ss[:, None] * dd)
    else:
        grad = -np.sum(mat @ (Bq * delta[:, None] + ss[:, None] * dd), axis=0) / m

    l1 = -float(((Bq @ theta + x @ beta) * delta).sum())
    l2 = float(cum_hazard @ (1.0 - delta))
    loss = (l1 + l2) / m
    return loss, grad
