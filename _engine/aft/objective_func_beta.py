"""Port of aft/objective_func_beta.m.

Negative log-likelihood as a function of regression coefficients ``beta``,
with spline parameters ``theta`` held fixed. If ``ci=True`` returns the
per-subject score matrix (shape ``(m, p)``).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..common import spcol_deriv, unique_sort_index
from .hazard_ode_func import hazard_ode_func


def objective_func_beta(r, x, time, delta, id_vec, theta, knots, k, ci=False):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    beta = np.asarray(r, dtype=float).ravel()

    N = x.shape[0]
    m = int(np.sum(1 - delta))

    x_coef = np.exp(x @ beta)
    temp = time * x_coef

    u, bin_idx = unique_sort_index(temp)
    tspan = np.concatenate([[0.0], u])

    def rhs(t, y):
        return hazard_ode_func(y, theta, knots, k)

    sol = solve_ivp(rhs, (tspan[0], tspan[-1]), [0.0],
                    t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-7)
    cum_hazard = sol.y[0][1:][bin_idx]

    u2, bin_idx2 = unique_sort_index(cum_hazard)
    Bq_u, dBq_u = spcol_deriv(knots, k, u2)
    Bq = Bq_u[bin_idx2]
    dBq = dBq_u[bin_idx2]

    ss = dBq @ theta
    ss = np.where(delta == 0, -1.0, ss)

    l1 = -((Bq @ theta) + x @ beta) @ delta
    l2 = cum_hazard @ (1 - delta)
    loss = (l1 + l2) / m

    dd = (np.exp(Bq @ theta) * temp)[:, None] * x

    if ci:
        censored_idx = np.where(delta == 0)[0]
        subject_ids = id_vec[censored_idx]
        mat = np.zeros((m, N))
        for row_idx, subj in enumerate(subject_ids):
            mat[row_idx, id_vec == subj] = 1.0
        grad = -mat @ (x * delta[:, None] + ss[:, None] * dd)
    else:
        grad = -(delta @ x + ss @ dd) / m

    return float(loss), grad
