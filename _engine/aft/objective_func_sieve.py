"""Port of aft/objective_func_sieve.m.

Negative log-likelihood as a function of the spline coefficients ``theta``,
with ``beta`` held fixed. Gradient is computed either by augmenting the ODE
(``forward=True``, matching MATLAB's primary path) or by a per-subject
adjoint solve. When ``ci=True`` the returned gradient is the per-subject
score matrix (shape ``(m, q)``).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..common import spcol, spcol_deriv, unique_sort_index
from .forward_odesystem_func import forward_odesystem_func
from .hazard_ode_func import hazard_ode_func
from .augode_func import augode_func


def objective_func_sieve(r, x, time, delta, id_vec, beta, knots, k,
                         ci=False, forward=True):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    theta = np.asarray(r, dtype=float).ravel()

    N = x.shape[0]
    q = len(knots) - k
    m = int(np.sum(1 - delta))

    x_coef = np.exp(x @ beta)
    temp = time * x_coef

    u, bin_idx = unique_sort_index(temp)

    if forward:
        tspan = np.concatenate([[0.0], u])
        def rhs(t, y):
            return forward_odesystem_func(y, theta, knots, k)
        sol = solve_ivp(rhs, (tspan[0], tspan[-1]), np.zeros(q + 1),
                        t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-7)
        res = sol.y[:, 1:].T
        cum_hazard = res[bin_idx, 0]
        dd = res[bin_idx, 1:]
    else:
        tspan = np.concatenate([[0.0], u])
        def rhs_cum(t, y):
            return hazard_ode_func(y, theta, knots, k)
        sol = solve_ivp(rhs_cum, (tspan[0], tspan[-1]), [0.0],
                        t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-7)
        cum_hazard = sol.y[0][1:][bin_idx]
        dd = np.zeros((N, q))
        for i in range(N):
            y0 = np.concatenate([[cum_hazard[i], 1.0], np.zeros(q)])
            tspan_i = [time[i], 0.0]
            def rhs_aug(t, y, xc=x_coef[i]):
                return augode_func(y, theta, xc, knots, k)
            sol_i = solve_ivp(rhs_aug, (tspan_i[0], tspan_i[1]), y0,
                              method='RK45', rtol=1e-6, atol=1e-7)
            dd[i] = sol_i.y[2:, -1]

    u2, bin_idx2 = unique_sort_index(cum_hazard)
    Bq_u, dBq_u = spcol_deriv(knots, k, u2)
    Bq = Bq_u[bin_idx2]
    dBq = dBq_u[bin_idx2]

    ss = np.sum(theta * dBq, axis=1)
    ss = np.where(delta == 0, -1.0, ss)

    l1 = -((Bq @ theta) + x @ beta) @ delta
    l2 = cum_hazard @ (1 - delta)
    loss = (l1 + l2) / m

    if ci:
        # Per-subject aggregation: rows index the m censored rows (one per subject).
        censored_idx = np.where(delta == 0)[0]
        subject_ids = id_vec[censored_idx]
        mat = np.zeros((m, N))
        for row_idx, subj in enumerate(subject_ids):
            mat[row_idx, id_vec == subj] = 1.0
        grad = -mat @ (Bq * delta[:, None] + ss[:, None] * dd)
    else:
        grad = -(delta @ Bq + ss @ dd) / m

    return float(loss), grad
