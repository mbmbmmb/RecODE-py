"""Port of random effect/ltm/objective_func_beta.m.

No ``ci`` branch in the RE version — ``mle`` only needs the aggregated
gradient, resampling uses ``inference_objective_func_sieve`` instead.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ...common import spcol, spcol_deriv, unique_sort_index
from .time_transform_func import time_transform_func
from .hazard_ode_func import hazard_ode_func


def objective_func_beta(r, x, time, delta, theta, alpha,
                        knots_0, knots_q, k0, kq):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    knots_0 = np.asarray(knots_0, dtype=float).ravel()
    knots_q = np.asarray(knots_q, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()

    N, p = x.shape
    m = int(np.sum(1 - delta))
    beta = np.concatenate([[1.0], np.asarray(r, dtype=float).ravel()])

    multi_coef = np.exp(x @ beta)

    u_time, bin_time = unique_sort_index(time)
    tspan = np.concatenate([[0.0], u_time])

    def rhs_a(t, y):
        return time_transform_func(t, alpha, knots_0, k0)

    sol_a = solve_ivp(rhs_a, (tspan[0], tspan[-1]), [0.0],
                      t_eval=tspan, method='RK45', rtol=1e-9, atol=1e-9)
    int_alpha = sol_a.y[0][1:][bin_time]
    time_transform = int_alpha * multi_coef

    u_t, bin_t = unique_sort_index(time_transform)
    tspan_t = np.concatenate([[0.0], u_t])

    def rhs_c(t, y):
        return hazard_ode_func(y, theta, knots_q, kq)

    sol_c = solve_ivp(rhs_c, (tspan_t[0], tspan_t[-1]), [0.0],
                      t_eval=tspan_t, method='RK45', rtol=1e-6, atol=1e-7)
    cum_hazard = sol_c.y[0][1:][bin_t]

    u_c, bin_c = unique_sort_index(cum_hazard)
    Bq_u, dBq_u = spcol_deriv(knots_q, kq, u_c)
    Bq = Bq_u[bin_c]
    dBq = dBq_u[bin_c]

    B0 = spcol(knots_0, k0, u_time)[bin_time]

    ss = dBq @ theta
    ss = np.where(delta == 0, -1.0, ss)

    l1 = -(Bq @ theta + x @ beta + B0 @ alpha) @ delta
    l2 = float(np.sum(cum_hazard * (1 - delta)))
    loss = (l1 + l2) / m

    x_free = x[:, 1:]
    dd = (np.exp(Bq @ theta) * time_transform)[:, None] * x_free

    grad = -(delta @ x_free + ss @ dd) / m
    return float(loss), grad
