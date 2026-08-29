"""Port of random effect/ltm/objective_func_sieve.m.

Unlike the non-RE variant, the RE file has no ``ci`` / ``id`` branch —
``mle`` only needs the aggregated gradient.  Per-subject scores live in
``inference_objective_func_sieve`` instead.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ...common import spcol, spcol_deriv, unique_sort_index, solve_ode, ode_guard
from .time_transform_func import time_transform_func
from .time_transform_grad_func import time_transform_grad_func
from .forward_odesystem_func import forward_odesystem_func


@ode_guard
def objective_func_sieve(r, x, time, delta, beta,
                         knots_0, knots_q, k0, kq):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    knots_0 = np.asarray(knots_0, dtype=float).ravel()
    knots_q = np.asarray(knots_q, dtype=float).ravel()
    beta_free = np.asarray(beta, dtype=float).ravel()

    m = int(np.sum(1 - delta))
    q_0 = len(knots_0) - k0
    q_q = len(knots_q) - kq

    beta = np.concatenate([[1.0], beta_free])
    theta = r[:q_q]
    alpha = r[q_q:]

    multi_coef = np.exp(x @ beta)

    u_time, bin_time = unique_sort_index(time)
    tspan = np.concatenate([[0.0], u_time])

    def rhs_alpha(t, y):
        return time_transform_func(t, alpha, knots_0, k0)

    sol_a = solve_ode(rhs_alpha, (tspan[0], tspan[-1]), [0.0],
                      t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-7)
    # exp(.) > 0 makes the true integral non-decreasing and non-negative, but
    # at a trial alpha that blows it up to ~1e166 RK45 can still return a tiny
    # negative value at the earliest time. That sign then puts t_eval outside
    # t_span in the transformed-time solve below and raises. Clamping is exact
    # for the true solution and turns the crash into a finite objective.
    int_alpha = np.maximum(sol_a.y[0][1:][bin_time], 0.0)
    time_transform = int_alpha * multi_coef

    u_t, bin_t = unique_sort_index(time_transform)
    # Prepend the origin only when it is not already the first evaluation
    # point: clamping int_alpha at 0 can make the earliest transformed time
    # exactly 0, and a duplicated 0.0 makes t_eval non-increasing.
    _skip = 1 if u_t[0] > 0.0 else 0
    tspan_t = np.concatenate([[0.0], u_t]) if _skip else u_t

    def rhs_fwd(t, y):
        return forward_odesystem_func(y, theta, knots_q, kq)

    sol_f = solve_ode(rhs_fwd, (tspan_t[0], tspan_t[-1]),
                      np.zeros(q_q + 1), t_eval=tspan_t, method='RK45',
                      rtol=1e-6, atol=1e-7)
    res = sol_f.y[:, _skip:].T
    cum_hazard = res[bin_t, 0]
    dd_theta = res[bin_t, 1:]

    u_c, bin_c = unique_sort_index(cum_hazard)
    Bq_u, dBq_u = spcol_deriv(knots_q, kq, u_c)
    Bq = Bq_u[bin_c]
    dBq = dBq_u[bin_c]

    B0 = spcol(knots_0, k0, u_time)[bin_time]

    l1 = -(Bq @ theta + x @ beta + B0 @ alpha) @ delta
    l2 = float(np.sum(cum_hazard * (1 - delta)))
    loss = (l1 + l2) / m

    def rhs_dalpha(t, y):
        return time_transform_grad_func(t, alpha, knots_0, k0).ravel()

    tspan_a = np.concatenate([[1e-8], u_time])
    sol_da = solve_ode(rhs_dalpha, (tspan_a[0], tspan_a[-1]),
                       np.zeros(q_0), t_eval=tspan_a, method='RK45',
                       rtol=1e-6, atol=1e-7)
    int_dalpha = sol_da.y[:, 1:].T[bin_time]
    dd_alpha = (np.exp(Bq @ theta) * multi_coef)[:, None] * int_dalpha

    ss_theta = dBq @ theta
    ss_theta = np.where(delta == 0, -1.0, ss_theta)

    grad_theta = -(delta @ Bq + ss_theta @ dd_theta)
    grad_alpha = -(delta @ B0 + ss_theta @ dd_alpha)
    grad = np.concatenate([grad_theta, grad_alpha]) / m
    return float(loss), grad
