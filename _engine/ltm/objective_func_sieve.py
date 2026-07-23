"""Port of ltm/objective_func_sieve.m.

Likelihood and gradient as a function of ``(theta, alpha)`` — the parameters
of the hazard function q(.) and the time-transform alpha(.) — with ``beta``
held fixed and implicitly prepended with 1 (identifiability constraint).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..common import spcol, spcol_deriv, unique_sort_index
from .time_transform_func import time_transform_func
from .time_transform_grad_func import time_transform_grad_func
from .forward_odesystem_func import forward_odesystem_func


def objective_func_sieve(r, x, time, delta, id_vec, beta,
                         knots_0, knots_q, k0, kq, ci=False):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()
    knots_0 = np.asarray(knots_0, dtype=float).ravel()
    knots_q = np.asarray(knots_q, dtype=float).ravel()
    beta_free = np.asarray(beta, dtype=float).ravel()

    N = x.shape[0]
    m = int(np.sum(1 - delta))
    q_0 = len(knots_0) - k0
    q_q = len(knots_q) - kq

    beta = np.concatenate([[1.0], beta_free])
    theta = r[:q_q]
    alpha = r[q_q:]

    multi_coef = np.exp(x @ beta)

    # time transform: int_0^t exp(B0(s) alpha) ds
    u_time, bin_time = unique_sort_index(time)
    tspan = np.concatenate([[0.0], u_time])

    def rhs_alpha(t, y):
        return time_transform_func(t, alpha, knots_0, k0)

    sol_a = solve_ivp(rhs_alpha, (tspan[0], tspan[-1]), [0.0],
                      t_eval=tspan, method='RK45', rtol=1e-9, atol=1e-9)
    int_alpha = sol_a.y[0][1:][bin_time]
    time_transform = int_alpha * multi_coef

    # cumulative hazard ODE in transformed time: d y1/dt = exp(Bq(y1) theta),
    # plus forward sensitivity for theta.
    u_t, bin_t = unique_sort_index(time_transform)
    tspan_t = np.concatenate([[0.0], u_t])

    def rhs_fwd(t, y):
        return forward_odesystem_func(y, theta, knots_q, kq)

    sol_f = solve_ivp(rhs_fwd, (tspan_t[0], tspan_t[-1]),
                      np.zeros(q_q + 1), t_eval=tspan_t, method='RK45',
                      rtol=1e-6, atol=1e-7)
    res = sol_f.y[:, 1:].T
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

    # gradient wrt alpha via sensitivity of int_alpha
    def rhs_dalpha(t, y):
        return time_transform_grad_func(t, alpha, knots_0, k0).ravel()

    tspan_a = np.concatenate([[1e-8], u_time])
    sol_da = solve_ivp(rhs_dalpha, (tspan_a[0], tspan_a[-1]),
                       np.zeros(q_0), t_eval=tspan_a, method='RK45',
                       rtol=1e-6, atol=1e-7)
    int_dalpha = sol_da.y[:, 1:].T[bin_time]
    dd_alpha = (np.exp(Bq @ theta) * multi_coef)[:, None] * int_dalpha

    ss_theta = dBq @ theta
    ss_theta = np.where(delta == 0, -1.0, ss_theta)

    if ci:
        censored_idx = np.where(delta == 0)[0]
        subject_ids = id_vec[censored_idx]
        mat = np.zeros((m, N))
        for row_idx, subj in enumerate(subject_ids):
            mat[row_idx, id_vec == subj] = 1.0
        grad_theta = -mat @ (Bq * delta[:, None] + ss_theta[:, None] * dd_theta)
        grad_alpha = -mat @ (B0 * delta[:, None] + ss_theta[:, None] * dd_alpha)
        grad = np.concatenate([grad_theta, grad_alpha], axis=1)
    else:
        grad_theta = -(delta @ Bq + ss_theta @ dd_theta)
        grad_alpha = -(delta @ B0 + ss_theta @ dd_alpha)
        grad = np.concatenate([grad_theta, grad_alpha]) / m

    return float(loss), grad
