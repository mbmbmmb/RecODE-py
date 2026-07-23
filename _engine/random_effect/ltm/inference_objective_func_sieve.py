"""Port of random effect/ltm/inference_objective_func_sieve.m.

Joint gradient w.r.t. ``(beta_free, theta, alpha)`` used by the
resampling-based inference.  ``ci=True`` returns a per-subject score
matrix of shape ``(m, p-1+q_q+q_0)``; ``ci=False`` returns the aggregated
``(p-1+q_q+q_0,)`` gradient — neither case divides by ``m`` because the
caller handles the ``1/sqrt(N)`` scaling explicitly.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ...common import spcol, spcol_deriv, unique_sort_index
from .time_transform_func import time_transform_func
from .time_transform_grad_func import time_transform_grad_func
from .forward_odesystem_func import forward_odesystem_func
from .hazard_ode_func import hazard_ode_func


def inference_objective_func_sieve(r, x, time, delta, id_vec, beta,
                                   knots_0, knots_q, k0, kq, ci):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()
    knots_0 = np.asarray(knots_0, dtype=float).ravel()
    knots_q = np.asarray(knots_q, dtype=float).ravel()
    beta_free = np.asarray(beta, dtype=float).ravel()

    NN, p = x.shape
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

    sol_a = solve_ivp(rhs_alpha, (tspan[0], tspan[-1]), [0.0],
                      t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-7)
    int_alpha = sol_a.y[0][1:][bin_time]
    time_transform = int_alpha * multi_coef

    # Forward-sensitivity ODE for theta gradient
    v_t, bin_tt = unique_sort_index(time_transform)
    tspan_v = np.concatenate([[0.0], v_t])

    def rhs_fwd(t, y):
        return forward_odesystem_func(y, theta, knots_q, kq)

    sol_f = solve_ivp(rhs_fwd, (tspan_v[0], tspan_v[-1]),
                      np.zeros(q_q + 1), t_eval=tspan_v, method='RK45',
                      rtol=1e-6, atol=1e-7)
    res_fwd = sol_f.y[:, 1:].T
    dd_theta = res_fwd[bin_tt, 1:]

    # Separate cum_hazard ODE (used for spline evaluation)
    u_t2, bin_t2 = unique_sort_index(time_transform)
    tspan_t2 = np.concatenate([[0.0], u_t2])

    def rhs_c(t, y):
        return hazard_ode_func(y, theta, knots_q, kq)

    sol_c = solve_ivp(rhs_c, (tspan_t2[0], tspan_t2[-1]), [0.0],
                      t_eval=tspan_t2, method='RK45', rtol=1e-6, atol=1e-7)
    cum_hazard = sol_c.y[0][1:][bin_t2]

    v_c, bin_c = unique_sort_index(cum_hazard)
    Bq_u, dBq_u = spcol_deriv(knots_q, kq, v_c)
    Bq = Bq_u[bin_c]
    dBq = dBq_u[bin_c]

    B0 = spcol(knots_0, k0, u_time)[bin_time]

    # d mu / d alpha
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

    x_free = x[:, 1:]
    dd_beta = (np.exp(Bq @ theta) * time_transform)[:, None] * x_free

    if ci:
        # Per-subject score residuals via sparse scatter: each row i in the
        # output corresponds to patient id (i+1). Replaces the literal port's
        # m x NN dense indicator matrix, which OOMs at cohort scale.
        # id_vec is assumed to be 1..N consecutive (api.fit / fit_ltm enforce).
        N_subj = int(id_vec.max())
        score_beta_rows  = -(x_free * delta[:, None]
                             + ss_theta[:, None] * dd_beta)
        score_theta_rows = -(Bq * delta[:, None]
                             + ss_theta[:, None] * dd_theta)
        score_alpha_rows = -(B0 * delta[:, None]
                             + ss_theta[:, None] * dd_alpha)
        grad_beta  = np.zeros((N_subj, score_beta_rows.shape[1]))
        grad_theta = np.zeros((N_subj, score_theta_rows.shape[1]))
        grad_alpha = np.zeros((N_subj, score_alpha_rows.shape[1]))
        np.add.at(grad_beta,  id_vec - 1, score_beta_rows)
        np.add.at(grad_theta, id_vec - 1, score_theta_rows)
        np.add.at(grad_alpha, id_vec - 1, score_alpha_rows)
        grad = np.concatenate([grad_beta, grad_theta, grad_alpha], axis=1)
    else:
        grad_beta = -(delta @ x_free + ss_theta @ dd_beta)
        grad_theta = -(delta @ Bq + ss_theta @ dd_theta)
        grad_alpha = -(delta @ B0 + ss_theta @ dd_alpha)
        grad = np.concatenate([grad_beta, grad_theta, grad_alpha])
    return grad
