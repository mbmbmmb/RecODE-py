"""Port of cox/objective_func_inf.m.

Returns the per-subject score (gradient w.r.t. r) used to form the empirical
Fisher information. Shape of the returned ``grad`` is ``(m, p + q)`` where
``m`` is the number of subjects.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..common import spcol, unique_sort_index
from .baseline_hazard_func import baseline_hazard_func
from .baseline_hazard_grad_func import baseline_hazard_grad_func


def objective_func_inf(r, x, time, delta, id_vec, knots, k):
    r = np.asarray(r, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()
    knots = np.asarray(knots, dtype=float).ravel()

    q = len(knots) - k
    N, p = x.shape

    censored_idx = np.where(delta == 0)[0]
    tau = time[censored_idx]
    x_tau = x[censored_idx]
    m = len(censored_idx)

    beta = r[:p]
    theta = r[p:p + q]

    multi_coef = np.exp(x_tau @ beta)

    u_time, bin_time = unique_sort_index(time)
    pre_B0 = spcol(knots, k, u_time)
    B = pre_B0[bin_time]

    l1 = -(B @ theta + x @ beta) @ delta

    u_tau, bin_tau = unique_sort_index(tau)
    tspan = np.concatenate([[1e-12], u_tau])

    def rhs_scalar(t, y):
        return baseline_hazard_func(np.array([t]), theta, knots, k)

    sol = solve_ivp(
        rhs_scalar, (tspan[0], tspan[-1]), [0.0],
        t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-8,
    )
    cum_unique = sol.y[0][1:]
    cum_baseline_hazard = cum_unique[bin_tau]

    l2 = cum_baseline_hazard @ multi_coef
    loss = (l1 + l2) / m

    # Build the per-subject selection matrix: mat[i, j] = 1 iff id[j] == i+1.
    # In MATLAB the row index i corresponds to subject id i; here we treat the
    # m censoring rows (one per subject) as the subject ordering.
    subject_ids = id_vec[censored_idx]
    mat = np.zeros((m, N))
    for row_idx, subj in enumerate(subject_ids):
        mat[row_idx, id_vec == subj] = 1.0

    grad_beta = -mat @ (x * delta[:, None]) + x_tau * (cum_baseline_hazard * multi_coef)[:, None]

    dtheta1 = -mat @ (B * delta[:, None])

    def rhs_grad(t, y):
        return baseline_hazard_grad_func(np.array([t]), theta, knots, k).ravel()

    y0 = np.zeros(q)
    tspan2 = np.concatenate([[1e-8], u_tau])
    sol_g = solve_ivp(
        rhs_grad, (tspan2[0], tspan2[-1]), y0,
        t_eval=tspan2, method='RK45', rtol=1e-6, atol=1e-8,
    )
    dtheta2_unique = sol_g.y[:, 1:].T
    dtheta2 = dtheta2_unique[bin_tau]

    grad_theta = dtheta1 + dtheta2 * multi_coef[:, None]

    grad = np.concatenate([grad_beta, grad_theta], axis=1)
    return loss, grad
