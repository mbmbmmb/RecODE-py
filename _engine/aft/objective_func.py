"""Port of aft/objective_func.m (Cox-type fit used to seed the AFT scheme)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..common import spcol, unique_sort_index
from .baseline_hazard_func import baseline_hazard_func
from .baseline_hazard_grad_func import baseline_hazard_grad_func


def objective_func(r, x, time, delta, knots, k):
    r = np.asarray(r, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()

    q = len(knots) - k
    p = x.shape[1]

    censored_idx = np.where(delta == 0)[0]
    tau = time[censored_idx]
    x_tau = x[censored_idx]
    m = len(censored_idx)

    beta = r[:p]
    theta = r[p:p + q]

    multi_coef = np.exp(x_tau @ beta)

    u_time, bin_time = unique_sort_index(time)
    B = spcol(knots, k, u_time)[bin_time]
    l1 = -(B @ theta + x @ beta) @ delta

    u_tau, bin_tau = unique_sort_index(tau)
    tspan = np.concatenate([[1e-12], u_tau])

    def rhs_scalar(t, y):
        return baseline_hazard_func(np.array([t]), theta, knots, k)

    sol = solve_ivp(rhs_scalar, (tspan[0], tspan[-1]), [0.0],
                    t_eval=tspan, method='RK45', rtol=1e-3, atol=1e-6)
    cum_baseline = sol.y[0][1:][bin_tau]

    l2 = cum_baseline @ multi_coef
    loss = (l1 + l2) / m

    def rhs_grad(t, y):
        return baseline_hazard_grad_func(np.array([t]), theta, knots, k).ravel()

    tspan2 = np.concatenate([[1e-8], u_tau])
    sol_g = solve_ivp(rhs_grad, (tspan2[0], tspan2[-1]), np.zeros(q),
                      t_eval=tspan2, method='RK45', rtol=1e-3, atol=1e-6)
    dtheta2 = sol_g.y[:, 1:].T[bin_tau]

    grad_beta = -(x.T @ delta) + x_tau.T @ (cum_baseline * multi_coef)
    grad_theta = -(B.T @ delta) + dtheta2.T @ multi_coef
    grad = np.concatenate([grad_beta, grad_theta]) / m
    return float(loss), grad
