"""Port of random effect/cox/objective_func_inf.m.

Per-subject negative-log-likelihood score.  With ``ci=True`` returns an
``(m, p+q)`` matrix of subject-level scores used by resampling-based
Fisher-information estimation.  With ``ci=False`` returns a 1-D gradient
(averaged over the ``m`` subjects) suitable for optimization.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ...common import spcol
from .baseline_hazard_func import baseline_hazard_func
from .baseline_hazard_grad_func import baseline_hazard_grad_func


def objective_func_inf(r, x, time, delta, id_vec, knots, k, ci=False):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()

    q = knots.size - k
    idx = np.where(delta == 0)[0]
    tau = time[idx]
    x_tau = x[idx]
    N, p = x.shape
    m = idx.size

    beta = np.asarray(r[:p], dtype=float)
    theta = np.asarray(r[p:p + q], dtype=float)

    multi_coef = np.exp(x_tau @ beta)

    u_all = np.unique(time)
    bin_all = np.searchsorted(u_all, time)
    B = spcol(knots, k, u_all)[bin_all]

    l1 = -float(((B @ theta + x @ beta) * delta).sum())

    u, bin_idx = np.unique(tau, return_inverse=True)
    tspan = np.concatenate([[0.0], u])
    sol = solve_ivp(
        lambda t, y: baseline_hazard_func(t, theta, knots, k),
        (tspan[0], tspan[-1]), [0.0], t_eval=tspan, method='RK45',
        rtol=1e-6, atol=1e-9,
    )
    cum_baseline_hazard = sol.y[0, 1:][bin_idx]
    l2 = float(cum_baseline_hazard @ multi_coef)
    loss = (l1 + l2) / m

    mat = np.zeros((m, N))
    for i in range(1, m + 1):
        mat[i - 1, id_vec == i] = 1.0

    grad_beta = -mat @ (x * delta[:, None]) + x_tau * (cum_baseline_hazard * multi_coef)[:, None]
    grad_theta_part1 = -mat @ (B * delta[:, None])

    sol_g = solve_ivp(
        lambda t, y: baseline_hazard_grad_func(t, theta, knots, k).ravel(),
        (tspan[0], tspan[-1]), np.zeros(q), t_eval=tspan, method='RK45',
        rtol=1e-6, atol=1e-9,
    )
    dtheta2 = sol_g.y[:, 1:].T[bin_idx]
    grad_theta = grad_theta_part1 + dtheta2 * multi_coef[:, None]
    grad = np.concatenate([grad_beta, grad_theta], axis=1)

    if not ci:
        grad = np.ones((1, m)) @ grad / m
        grad = grad.ravel()
    return loss, grad
