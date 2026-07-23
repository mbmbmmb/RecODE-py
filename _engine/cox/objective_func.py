"""Port of cox/objective_func.m.

Negative log-likelihood (and gradient) for the Cox-type recurrent-event model
with B-spline log-baseline hazard::

    lambda(t | x) = exp(B(t) @ theta) * exp(x @ beta)

``params_vec = [beta; theta]``.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..common import spcol, unique_sort_index
from .baseline_hazard_func import baseline_hazard_func
from .baseline_hazard_grad_func import baseline_hazard_grad_func


def objective_func(params_vec, x, time, delta, knots, spline_order):
    knots = np.asarray(knots, dtype=float).ravel()
    params_vec = np.asarray(params_vec, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()

    num_basis = len(knots) - spline_order
    num_covariates = x.shape[1]

    beta = params_vec[:num_covariates]
    theta = params_vec[num_covariates:num_covariates + num_basis]

    censored_idx = np.where(delta == 0)[0]
    observation_times = time[censored_idx]
    x_censored = x[censored_idx]
    num_subjects = len(censored_idx)

    # --- log-hazard sum over events ---
    unique_times, time_to_unique = unique_sort_index(time)
    B_unique = spcol(knots, spline_order, unique_times)
    B_full = B_unique[time_to_unique, :]
    log_hazard_sum = (B_full @ theta + x @ beta) @ delta

    # --- integrated hazard: solve ODE for Lambda_0(t) = int_0^t lambda_0(s) ds ---
    linear_pred = np.exp(x_censored @ beta)
    unique_obs, obs_to_unique = unique_sort_index(observation_times)
    t0 = 1e-12
    tspan = np.concatenate([[t0], unique_obs])

    def rhs_scalar(t, y):
        return baseline_hazard_func(np.array([t]), theta, knots, spline_order)

    sol = solve_ivp(
        rhs_scalar, (tspan[0], tspan[-1]), [0.0],
        t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-8,
    )
    cum_unique = sol.y[0][1:]  # drop the initial t0 evaluation
    cum_obs = cum_unique[obs_to_unique]
    integrated_hazard_sum = cum_obs @ linear_pred

    neg_log_likelihood = (-log_hazard_sum + integrated_hazard_sum) / num_subjects

    # --- gradient wrt beta ---
    grad_beta = -x.T @ delta + x_censored.T @ (cum_obs * linear_pred)

    # --- gradient wrt theta: integrate dlambda/dtheta along time ---
    def rhs_grad(t, y):
        return baseline_hazard_grad_func(np.array([t]), theta, knots, spline_order).ravel()

    sol_g = solve_ivp(
        rhs_grad, (tspan[0], tspan[-1]), np.zeros(num_basis),
        t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-8,
    )
    dLambda_unique = sol_g.y[:, 1:].T  # (len(unique_obs), q)
    dLambda = dLambda_unique[obs_to_unique]
    grad_theta = -(delta @ B_full) + linear_pred @ dLambda

    grad = np.concatenate([grad_beta, grad_theta]) / num_subjects
    return float(neg_log_likelihood), grad
