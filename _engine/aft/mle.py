"""Port of aft/mle.m.

Coordinate gradient descent: alternates between minimising the sieve
objective over ``theta`` and the beta objective over ``beta``, terminating
when both step sizes drop below the 3e-3 tolerance used in the MATLAB code.
"""
from __future__ import annotations

import time as _time
import numpy as np
from scipy.optimize import minimize

from .objective_func_sieve import objective_func_sieve
from .objective_func_beta import objective_func_beta


def mle(x, time, delta, id_vec, knots, k, r0, forward=True):
    x = np.asarray(x, dtype=float)
    N, p = x.shape
    r0 = np.asarray(r0, dtype=float).ravel()

    beta = r0[:p].copy()
    theta = r0[p:].copy()
    print(beta)
    n_warmup = 2

    t0 = _time.time()
    for i in range(200):
        if i < n_warmup:
            # Warm-up: cap function evaluations at ~20 for theta update.
            maxf_theta = 20
            tol_step_theta = 1e-3
        else:
            maxf_theta = 30
            tol_step_theta = max(1e-2, (100.0 / N) ** 2)

        def fun_theta(th):
            return objective_func_sieve(th, x, time, delta, id_vec, beta,
                                        knots, k, ci=False, forward=forward)

        res_theta = minimize(
            fun_theta, theta, jac=True, method='BFGS',
            options={'maxiter': maxf_theta, 'gtol': 1e-4,
                     'xrtol': tol_step_theta},
        )
        est_theta = res_theta.x
        step_theta = float(np.max(np.abs(est_theta - theta)))
        theta = est_theta

        def fun_beta(b):
            return objective_func_beta(b, x, time, delta, id_vec, theta,
                                       knots, k, ci=False)

        res_beta = minimize(
            fun_beta, beta, jac=True, method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-4},
        )
        est_beta = res_beta.x
        step_beta = float(np.max(np.abs(est_beta - beta)))
        beta = est_beta

        print(beta)
        if max(step_beta, step_theta) < 3e-3:
            break

    est = np.concatenate([beta, theta])
    runtime = _time.time() - t0
    return est, runtime
