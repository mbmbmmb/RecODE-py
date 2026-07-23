"""Port of random effect/aft_rec/mle.m.

Coordinate descent with a 1e-4 joint-step tolerance (tighter than the
non-RE variant's 3e-3) over ``(theta, beta)``.  No explicit warm-up
counter is replicated here — scipy's BFGS already handles the outer
iterations well within the aggregate 200-iter budget.
"""
from __future__ import annotations

import time as _time
import numpy as np
from scipy.optimize import minimize

from .objective_func_sieve import objective_func_sieve
from .objective_func_beta import objective_func_beta


def mle(x, time, delta, id_vec, knots, k, r0, forward=True):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()

    N, p = x.shape
    beta = np.asarray(r0[:p], dtype=float).copy()
    theta = np.asarray(r0[p:], dtype=float).copy()
    print(beta)

    t0 = _time.time()
    for _ in range(200):
        def fun_theta(r):
            return objective_func_sieve(
                r, x, time, delta, id_vec, beta, knots, k,
                ci=False, forward=forward,
            )

        res_t = minimize(
            fun_theta, theta, jac=True, method='BFGS',
            options={'maxiter': 30},
        )
        step_theta = float(np.max(np.abs(res_t.x - theta)))
        theta = res_t.x

        def fun_beta(rr):
            return objective_func_beta(
                rr, x, time, delta, id_vec, theta, knots, k, ci=False,
            )

        res_b = minimize(
            fun_beta, beta, jac=True, method='BFGS',
            options={'maxiter': 500},
        )
        step_beta = float(np.max(np.abs(res_b.x - beta)))
        beta = res_b.x

        if max(step_beta, step_theta) < 1e-4:
            print(beta)
            break

    est = np.concatenate([beta, theta])
    runtime = _time.time() - t0
    return est, runtime
