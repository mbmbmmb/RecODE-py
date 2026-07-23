"""Port of ltm/mle.m.

Coordinate descent over ``(theta, alpha)`` (the sieve part, subject to
``alpha(2) = 1`` via a linear equality constraint) and the free part of
``beta`` (first element fixed at 1 for identifiability).

``succ_ind`` flips to 0 when any step raises — matching MATLAB's try/catch.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, LinearConstraint

from ..common import augknt, spcol
from .cox_rec import cox_rec
from .objective_func_sieve import objective_func_sieve
from .objective_func_beta import objective_func_beta


def mle(x, time, delta, id_vec, knots_setting):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    id_vec = np.asarray(id_vec, dtype=int).ravel()

    N, p = x.shape
    k0 = 3; kq = 3
    l0 = int(np.ceil(len(np.unique(time)) ** (1.0 / 5.0)))
    lq = int(np.ceil(N ** (1.0 / 5.0)))
    temp1 = time[delta > 0]
    temp2, beta_init = cox_rec(x, time, delta)

    if knots_setting == 'K1':
        knots_0 = augknt(np.linspace(0, float(np.max(time)), l0 + 1), k0)
        knots_q = augknt(np.linspace(0, 2 * float(np.max(temp2)), lq + 1), kq)
    elif knots_setting == 'K2':
        knots_0 = augknt(np.linspace(0, float(np.max(time)), l0 + 1), k0)
        interior = np.quantile(temp2, np.arange(1, lq) / lq)
        knots_q = augknt(
            np.concatenate([[0.0], interior, [float(np.max(temp2))]]), kq,
        )
    elif knots_setting == 'K3':
        interior = np.quantile(temp1, np.arange(1, l0) / l0)
        knots_0 = augknt(
            np.concatenate([[0.0], interior, [float(np.max(time))]]), k0,
        )
        knots_q = augknt(np.linspace(0, 2 * float(np.max(temp2)), lq + 1), kq)
    elif knots_setting == 'K4':
        interior0 = np.quantile(temp1, np.arange(1, l0) / l0)
        knots_0 = augknt(
            np.concatenate([[0.0], interior0, [float(np.max(time))]]), k0,
        )
        interiorq = np.quantile(temp2, np.arange(1, lq) / lq)
        knots_q = augknt(
            np.concatenate([[0.0], interiorq, [float(np.max(temp2))]]), kq,
        )
    else:
        raise ValueError(f'unknown knots_setting={knots_setting}')

    q_0 = len(knots_0) - k0
    q_q = len(knots_q) - kq

    r0 = np.zeros(p + q_q + q_0)
    r0[:p] = beta_init / beta_init[0]

    beta = r0[1:p].copy()
    theta = r0[p:p + q_q].copy()
    alpha = r0[p + q_q:].copy()
    print(beta)

    # Equality constraint row (on the sieve concatenation [theta; alpha]):
    # spcol(knots_0, k0, 2) @ alpha == 0  (i.e., alpha(2) = 1 in log-scale).
    Aeq_q = np.zeros(q_q + q_0)
    Aeq_q[q_q:] = spcol(knots_0, k0, np.array([2.0]))[0]
    beq_q = 0.0

    succ_ind = 1
    try:
        for i in range(100):
            sieve = np.concatenate([theta, alpha])

            def fun_theta(r):
                return objective_func_sieve(
                    r, x, time, delta, id_vec, beta,
                    knots_0, knots_q, k0, kq, ci=False,
                )

            constraint = LinearConstraint(Aeq_q.reshape(1, -1), beq_q, beq_q)
            res = minimize(
                fun_theta, sieve, jac=True, method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 100, 'ftol': 5e-4},
            )
            est_sieve = res.x
            step_sieve = float(np.max(np.abs(est_sieve - sieve)))
            theta = est_sieve[:q_q]
            alpha = est_sieve[q_q:]

            def fun_beta(rr):
                return objective_func_beta(
                    rr, x, time, delta, id_vec, theta, alpha,
                    knots_0, knots_q, k0, kq, ci=False,
                )

            res_b = minimize(
                fun_beta, beta, jac=True, method='BFGS',
                options={'maxiter': 500},
            )
            est_beta = res_b.x
            step_beta = float(np.max(np.abs(est_beta - beta)))
            beta = est_beta

            print(beta)
            if max(step_beta, step_sieve) < 5e-4:
                break
    except Exception:
        beta = r0[1:p].copy()
        theta = r0[p:p + q_q].copy()
        alpha = r0[p + q_q:].copy()
        succ_ind = 0

    est = np.concatenate([[1.0], beta, theta, alpha, [succ_ind]])
    return est, knots_0, knots_q, k0, kq, q_0, q_q
