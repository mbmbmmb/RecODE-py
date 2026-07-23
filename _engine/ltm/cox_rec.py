"""Port of ltm/cox_rec.m."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ..common import augknt
from .baseline_hazard_func import baseline_hazard_func
from .objective_func import objective_func


def cox_rec(x, time, delta):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()

    order = np.argsort(time, kind='stable')
    time = time[order]; x = x[order]; delta = delta[order]

    censored_idx = np.where(delta == 0)[0]
    m = len(censored_idx)
    tau = time[censored_idx]
    x_tau = x[censored_idx]

    p = x.shape[1]
    k = 3
    l = int(np.ceil(x.shape[0] ** (1.0 / 7.0)))
    knots = augknt(np.linspace(0.0, float(np.max(time)), l + 1), k)
    q = len(knots) - k

    r0 = np.zeros(p + q)

    def fun(r):
        return objective_func(r, x, time, delta, p, q, knots, k)

    res = minimize(fun, r0, jac=True, method='BFGS',
                   options={'maxiter': 500})
    est_r = res.x

    beta = est_r[:p]
    theta = est_r[p:p + q]

    tspan = np.concatenate([[1e-12], tau])

    def rhs(t, y):
        return baseline_hazard_func(np.array([t]), theta, knots, k)

    sol = solve_ivp(rhs, (tspan[0], tspan[-1]), [0.0],
                    t_eval=tspan, method='RK45', rtol=1e-3, atol=1e-6)
    cum_baseline = sol.y[0][1:]
    x_coef = np.exp(x_tau @ beta)
    temp = x_coef * cum_baseline
    return temp, beta
