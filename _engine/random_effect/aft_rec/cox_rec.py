"""Port of random effect/aft_rec/cox_rec.m (identical to aft/)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _cox_negll(beta, x, time, delta):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    lin = x @ beta
    ex = np.exp(lin)
    order = np.argsort(-time)
    ex_s = ex[order]
    lin_s = lin[order]
    delta_s = delta[order]
    x_s = x[order]
    cum_ex = np.cumsum(ex_s)
    cum_x_ex = np.cumsum(x_s * ex_s[:, None], axis=0)
    obs = delta_s > 0
    logsum = np.log(cum_ex[obs])
    ll = (lin_s[obs] - logsum).sum()
    xbar = cum_x_ex[obs] / cum_ex[obs][:, None]
    grad = (x_s[obs] - xbar).sum(axis=0)
    return -ll, -grad


def cox_rec(x, time, delta):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    p = x.shape[1]
    beta0 = np.zeros(p)
    res = minimize(
        _cox_negll, beta0, jac=True, method='BFGS',
        args=(x, time, delta), options={'maxiter': 500},
    )
    beta = res.x
    temp = time * np.exp(x @ beta)
    return temp, beta
