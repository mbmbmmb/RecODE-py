"""Port of random effect/cox/inference_beta.m.

Closed-form sandwich variance for the regression coefficients under the
Cox-type setting with a frailty.  Only valid for ``data_setting == 1``.

Two implementation notes vs. the literal port:

* ``v(t)`` is computed in O(m + n log m + n p) via a reverse cumulative
  sum over the censoring rows sorted by ``tau``, instead of the
  O(n m p) per-row scan in the MATLAB source.  Mathematically identical
  (same at-risk weighted means).
* The N x n indicator matrix that the MATLAB code uses to aggregate
  score residuals to the subject level is replaced with
  ``np.add.at``-style sparse scatter; identical result, but avoids the
  multi-GB dense allocation that breaks at our cohort size.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.integrate import solve_ivp

from .baseline_hazard_func import baseline_hazard_func
from .baseline_hazard_func_v import baseline_hazard_func_v
from .generator_rec import generator_rec


def _v_at_times(query_times, tau, x_tau, multi_coef):
    """Weighted at-risk means of ``x_tau`` evaluated at each query time.

    For each ``t`` in ``query_times`` returns
    ``sum_{j: tau_j >= t} multi_coef_j x_tau_j / sum_{j: tau_j >= t} multi_coef_j``,
    i.e. the v(t) used by the sandwich variance.  Computed via a reverse
    cumulative sum on tau-sorted (multi_coef, multi_coef * x_tau).
    """
    query_times = np.asarray(query_times, dtype=float).ravel()
    n_q = query_times.size
    p = x_tau.shape[1]
    if tau.size == 0:
        return np.zeros((n_q, p))

    order = np.argsort(tau, kind='mergesort')
    tau_sorted = tau[order]
    w_sorted = multi_coef[order]
    wx_sorted = w_sorted[:, None] * x_tau[order]

    # suffix_w[k] = sum_{j >= k} w_sorted[j];   suffix_wx similarly.
    # Append a zero row so suffix_*[m] = 0 (empty at-risk set).
    suffix_w = np.concatenate([w_sorted[::-1].cumsum()[::-1], [0.0]])
    suffix_wx = np.vstack([
        wx_sorted[::-1].cumsum(axis=0)[::-1],
        np.zeros((1, p)),
    ])

    # tau_j >= t  <=>  j >= searchsorted(tau_sorted, t, side='left').
    idx = np.searchsorted(tau_sorted, query_times, side='left')
    denom = suffix_w[idx]
    num = suffix_wx[idx]
    out = np.zeros((n_q, p))
    nz = denom > 0
    out[nz] = num[nz] / denom[nz, None]
    return out


def inference_beta(N, seed, data_setting, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    setting = 1
    data_file = os.path.join(
        root, 'data', f'simudata_N{N}_seed{seed}_setting{setting}.npz',
    )
    if not os.path.isfile(data_file):
        generator_rec(N, seed, setting, data_dir=os.path.dirname(data_file))
    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel().astype(int)

    n, p = x.shape
    idx = np.where(delta == 0)[0]
    tau = time[idx]
    x_tau = x[idx]

    res_file = os.path.join(
        root, 'res', f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    res = np.load(res_file)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    knots = res['knots'].ravel()

    beta = est_r[:p]
    theta = est_r[p:]
    multi_coef = np.exp(x_tau @ beta)

    v_val = _v_at_times(time, tau, x_tau, multi_coef)

    u, bin_idx = np.unique(tau, return_inverse=True)
    tspan = np.concatenate([[1e-12], u])
    sol = solve_ivp(
        lambda t, y: baseline_hazard_func(t, theta, knots, k),
        (tspan[0], tspan[-1]), [0.0], t_eval=tspan, method='RK45',
        rtol=1e-6, atol=1e-9,
    )
    cum_baseline_hazard = sol.y[0, 1:][bin_idx]

    sol_v = solve_ivp(
        lambda t, y: baseline_hazard_func_v(
            np.atleast_1d(t), tau, x_tau, beta, theta, knots, k,
        ).ravel(),
        (tspan[0], tspan[-1]), np.zeros(p), t_eval=tspan, method='RK45',
        rtol=1e-6, atol=1e-9,
    )
    cum_baseline_hazard_v = sol_v.y[:, 1:].T[bin_idx]

    # Subject-level aggregation of (x - v_val) * delta via sparse scatter.
    # id_vec is assumed to be 1..N (the per-setting mains and the api.fit
    # wrapper both ensure this).
    b_1 = np.zeros((N, p))
    np.add.at(b_1, id_vec - 1, (x - v_val) * delta[:, None])

    b_2 = multi_coef[:, None] * (
        -x_tau * cum_baseline_hazard[:, None] + cum_baseline_hazard_v
    )
    Bmat_b2 = np.zeros((N, p))
    np.add.at(Bmat_b2, id_vec[idx] - 1, b_2)
    Bmat = b_1 + Bmat_b2
    Bmat = Bmat.T @ Bmat / N

    a_beta = (x_tau * (multi_coef * cum_baseline_hazard)[:, None]).T @ x_tau
    a_theta = cum_baseline_hazard_v.T @ (x_tau * multi_coef[:, None])
    A = (-a_beta + a_theta) / N
    X = np.linalg.solve(A, Bmat @ np.linalg.inv(A.T))
    return X / N
