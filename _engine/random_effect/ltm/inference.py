"""Port of random effect/ltm/inference.m.

Resampling-based sandwich variance: draw ``B`` zero-mean Gaussians,
perturb the estimate by ``Z/sqrt(N)``, evaluate the per-sample gradient
under each perturbation, and recover the information matrix by OLS on
``Z`` against the scaled gradients.  ``B`` is ``800`` (setting 1) or
``1000`` (setting 2) — matching MATLAB.
"""
from __future__ import annotations

import os
import numpy as np

from .inference_objective_func_sieve import inference_objective_func_sieve


def _paths(root, data_setting):
    if data_setting == 1:
        data_sub, res_sub, res_prefix = 'cox_rec_rd', 'cox_rec_rd', 'res_cox_N'
    elif data_setting == 2:
        data_sub, res_sub, res_prefix = 'aft_rd', 'aft_rd', 'res_aft_N'
    else:
        raise ValueError(f'unsupported data_setting={data_setting}')
    data_dir = os.path.join(root, 'data', data_sub)
    res_dir = os.path.join(root, 'res', res_sub)
    return data_dir, res_dir, res_prefix


def inference(N, seed, data_setting, knots_setting, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_dir, res_dir, res_prefix = _paths(root, data_setting)
    data_file = os.path.join(
        data_dir, f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    result_file = os.path.join(
        res_dir,
        f'{res_prefix}{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}.npz',
    )

    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel()

    res = np.load(result_file)
    est_r = res['est_r'].ravel()
    p = int(res['p'].ravel()[0])
    q_q = int(res['q_q'].ravel()[0])
    q_0 = int(res['q_0'].ravel()[0])
    knots_0 = res['knots_0'].ravel()
    knots_q = res['knots_q'].ravel()
    k0 = int(res['k0'].ravel()[0])
    kq = int(res['kq'].ravel()[0])

    beta = est_r[1:p]
    theta = est_r[p:p + q_q]
    alpha = est_r[p + q_q:p + q_q + q_0]

    grad = inference_objective_func_sieve(
        np.concatenate([theta, alpha]), x, time, delta, id_vec, beta,
        knots_0, knots_q, k0, kq, ci=True,
    )
    V = (grad.T @ grad) / N

    B = 800 if data_setting == 1 else 1000
    d = est_r.size
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((d, B))
    Y = np.zeros((B, d - 1))
    print('Resampling!')
    for i in range(B):
        if i + 1 == 300:
            print('300')
        elif i + 1 == 600:
            print('600')
        esti = est_r + Z[:, i] / np.sqrt(N)
        beta_i = esti[1:p]
        theta_i = esti[p:p + q_q]
        alpha_i = esti[p + q_q:p + q_q + q_0]
        gradi = inference_objective_func_sieve(
            np.concatenate([theta_i, alpha_i]), x, time, delta, id_vec,
            beta_i, knots_0, knots_q, k0, kq, ci=False,
        )
        Y[i] = gradi / np.sqrt(N)
    Z = Z[1:, :]          # drop the (fixed) first-beta row
    A = np.linalg.solve(Z @ Z.T, Z @ Y)   # (d-1, d-1)
    # MATLAB:  X = (A')\(V/A) = inv(A.T) * V * inv(A)
    V_invA = np.linalg.solve(A.T, V.T).T        # V @ inv(A)
    X = np.linalg.solve(A.T, V_invA)            # inv(A.T) @ (V @ inv(A))
    fish = X / N
    return fish
