"""Port of random effect/aft_rec/inference.m.

Resampling-based sandwich-variance estimator over the non-zero columns of
the empirical score.  ``B = 150`` perturbations by default.
"""
from __future__ import annotations

import os
import numpy as np

from .generator_rec import generator_rec
from .objective_func_grad import objective_func_grad


def inference(N, seed, data_setting, knots_setting, root=None):
    if root is None:
        root = os.path.dirname(__file__)

    res_file = os.path.join(
        root, 'res',
        f'res_aft_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}.npz',
    )
    res = np.load(res_file)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    p = int(res['p'].ravel()[0])
    q = int(res['q'].ravel()[0])
    knots = res['knots'].ravel()

    data_file = os.path.join(
        root, 'data',
        f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    if not os.path.isfile(data_file):
        generator_rec(N, seed, data_setting, data_dir=os.path.dirname(data_file))
    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel()

    beta = est_r[:p]
    theta = est_r[p:p + q]
    grad = objective_func_grad(
        theta, x, time, delta, id_vec, beta, knots, k, ci=True,
    )
    is_nonzero_col = (np.sum(np.abs(grad), axis=0) > 5).ravel()
    grad_reduced = grad[:, is_nonzero_col]

    Bsamp = 150
    d = est_r.size
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((Bsamp, d))
    Y = np.zeros((Bsamp, d))
    print(f'Inference for seed {seed} with {Bsamp} resamplings')
    for i in range(Bsamp):
        esti = est_r + Z[i] / np.sqrt(N)
        beta_i = esti[:p]
        theta_i = esti[p:p + q]
        gradi = objective_func_grad(
            theta_i, x, time, delta, id_vec, beta_i, knots, k, ci=False,
        )
        Y[i] = gradi / np.sqrt(N)

    Z_red = Z[:, is_nonzero_col]
    Y_red = Y[:, is_nonzero_col]
    V_red = grad_reduced.T @ grad_reduced / N
    A_red = np.linalg.solve(Z_red.T @ Z_red, Z_red.T @ Y_red)
    X_red = np.linalg.solve(A_red.T, V_red) @ np.linalg.inv(A_red)
    se_reduced = np.sqrt(np.abs(np.diag(X_red / N)))

    num_params = grad.shape[1]
    se_all = np.zeros(num_params)
    se_all[is_nonzero_col] = se_reduced
    print(se_all[:3])
    return se_all
