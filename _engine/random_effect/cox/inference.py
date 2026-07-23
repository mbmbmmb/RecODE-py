"""Port of random effect/cox/inference.m.

Resampling-based sandwich estimator for the full parameter vector
``(beta, theta)``.  Uses ``B=50`` perturbations and regresses perturbed
gradients against Gaussian design to estimate the Hessian.
"""
from __future__ import annotations

import os
import numpy as np

from .generator_rec import generator_rec
from .objective_func_inf import objective_func_inf


def inference(N, seed, data_setting, root=None):
    if root is None:
        root = os.path.dirname(__file__)
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

    res_file = os.path.join(
        root, 'res',
        f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    res = np.load(res_file)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    knots = res['knots'].ravel()

    _, grad = objective_func_inf(
        est_r, x, time, delta, id_vec, knots, k, ci=True,
    )
    V = grad.T @ grad / N

    Bsamp = 50
    d = est_r.size
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((Bsamp, d))
    Y = np.zeros((Bsamp, d))
    print('Resampling for inference')
    for i in range(Bsamp):
        _, gradi = objective_func_inf(
            est_r + Z[i] / np.sqrt(N),
            x, time, delta, id_vec, knots, k, ci=False,
        )
        Y[i] = gradi * np.sqrt(N)
    A = np.linalg.solve(Z.T @ Z, Z.T @ Y)

    X = np.linalg.solve(A.T, V) @ np.linalg.inv(A)
    fish = X / N

    out_path = os.path.join(
        root, 'res',
        f'res_cox_N{N}_seed{seed}_setting{data_setting}_inference.npz',
    )
    se_all = np.sqrt(np.abs(np.diag(fish)))
    np.savez_compressed(out_path, se_all=se_all.reshape(-1, 1))
    return fish
