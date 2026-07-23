"""Port of aft/augode_func.m. Per-subject augmented ODE for adjoint gradients."""
from __future__ import annotations

import numpy as np

from ..common import spcol_deriv


def augode_func(y, theta, x_coef, knots, k):
    y = np.asarray(y, dtype=float).ravel()
    cumhaz = y[0]
    a = y[1]
    B, dB = spcol_deriv(knots, k, np.array([cumhaz]))
    B = B[0]
    dB = dB[0]
    res_1 = np.exp(B @ theta) * x_coef
    res_2 = -a * res_1 * (dB @ theta)
    res_4 = -a * res_1 * B
    return np.concatenate([[res_1, res_2], res_4])
