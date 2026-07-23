"""Port of ltm/forward_odesystem_func.m."""
from __future__ import annotations

import numpy as np

from ..common import spcol_deriv


def forward_odesystem_func(y, theta, knots, k):
    y = np.asarray(y, dtype=float).ravel()
    cumhaz = y[0]
    grad_vec = y[1:]
    B, dB = spcol_deriv(knots, k, np.array([cumhaz]))
    B = B[0]
    dB = dB[0]
    d_cumhaz = np.exp(B @ theta)
    d_cumhaz_theta = d_cumhaz * (B + (dB @ theta) * grad_vec)
    return np.concatenate([[d_cumhaz], d_cumhaz_theta])
