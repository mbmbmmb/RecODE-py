"""Port of random effect/aft_rec/forward_odesystem_func.m (identical to aft/).

Augmented forward ODE for the AFT model: the state is
``y = [cumhaz; d cumhaz / d theta]``.  The spline is evaluated at the
integration state ``y[0]`` (not at time ``t``), matching the MATLAB code.
"""
from __future__ import annotations

import numpy as np

from ...common import spcol_deriv


def forward_odesystem_func(y, theta, knots, k):
    y = np.asarray(y, dtype=float).ravel()
    cumhaz = y[0]
    grad_vec = y[1:]
    B, dB = spcol_deriv(knots, k, np.array([cumhaz]))
    B = B[0]; dB = dB[0]
    d_cumhaz = np.exp(B @ theta)
    d_cumhaz_theta = d_cumhaz * (B + (dB @ theta) * grad_vec)
    return np.concatenate([[d_cumhaz], d_cumhaz_theta])
