"""Port of npmle/baseline_hazard_grad_func.m."""
from __future__ import annotations

import numpy as np

from ..common import spcol


def baseline_hazard_grad_func(time, theta, knots, k):
    """Gradient of lambda_0(t) w.r.t. theta: shape (len(time), q)."""
    time = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    B = spcol(knots, k, time)
    return np.exp(B @ theta)[:, None] * B
