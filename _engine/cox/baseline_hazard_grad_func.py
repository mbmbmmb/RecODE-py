"""Port of cox/baseline_hazard_grad_func.m."""
from __future__ import annotations

import numpy as np

from ..common import spcol


def baseline_hazard_grad_func(time, theta, knots, k):
    """d lambda_0(t) / d theta = exp(B theta) * B, shape (len(time), q)."""
    time = np.atleast_1d(np.asarray(time, dtype=float))
    B = spcol(knots, k, time)
    return np.exp(B @ theta)[:, None] * B
