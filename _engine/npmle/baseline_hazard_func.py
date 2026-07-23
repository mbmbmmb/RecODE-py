"""Port of npmle/baseline_hazard_func.m."""
from __future__ import annotations

import numpy as np

from ..common import spcol


def baseline_hazard_func(time, theta, knots, k):
    """Evaluate lambda_0(t) = exp(B(t) @ theta)."""
    time = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    B = spcol(knots, k, time)
    return np.exp(B @ theta)
