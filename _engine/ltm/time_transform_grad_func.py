"""Port of ltm/time_transform_grad_func.m."""
from __future__ import annotations

import numpy as np

from ..common import spcol


def time_transform_grad_func(t, alpha, knots_0, k0):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    B0 = spcol(knots_0, k0, t)
    return np.exp(B0 @ alpha)[:, None] * B0
