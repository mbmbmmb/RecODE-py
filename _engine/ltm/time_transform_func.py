"""Port of ltm/time_transform_func.m."""
from __future__ import annotations

import numpy as np

from ..common import spcol


def time_transform_func(t, alpha, knots_0, k0):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.exp(spcol(knots_0, k0, t) @ alpha)
