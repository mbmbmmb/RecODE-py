"""Port of random effect/aft_rec/hazard_ode_func.m (identical to aft/)."""
from __future__ import annotations

import numpy as np

from ...common import spcol


def hazard_ode_func(y, theta, knots, k):
    y_arr = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    B = spcol(knots, k, y_arr)
    return np.exp(B @ theta)
