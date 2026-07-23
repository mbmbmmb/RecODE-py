"""Port of ltm/hazard_ode_func.m."""
from __future__ import annotations

import numpy as np

from ..common import spcol


def hazard_ode_func(y, theta, knots_q, kq):
    y = np.atleast_1d(np.asarray(y, dtype=float))
    return np.exp(spcol(knots_q, kq, y) @ theta)
