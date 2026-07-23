"""Port of aft/hazard_ode_func.m.

ODE right-hand side for the cumulative hazard in AFT scale: y is the
cumulative hazard and the derivative is ``exp(B(y) @ theta)``.
"""
from __future__ import annotations

import numpy as np

from ..common import spcol


def hazard_ode_func(y, theta, knots, k):
    y = np.atleast_1d(np.asarray(y, dtype=float))
    B = spcol(knots, k, y)
    return np.exp(B @ theta)
