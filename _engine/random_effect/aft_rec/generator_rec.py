"""Port of random effect/aft_rec/generator_rec.m.

Identical to the RE-cox generator: Gamma(2, 0.5) frailty applied as a
multiplicative scalar on the intensity.
"""
from __future__ import annotations

from ..cox.generator_rec import generator_rec as _generator_rec
import os as _os


def generator_rec(N, seed, data_setting, rho1=0.5, r1=1.0, data_dir=None):
    if data_dir is None:
        data_dir = _os.path.join(_os.path.dirname(__file__), 'data')
    return _generator_rec(N, seed, data_setting, rho1, r1, data_dir=data_dir)
