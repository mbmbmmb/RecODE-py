"""Port of npmle/Gtransform.m."""
from __future__ import annotations

import numpy as np


def Gtransform(temp, rho, r):
    """G-transform with parameters ``(rho, r)`` and its derivative.

    For rho > 0::
        G(t)  = ((1+t)^rho - 1) / rho
        G'(t) = (1+t)^(rho-1)

    For rho == 0, r == 0 (Cox as special case)::
        G(t) = t, G'(t) = 1

    For rho == 0, r != 0 (logarithmic transform)::
        newr  = 1/r
        G(t)  = newr * log(1 + t/newr)
        G'(t) = 1 / (1 + t/newr)
    """
    temp = np.asarray(temp, dtype=float)
    if rho > 0:
        G = (np.power(1.0 + temp, rho) - 1.0) / rho
        dG = np.power(1.0 + temp, rho - 1.0)
        return G, dG
    if rho == 0:
        if r == 0:
            return temp.copy(), np.ones_like(temp)
        newr = 1.0 / r
        G = newr * np.log1p(temp / newr)
        dG = 1.0 / (1.0 + temp / newr)
        return G, dG
    raise ValueError(f'unsupported rho={rho}')
