"""
Shared utilities used across the Python ports of the RecurrentODE MATLAB code.

This module provides Python equivalents for the MATLAB Curve-Fitting-Toolbox
functions that the original code relies on (``augknt``, ``spcol``), plus a few
convenience helpers for sort/unsort indexing and .mat I/O.

Notation convention
-------------------
MATLAB's B-spline order ``k`` equals ``degree + 1``. ``scipy`` parametrises a
B-spline by its *degree*. Throughout this port we keep MATLAB's ``k`` (order)
as the user-facing argument and convert to degree internally.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.interpolate import BSpline


def augknt(breaks, k):
    """MATLAB ``augknt(breaks, k)``: augmented knot sequence for order-``k``
    splines. The first and last break are repeated ``k`` times."""
    breaks = np.asarray(breaks, dtype=float).ravel()
    return np.concatenate([
        np.full(k, breaks[0]),
        breaks[1:-1],
        np.full(k, breaks[-1]),
    ])


def spcol(knots, k, tau):
    """MATLAB ``spcol(knots, k, tau)``: collocation matrix whose (i, j) entry
    is the j-th order-``k`` B-spline evaluated at ``tau[i]``.

    ``tau`` must be non-decreasing, matching MATLAB's requirement. The returned
    array is dense (``len(tau), n_basis``) where ``n_basis = len(knots) - k``.
    """
    knots = np.asarray(knots, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    degree = k - 1
    n_basis = len(knots) - k
    # design_matrix requires tau inside the support and sorted ascending.
    # Clip tiny floating-point excursions past the right endpoint.
    eps = 1e-12
    tau_clipped = np.clip(tau, knots[0], knots[-1] - eps)
    B = BSpline.design_matrix(tau_clipped, knots, degree, extrapolate=False).toarray()
    # Right endpoint: scipy's half-open convention returns 0 at the last knot.
    # MATLAB's spcol uses the limit from the left, so fix those rows.
    right_mask = tau >= knots[-1] - eps
    if np.any(right_mask):
        B[right_mask, :] = 0.0
        B[right_mask, n_basis - 1] = 1.0
    return B


def spcol_deriv(knots, k, tau):
    """Return ``(B, dB)`` at ``tau`` where ``B[i, j]`` is the j-th order-``k``
    B-spline evaluated at ``tau[i]`` and ``dB[i, j]`` is its derivative.

    Matches the MATLAB pattern ``spcol(knots, k, brk2knt(u, 2))`` where rows
    ``1:2:end`` give the values and ``2:2:end`` give the first derivatives.
    """
    knots = np.asarray(knots, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    n_basis = len(knots) - k
    B = spcol(knots, k, tau)
    if k <= 1:
        return B, np.zeros_like(B)

    lower_degree = k - 2
    n_lower = len(knots) - (lower_degree + 1)  # == n_basis + 1
    eps = 1e-12
    tau_clipped = np.clip(tau, knots[0], knots[-1] - eps)
    B_lower = BSpline.design_matrix(
        tau_clipped, knots, lower_degree, extrapolate=False
    ).toarray()
    right_mask = tau >= knots[-1] - eps
    if np.any(right_mask):
        B_lower[right_mask, :] = 0.0
        B_lower[right_mask, n_lower - 1] = 1.0

    denom1 = knots[k - 1:k - 1 + n_basis] - knots[:n_basis]
    denom2 = knots[k:k + n_basis] - knots[1:1 + n_basis]
    coef1 = np.divide(k - 1, denom1, out=np.zeros_like(denom1), where=denom1 > 0)
    coef2 = np.divide(k - 1, denom2, out=np.zeros_like(denom2), where=denom2 > 0)

    dB = coef1 * B_lower[:, :n_basis] - coef2 * B_lower[:, 1:n_basis + 1]
    return B, dB


def brk2knt(x, mult):
    """MATLAB ``brk2knt(x, mult)`` with scalar ``mult``. For ``mult=1`` this is
    just ``x``; larger values repeat each entry."""
    x = np.asarray(x, dtype=float).ravel()
    if np.isscalar(mult) or np.size(mult) == 1:
        m = int(mult)
        if m == 1:
            return x
        return np.repeat(x, m)
    return np.repeat(x, np.asarray(mult, dtype=int).ravel())


def unique_sort_index(v):
    """Return (u, bin) where ``u = np.unique(v)`` is sorted ascending and
    ``bin`` contains, for each entry of ``v``, the 0-based index into ``u``.

    This mirrors MATLAB's ``[~, bin] = histc(v, u)`` idiom used throughout
    the original code (for 1-based indices add 1)."""
    v = np.asarray(v, dtype=float).ravel()
    u = np.unique(v)
    bin_idx = np.searchsorted(u, v)
    return u, bin_idx


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------- #
# ODE integration with failure detection
# --------------------------------------------------------------------------- #

class ODEIntegrationError(RuntimeError):
    """``solve_ivp`` did not integrate over the whole requested span.

    Raised instead of letting a truncated solution propagate. When RK45 cannot
    take a step -- typically because a trial parameter has pushed the
    transformed time scale into a stiff or degenerate region -- it returns
    *fewer* columns than ``t_eval`` has entries. Downstream code indexes the
    result as though every requested point were present, which surfaced as a
    confusing ``IndexError: index N is out of bounds for axis 0 with size N``
    far from the real cause.
    """


def solve_ode(rhs, t_span, y0, t_eval, **kw):
    """``solve_ivp`` that fails loudly instead of returning a partial solution.

    Verifies both that the solver reports success and that it produced one
    column per requested evaluation point, so callers may index the result by
    position without checking.
    """
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, **kw)
    n_want = len(np.asarray(t_eval).ravel())
    if (not sol.success) or sol.y.shape[1] != n_want:
        raise ODEIntegrationError(
            f'integration failed over [{t_span[0]:.6g}, {t_span[-1]:.6g}]: '
            f'{sol.message} (returned {sol.y.shape[1]} of {n_want} points)')
    return sol


# large finite objective returned when the ODE solve fails, so that a line
# search treats the point as very bad and steps back rather than aborting
_ODE_FAIL_OBJECTIVE = 1e10


def ode_guard(fn):
    """Make a (loss, gradient) objective robust to a failed ODE solve.

    An optimiser exploring a trial parameter can push the transformed time
    scale into a region where ``solve_ivp`` cannot integrate. Previously that
    raised out of the objective and aborted the whole fit. Returning a large
    finite value with a zero gradient instead lets the line search reject the
    point and continue, which is the conventional handling for an objective
    that is undefined on part of its domain.

    Only the optimiser's ``(loss, gradient)`` path is guarded. Calls made with
    ``ci=True`` return a per-subject score matrix used for variance estimation,
    where a silent sentinel would be wrong, so those re-raise.
    """
    import functools

    import os as _os

    @functools.wraps(fn)
    def wrapper(r, *args, **kw):
        try:
            return fn(r, *args, **kw)
        except ODEIntegrationError:
            if _os.environ.get('ODE_GUARD_DISABLE'):
                raise
            ci = kw.get('ci', args[9] if len(args) > 9 else False)
            if ci:
                raise
            # Soft barrier rather than a flat penalty. A zero gradient tells the
            # optimiser the surface is flat here, which breaks BFGS's curvature
            # update (the secant denominator collapses) and can produce NaN
            # steps -- that made previously-converging fits fail. Returning a
            # quadratic bowl centred at the origin instead gives a well-defined
            # descent direction back toward the starting point (theta = alpha =
            # 0), where the ODE is well behaved.
            r = np.asarray(r, dtype=float)
            return (_ODE_FAIL_OBJECTIVE * (1.0 + float(r @ r)),
                    _ODE_FAIL_OBJECTIVE * 2.0 * r)
    return wrapper
