"""Visualization for ode_unify fits: functional parameters with 95% bands.

Works directly on :class:`ode_unify.estimator.Estimate` objects (no result
files), reproducing the plot semantics of the per-model ``visual.py`` /
``visual.m``:

* :func:`curve` -- reconstruct a fitted functional parameter
  ``exp(B(grid) @ theta)`` on a grid, with the pointwise 95% Wald band
  ``exp(B @ (theta +/- 1.96 se_theta))`` when spline SEs are available
  (run :func:`ode_unify.inference.inference` first; for the frailty cox pass
  ``spline_se=True`` there).
* :func:`plot_fit` -- one fitted replication vs an optional true curve
  (one panel for cox/aft; two panels, alpha(t) and q(u), for ltm).
* :func:`band_plot` -- many replications of the same setting: true curve,
  mean/median estimate, mean 95% band, and pointwise/simultaneous coverage
  (the ``test/plot_bands.py`` style used for the paper's simulation figures).

Which functional a fit exposes:

=========  ==========================================
estimator  functional(s)
=========  ==========================================
cox        ``alpha(t)`` (baseline rate)
aft        ``q(u)`` (transformation)
ltm        both ``alpha(t)`` and ``q(u)``
=========  ==========================================
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from ._engine.common import spcol
from .estimator import Estimate

_BAND_COLOR = (0.929, 0.694, 0.125)   # dashed-band colour of the .m figures


# --------------------------------------------------------------------------- #
# curve reconstruction from an Estimate
# --------------------------------------------------------------------------- #

def _se_parts(est):
    """Split est.se_all into per-functional spline SEs (or Nones)."""
    if est.se_all is None:
        return None, None
    se = np.asarray(est.se_all).ravel()
    p = est.beta.size
    if est.estimator == 'ltm':
        q_q = est.spline['q_q']
        q_0 = est.spline['q_0']
        se_theta = se[p - 1:p - 1 + q_q]          # no entry for fixed beta_1
        se_alpha = se[p - 1 + q_q:p - 1 + q_q + q_0]
        return se_theta, se_alpha
    return se[p:], None


def curve(est: Estimate, grid: Sequence[float], which: str = 'auto',
          scale: float = 1.0):
    """Reconstruct a fitted functional parameter on ``grid``.

    Parameters
    ----------
    est : Estimate
    grid : array-like
    which : {'auto', 'alpha', 'q'}
        For cox/aft ``'auto'`` picks their single functional; for ltm you must
        choose ``'alpha'`` or ``'q'``.
    scale : float
        Identifiability rescaling applied to curve and band.

    Returns ``(y, lower, upper)``; the bands are ``None`` when the fit carries
    no spline SEs (run :func:`ode_unify.inference` first).
    """
    grid = np.asarray(grid, dtype=float).ravel()
    p = est.beta.size

    if est.estimator == 'ltm':
        if which not in ('alpha', 'q'):
            raise ValueError("ltm has two functionals: pass which='alpha' "
                             "or which='q'")
        se_theta, se_alpha = _se_parts(est)
        if which == 'q':
            coefs = est.spline['coefs_q']
            B = spcol(np.asarray(est.spline['knots_q']).ravel(),
                      est.spline['kq'], grid)
            se = se_theta
        else:
            coefs = est.spline['coefs_alpha']
            B = spcol(np.asarray(est.spline['knots_0']).ravel(),
                      est.spline['k0'], grid)
            se = se_alpha
    else:
        coefs = est.spline['coefs']
        B = spcol(np.asarray(est.spline['knots']).ravel(),
                  est.spline['k'], grid)
        se, _ = _se_parts(est)

    y = scale * np.exp(B @ coefs)
    if se is None:
        return y, None, None
    upper = scale * np.exp(B @ (coefs + 1.96 * se))
    lower = scale * np.exp(B @ (coefs - 1.96 * se))
    return y, lower, upper


def _default_label(est, which):
    if est.estimator == 'cox':
        return 't', r'$\alpha(t)$'
    if est.estimator == 'aft':
        return 'u', 'q(u)'
    return ('t', r'$\alpha(t)$') if which == 'alpha' else ('u', 'q(u)')


# --------------------------------------------------------------------------- #
# single-fit plot (port of the per-model visual.py `show=True` figures)
# --------------------------------------------------------------------------- #

def plot_fit(est: Estimate, out: str, *, grid=None, grid_u=None,
             truth=None, truth_q=None, scale=1.0, scale_q=1.0,
             title: Optional[str] = None) -> str:
    """Plot one fitted replication (estimate + 95% band, optional truth).

    For cox/aft pass ``grid`` (and optionally ``truth``, a callable). For ltm
    pass ``grid`` (alpha) and ``grid_u`` (q), with optional ``truth`` /
    ``truth_q`` callables and ``scale`` / ``scale_q`` rescalings. Writes the
    figure to ``out`` and returns the path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if grid is None:
        grid = np.linspace(0.05, 2.5, 100)
    grid = np.asarray(grid, dtype=float).ravel()
    two = est.estimator == 'ltm'

    def _panel(ax, which, g, tr, sc):
        y, lo, up = curve(est, g, which=which, scale=sc)
        xlabel, ylabel = _default_label(est, which)
        if tr is not None:
            ax.plot(g, np.asarray(tr(g), dtype=float), 'b', lw=2, label='True')
        ax.plot(g, y, 'r', lw=2, label='Estimate')
        if lo is not None:
            ax.plot(g, up, '--', color=_BAND_COLOR, lw=1)
            ax.plot(g, lo, '--', color=_BAND_COLOR, lw=1, label='95% CI')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    if two:
        gu = (np.asarray(grid_u, dtype=float).ravel()
              if grid_u is not None else grid)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        _panel(axes[0], 'alpha', grid, truth, scale)
        _panel(axes[1], 'q', gu, truth_q, scale_q)
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _panel(ax, 'auto', grid, truth, scale)

    re_tag = ' + frailty' if est.random_effect else ''
    fig.suptitle(title or f'{est.estimator}{re_tag} fit')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# replication band plot (port of test/plot_bands.py, on Estimate lists)
# --------------------------------------------------------------------------- #

def _coverage(true_curve, up, lo):
    inside = (lo <= true_curve[None, :]) & (true_curve[None, :] <= up)
    pw = inside.mean(axis=0)
    return pw.mean(), pw.min(), float(inside.all(axis=1).mean())


def band_plot(estimates: Sequence[Estimate], out: str, *, truth, grid,
              which: str = 'auto', scale: float = 1.0,
              title: Optional[str] = None, use_median: bool = False,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None) -> str:
    """Aggregate many replications: truth, mean estimate, mean 95% band.

    ``estimates`` is a list of inference-completed fits of the *same* model on
    independent replications; ``truth`` is a callable evaluated on ``grid``.
    The subtitle reports pointwise and simultaneous coverage of the truth by
    the per-replication Wald bands. Returns the written path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = np.asarray(grid, dtype=float).ravel()
    est_all, up_all, lo_all = [], [], []
    for e in estimates:
        if not e.success:
            continue
        y, lo, up = curve(e, grid, which=which, scale=scale)
        if lo is None:
            raise ValueError('band_plot needs spline SEs: run inference() '
                             '(spline_se=True for the frailty cox) first')
        est_all.append(y)
        up_all.append(up)
        lo_all.append(lo)
    est_arr = np.asarray(est_all)
    up_arr = np.asarray(up_all)
    lo_arr = np.asarray(lo_all)
    tc = np.asarray(truth(grid), dtype=float)

    m, mn, sim = _coverage(tc, up_arr, lo_arr)
    sub = (f'pointwise cov={m:.3f} (min {mn:.2f})  sim={sim:.2f}  '
           f'n={len(est_arr)}')

    agg = np.median if use_median else np.mean
    tag = 'Median' if use_median else 'Mean'
    e0 = estimates[0]
    xl, yl = _default_label(e0, which)
    xlabel = xlabel or xl
    ylabel = ylabel or yl

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(grid, tc, 'b-', lw=2, label='True curve')
    ax.plot(grid, agg(est_arr, axis=0), 'r-', lw=1.5, label=f'{tag} estimate')
    ax.plot(grid, agg(up_arr, axis=0), '--', color=_BAND_COLOR, lw=1.2,
            label=f'{tag} 95% upper')
    ax.plot(grid, agg(lo_arr, axis=0), '--', color=_BAND_COLOR, lw=1.2,
            label=f'{tag} 95% lower')
    re_tag = ' + frailty' if e0.random_effect else ''
    ax.set_title((title or f'{e0.estimator}{re_tag}') + f'\n{sub}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def ltm_band_plot(estimates: Sequence[Estimate], out_alpha: str, out_q: str,
                  *, truth_alpha, truth_q, grid_t, grid_u,
                  scale_a: float = 1.0, scale_q: float = 1.0,
                  title_alpha: Optional[str] = None,
                  title_q: Optional[str] = None,
                  use_median: bool = False) -> tuple[str, str]:
    """Two aggregate band plots (``alpha(t)`` and ``q(u)``) for LTM fits.

    ``scale_a`` / ``scale_q`` are the identifiability rescalings that put the
    estimates on the truth's scale (see ``ltm/visual.py``).
    """
    pa = band_plot(estimates, out_alpha, truth=truth_alpha, grid=grid_t,
                   which='alpha', scale=scale_a, title=title_alpha,
                   use_median=use_median)
    pq = band_plot(estimates, out_q, truth=truth_q, grid=grid_u,
                   which='q', scale=scale_q, title=title_q,
                   use_median=use_median)
    return pa, pq
