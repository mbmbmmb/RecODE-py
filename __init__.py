"""Unified recurrent-event ODE toolkit: one generator, one estimator.

* :func:`simulate` -- single data-generating function; the true rate is chosen
  by ``setting`` (or custom ``alpha``/``q``/``rate``) and the frailty by
  ``random_effect`` with a configurable distribution.
* :func:`estimate` -- fast point estimation for ``estimator`` in
  {'cox', 'aft', 'ltm'}, with or without a gamma frailty.
* :func:`inference` -- standard errors, kept separate because it can be much
  slower: closed form (no frailty), closed-form beta adjustment (frailty cox),
  resampling (frailty aft / ltm, automatically).
* :func:`fit` -- convenience: estimate + inference in one call.
* :mod:`ode_unify.visual` -- functional-parameter curves and 95%-band plots
  (:func:`curve`, :func:`plot_fit`, :func:`band_plot`, :func:`ltm_band_plot`).
"""
from .dgp import simulate, true_rate, frailty
from .estimator import Estimate, estimate
from .inference import fit, inference
from .visual import band_plot, curve, ltm_band_plot, plot_fit

__all__ = ['simulate', 'true_rate', 'frailty',
           'Estimate', 'estimate', 'inference', 'fit',
           'curve', 'plot_fit', 'band_plot', 'ltm_band_plot']
