"""Paper-faithful data generation for the six simulation settings of §5.

:mod:`ode_unify.dgp` is the *general* generator: its covariate design is fixed
(two N(0,1) clipped at ±1 plus a Bernoulli(0.5)) and its censoring window
defaults to ``U(2,4)`` for every setting. That combination reproduces only the
paper's Setting 3, so Monte-Carlo summaries produced with it are not comparable
to Tables 1-3 of ``latex/main.tex``.

This module pins the covariate distribution, the censoring distribution and the
frailty to what §5 actually specifies, per setting:

===  ==============  ===============  ===========================  =========
 #   ``alpha(t)``    ``q(u)``         covariates                   censoring
===  ==============  ===============  ===========================  =========
 1   ``t^2 + 1``     ``1``            N(0, 0.5) trunc ±4           U(0, 2)
 2   ``1``           ``2/(1+u)``      N(0, 0.5) trunc ±4           U(1, 3)
 3   ``0.2/(1+t)``   ``1/(u/2+1)``    N(0,1) trunc ±1 (x1,x2),     min{U(2,6), 4}
                                      Bernoulli(0.5) (x3)
 4   ``t + 1``       ``2/(1+u)``      N(0, 0.5) trunc ±4           U(1, 3)
 5   ``t^2 + 1``     ``1``            N(0, 0.5) trunc ±4           U(0, 2)
 6   ``1``           ``2/(t+1)``      N(0, 0.5) trunc ±4           U(1, 3)
===  ==============  ===============  ===========================  =========

Settings 5 and 6 are Settings 1 and 2 with a Gamma frailty
``xi ~ Gamma(mean 1, var 0.5)`` multiplying the intensity; the paper states
their covariates are "generated in the same way as in Setting 1". The paper does
not restate their censoring windows, so we inherit them from the matching
non-frailty setting (1 -> 5, 2 -> 6).

Usage::

    from ode_unify.paper_dgp import simulate_paper
    data = simulate_paper(1000, seed=1, setting=1)
"""
from __future__ import annotations

import numpy as np

from .dgp import _intensity_factory, frailty

__all__ = ['true_rate_paper', 'paper_design', 'simulate_paper', 'PAPER_DESIGN',
           'CUSTOM_FUNCS']

# Settings beyond the paper's six. Setting 7 is a stress test for model
# selection: q is NON-MONOTONE and oscillating, so it lies far outside the
# Box-Cox family e^c/(1+rho*u), which is monotone decreasing and hyperbolic.
# No member of that family can mimic it at any rho or scale, so a criterion
# that still prefers the parametric transformation here would be at fault.
# q stays in [0.3, 1.7], safely positive, and over the realised mu range
# (roughly 0 to 6) the argument 1.5*u spans about 1.4 periods.
CUSTOM_FUNCS = {
    7: dict(alpha=lambda t: np.asarray(t, dtype=float) + 1.0,
            q=lambda u: 1.0 + 0.7 * np.sin(1.5 * np.asarray(u, dtype=float)),
            desc='alpha(t)=t+1, q(u)=1+0.7 sin(1.5u)  (non-monotone)'),
}


# --------------------------------------------------------------------------- #
# 1. Closed-form intensities  mu'_x(t)  with  m = exp(x'beta)
# --------------------------------------------------------------------------- #

def true_rate_paper(setting, rho1=0.5):
    """Analytic ``rate(t, m)`` for paper settings 1-6.

    Each is the closed-form solution of ``mu' = m q(mu) alpha(t)``, ``mu(0)=0``:

    * 1, 5  ``alpha = t^2+1``, ``q = 1``            -> ``m (t^2+1)``
    * 2, 6  ``alpha = 1``, ``q = 2/(1+u)``          -> ``2m / sqrt(4mt+1)``
    * 3     Box-Cox, ``rho1``                       -> see below
    * 4     ``alpha = t+1``, ``q = 2/(1+u)``        -> ``2m(t+1)/sqrt(2mt(t+2)+1)``

    Settings 5 and 6 share the intensity of 1 and 2; the frailty enters
    multiplicatively in :func:`simulate_paper`, not here.
    """
    if setting in (1, 5):
        return lambda t, m: m * (t ** 2 + 1.0)
    if setting in (2, 6):
        return lambda t, m: 2.0 * m / np.sqrt(4.0 * m * t + 1.0)
    if setting == 3:
        a0 = 0.2
        return lambda t, m: (m * (a0 / (1.0 + t))
                             * (1.0 + m * a0 * np.log1p(t)) ** (rho1 - 1.0))
    if setting == 4:
        return lambda t, m: (2.0 * m * (t + 1.0)
                             / np.sqrt(2.0 * m * t * (t + 2.0) + 1.0))
    if setting in CUSTOM_FUNCS:
        return None          # no closed form; integrated numerically
    raise ValueError(f'unknown paper setting={setting} (expected 1-7)')


# --------------------------------------------------------------------------- #
# 2. Per-setting covariate / censoring / frailty design
# --------------------------------------------------------------------------- #

# cov: 'normal'  -> all p covariates N(0, sd) truncated at +-trunc
#      'mixed'   -> (p-1) N(0, sd) truncated at +-trunc, last one Bernoulli(0.5)
# censor: ('unif', a, b)          -> C ~ U(a, b)
#         ('unif_cap', a, b, cap) -> C ~ min{U(a, b), cap}
PAPER_DESIGN = {
    1: dict(cov='normal', sd=0.5, trunc=4.0, censor=('unif', 0.0, 2.0),
            random_effect=False, n_paper=1000,
            desc='Cox-type: alpha=t^2+1, q=1'),
    2: dict(cov='normal', sd=0.5, trunc=4.0, censor=('unif', 1.0, 3.0),
            random_effect=False, n_paper=1000,
            desc='AFT-type: alpha=1, q=2/(1+u)'),
    3: dict(cov='mixed', sd=1.0, trunc=1.0, censor=('unif_cap', 2.0, 6.0, 4.0),
            random_effect=False, n_paper=1000,
            desc='Box-Cox LT (rho=0.5): alpha=0.2/(1+t), q=1/(u/2+1)'),
    4: dict(cov='normal', sd=0.5, trunc=4.0, censor=('unif', 1.0, 3.0),
            random_effect=False, n_paper=1000,
            desc='General LT: alpha=t+1, q=2/(1+u)'),
    5: dict(cov='normal', sd=0.5, trunc=4.0, censor=('unif', 0.0, 2.0),
            random_effect=True, n_paper=2000,
            desc='Gamma frailty, Cox-type: alpha=t^2+1, q=1'),
    6: dict(cov='normal', sd=0.5, trunc=4.0, censor=('unif', 1.0, 3.0),
            random_effect=True, n_paper=2000,
            desc='Gamma frailty, AFT-type: alpha=1, q=2/(t+1)'),
    7: dict(cov='normal', sd=0.5, trunc=4.0, censor=('unif', 1.0, 3.0),
            random_effect=False, n_paper=1000,
            desc='non-monotone q: alpha=t+1, q=1+0.7 sin(1.5 u)'),
}


def paper_design(setting):
    """Return a copy of the design dict for ``setting`` (1-6)."""
    if setting not in PAPER_DESIGN:
        raise ValueError(f'unknown paper setting={setting} (expected 1-6)')
    return dict(PAPER_DESIGN[setting])


def _draw_covariates(rng, N, p, design):
    """Covariates exactly as §5 specifies them for this setting."""
    sd, trunc = design['sd'], design['trunc']
    if design['cov'] == 'normal':
        # all p covariates ~ N(0, sd) truncated at +-trunc
        return np.clip(sd * rng.standard_normal((N, p)), -trunc, trunc)
    if design['cov'] == 'mixed':
        # Setting 3: x1..x_{p-1} ~ N(0, sd) truncated, x_p ~ Bernoulli(0.5)
        cols = [np.clip(sd * rng.standard_normal(N), -trunc, trunc)
                for _ in range(p - 1)]
        cols.append((rng.random(N) < 0.5).astype(float))
        return np.column_stack(cols)
    raise ValueError(f'unknown covariate design {design["cov"]!r}')


def _draw_censoring(rng, N, design):
    """Censoring times exactly as §5 specifies them for this setting."""
    spec = design['censor']
    if spec[0] == 'unif':
        _, a, b = spec
        return a + (b - a) * rng.random(N)
    if spec[0] == 'unif_cap':
        _, a, b, cap = spec
        return np.minimum(a + (b - a) * rng.random(N), cap)
    raise ValueError(f'unknown censoring spec {spec!r}')


# --------------------------------------------------------------------------- #
# 3. Generator
# --------------------------------------------------------------------------- #

def simulate_paper(N, seed, setting, *, beta=(1.0, 1.0, 1.0), rho1=0.5, r1=1.0,
                   frailty_params=(2.0, 0.5), censor_coef=None,
                   censor_frailty_coef=0.0, n_grid=200,
                   random_effect=None, censor=None):
    """Simulate one replication under the paper's Setting ``setting`` (1-6).

    Mirrors :func:`ode_unify.dgp.simulate`'s thinning loop and per-subject RNG
    streams exactly; only the covariate design, the censoring distribution and
    the frailty switch come from :data:`PAPER_DESIGN` instead of the defaults.

    Parameters
    ----------
    N, seed, setting : int
    beta : array-like, default ``(1, 1, 1)`` -- the paper's truth.
    frailty_params : Gamma ``(shape, scale)``; ``(2, 0.5)`` gives mean 1,
        variance 0.5 as specified in §5.2.
    censor_coef, censor_frailty_coef :
        Informative-censoring knobs with the same meaning as in
        :func:`ode_unify.dgp.simulate`: ``C_i = C0_i * exp(x_i'gamma_c) *
        xi_i**c_xi``, where ``C0_i`` is the setting's paper censoring draw.
    random_effect, censor :
        Override the setting's design (used by the informative-censoring study);
        ``None`` keeps the paper value.

    Returns the same long-format dict as :func:`ode_unify.dgp.simulate`.
    """
    design = paper_design(setting)
    if random_effect is None:
        random_effect = design['random_effect']
    if censor is not None:
        design['censor'] = ('unif', float(censor[0]), float(censor[1]))

    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    p = beta.size

    x = _draw_covariates(rng, N, p, design)
    xi = frailty(N, rng, random_effect, 'gamma', frailty_params)
    u = _draw_censoring(rng, N, design)

    # informative censoring (no-ops when both knobs are at their defaults)
    if censor_coef is not None:
        gc = np.asarray(censor_coef, dtype=float).reshape(-1)
        if gc.size != p:
            raise ValueError(f'censor_coef must have length p={p}, got {gc.size}')
        u = u * np.exp(x @ gc)
    if censor_frailty_coef:
        u = u * np.power(np.maximum(xi, 1e-12), float(censor_frailty_coef))

    rate = true_rate_paper(setting, rho1)
    if rate is None:                       # custom alpha/q, solved numerically
        cf = CUSTOM_FUNCS[setting]
        make_lam = _intensity_factory(None, None, cf['alpha'], cf['q'],
                                      rho1, None)
    else:
        make_lam = _intensity_factory(None, rate, None, None, rho1, None)

    rows = []
    for i in range(N):
        sub_rng = np.random.default_rng(seed * (i + 1))
        m = float(np.exp(x[i] @ beta))
        xi_i = float(xi[i])
        ui = float(u[i])

        lam = make_lam(m, ui)
        t_grid = np.linspace(0.0, ui, n_grid)
        lambda_max = float(np.max(xi_i * lam(t_grid)))
        if lambda_max < 1e-9:
            lambda_max = 1e-9

        t_events = []
        s = 0.0
        while s < ui:
            s = s - np.log(sub_rng.random()) / lambda_max
            if sub_rng.random() <= xi_i * lam(s) / lambda_max:
                t_events.append(s)

        xrow = x[i]
        if not t_events:
            rows.append(np.concatenate([[i + 1, ui, 0.0], xrow]))
        else:
            n = len(t_events)
            if t_events[-1] < ui:
                for j in range(n):
                    rows.append(np.concatenate([[i + 1, t_events[j], 1.0], xrow]))
                rows.append(np.concatenate([[i + 1, ui, 0.0], xrow]))
            else:
                t_events[-1] = ui
                for j in range(n - 1):
                    rows.append(np.concatenate([[i + 1, t_events[j], 1.0], xrow]))
                rows.append(np.concatenate([[i + 1, t_events[-1], 0.0], xrow]))

    out = np.asarray(rows)
    return {
        'x': np.ascontiguousarray(out[:, 3:]),
        'time': np.ascontiguousarray(out[:, 1].reshape(-1, 1)),
        'delta': np.ascontiguousarray(out[:, 2].reshape(-1, 1)),
        'id': np.ascontiguousarray(out[:, 0].astype(int).reshape(-1, 1)),
        'rho1': float(rho1),
        'r1': float(r1),
    }
