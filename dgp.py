"""One flexible data-generating function for the RecurrentODE family.

The recurrent-event mean function solves the ODE

    mu'_x(t) = xi * q(mu(t)) * exp(x'beta) * alpha(t),      mu(0) = 0,

and events are the points of a non-homogeneous Poisson process with intensity
``mu'_x(t)``. A single :func:`simulate` covers every case:

* **presets** -- ``setting in {1,2,3,4}`` uses the paper's closed-form rate
  (fast, and bit-identical to the standalone generators);
* **custom functionals** -- pass any ``alpha(t)`` and/or ``q(u)`` (defaulting to
  1) and the intensity is obtained by integrating the ODE, or pass a closed-form
  ``rate(t, m)`` directly (``m = exp(x'beta)``);
* **frailty** -- ``random_effect=True`` draws ``xi`` from a configurable
  distribution (gamma / lognormal / inverse-Gaussian / any callable), else
  ``xi = 1``.

``alpha``, ``q`` and ``rate`` are the ``f(setting)`` handles generalised: instead
of only the four presets you may supply arbitrary functions.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Rate / functional-parameter specification:  lambda = f(setting)
# --------------------------------------------------------------------------- #

def true_rate(setting, rho1=0.5):
    """Preset closed-form intensity ``rate(t, m)`` with ``m = exp(x'beta)``.

    These are the analytic solutions ``mu'_x(t)`` of the four canonical models.
    """
    if setting == 1:                       # Cox: alpha(t) = t^2 + 1, q = 1
        return lambda t, m: m * (t ** 2 + 1.0)
    if setting == 2:                       # AFT: alpha = 1
        return lambda t, m: 2.0 * m / np.sqrt(4.0 * m * t + 1.0)
    if setting == 3:                       # Box-Cox linear transformation
        a0 = 0.2
        return lambda t, m: (m * (a0 / (1.0 + t))
                             * (1.0 + m * a0 * np.log1p(t)) ** (rho1 - 1.0))
    if setting == 4:                       # general linear transformation
        return lambda t, m: (2.0 * m * (t + 1.0)
                             / np.sqrt(2.0 * m * t * (t + 2.0) + 1.0))
    raise ValueError(f'unknown setting={setting}')


def _intensity_factory(setting, rate, alpha, q, rho1, ode_kw):
    """Return ``make_lam(m, u)`` giving the base intensity ``lambda0(t)`` (no
    frailty) for a subject with linear predictor ``m = exp(x'beta)`` over
    ``[0, u]``.

    Priority: explicit ``rate`` > preset ``setting`` > custom ``alpha``/``q``
    (integrated via the ODE).
    """
    if rate is not None:                                   # user closed form
        def make_lam(m, u):
            return lambda t: np.asarray(rate(t, m), dtype=float)
        return make_lam

    if setting is not None:                                # preset closed form
        rf = true_rate(setting, rho1)

        def make_lam(m, u):
            return lambda t: np.asarray(rf(t, m), dtype=float)
        return make_lam

    # custom alpha(t) and/or q(u): integrate  mu' = m q(mu) alpha(t)
    a = alpha if alpha is not None else (lambda t: np.ones_like(np.asarray(t, float)))
    qq = q if q is not None else (lambda u: np.ones_like(np.asarray(u, float)))
    from scipy.integrate import solve_ivp
    kw = dict(rtol=1e-9, atol=1e-12, dense_output=True)
    kw.update(ode_kw or {})

    def make_lam(m, u):
        sol = solve_ivp(lambda t, y: m * float(qq(y[0])) * float(a(t)),
                        (0.0, float(u)), [0.0], **kw)

        def lam(t):
            tt = np.atleast_1d(np.asarray(t, dtype=float))
            mu = sol.sol(tt)[0]
            val = m * np.asarray(qq(mu), float) * np.asarray(a(tt), float)
            return val if np.ndim(t) else float(val[0])
        return lam
    return make_lam


# --------------------------------------------------------------------------- #
# 2. Frailty (random-effect) distribution
# --------------------------------------------------------------------------- #

def frailty(N, rng, random_effect, dist='gamma', params=(2.0, 0.5)):
    """Subject multipliers ``xi`` (length ``N``).

    ``random_effect=False`` returns all ones **without drawing** any random
    numbers (so the downstream RNG stream matches the non-frailty generators).
    Otherwise draw from ``dist``:

    * ``'gamma'`` -- ``params=(shape, scale)``; ``(2, 0.5)`` has mean 1.
    * ``'lognormal'`` -- ``params=(sigma,)``; mean pinned to 1 (``mu=-sigma^2/2``).
    * ``'invgauss'`` -- ``params=(scale,)``; mean-1 inverse Gaussian (Wald).
    * a **callable** ``dist(N, rng) -> array`` for anything else.
    """
    if not random_effect:
        return np.ones(N)
    if callable(dist):
        return np.asarray(dist(N, rng), dtype=float).reshape(N)
    if dist == 'gamma':
        shape, scale = params
        return rng.gamma(shape, scale, size=N)
    if dist == 'lognormal':
        sigma = params[0]
        return rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=N)
    if dist == 'invgauss':
        scale = params[0] if params else 1.0
        return rng.wald(mean=1.0, scale=scale, size=N)
    raise ValueError(f'unknown frailty dist={dist!r}')


# --------------------------------------------------------------------------- #
# 3. The single generator
# --------------------------------------------------------------------------- #

def simulate(N, seed, setting=None, *, random_effect=False,
             beta=(1.0, 1.0, 1.0), rho1=0.5, r1=1.0,
             alpha=None, q=None, rate=None,
             frailty_dist='gamma', frailty_params=(2.0, 0.5),
             censor=(2.0, 4.0), n_grid=200, ode_kw=None):
    """Simulate ``N`` recurrent-event trajectories (long format).

    Parameters
    ----------
    N, seed : int
    setting : {1,2,3,4} or None
        Preset true rate. Use ``None`` with ``alpha``/``q``/``rate`` for a
        custom specification.
    random_effect : bool
        Turn the frailty on (``xi ~ frailty_dist``) or off (``xi = 1``).
    beta : array-like
        Regression coefficients (number of covariates, default 3).
    alpha, q : callables or None
        Custom ``alpha(t)`` and ``q(u)``; the intensity ``mu'_x`` is obtained by
        integrating ``mu' = m q(mu) alpha(t)``. Ignored if ``setting``/``rate``.
    rate : callable or None
        Custom closed-form intensity ``rate(t, m)`` (``m = exp(x'beta)``).
    frailty_dist, frailty_params : see :func:`frailty`.
    censor : (a, b)
        Administrative censoring ``~ U(a, b)``.

    Returns a dict with ``x``, ``time``, ``delta`` (1 event / 0 censoring),
    ``id`` and ``rho1``/``r1``.
    """
    if setting is None and rate is None and alpha is None and q is None:
        raise ValueError('specify a preset `setting`, a `rate`, or `alpha`/`q`')

    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    p = beta.size

    # covariates: (p-1) clipped standard normals + one Bernoulli(0.5)
    cols = [np.clip(rng.standard_normal(N), -1.0, 1.0) for _ in range(p - 1)]
    cols.append((rng.random(N) < 0.5).astype(float))
    x = np.column_stack(cols)

    xi = frailty(N, rng, random_effect, frailty_dist, frailty_params)
    a0, b0 = censor
    u = a0 + (b0 - a0) * rng.random(N)

    make_lam = _intensity_factory(setting, rate, alpha, q, rho1, ode_kw)

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
