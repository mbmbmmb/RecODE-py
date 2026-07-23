"""Unified point estimation for the RecurrentODE family.

:func:`estimate` fits any of the three estimators -- ``'cox'``, ``'aft'``,
``'ltm'`` -- with or without a gamma frailty (``random_effect``), **in memory**,
and returns only the point estimates (regression coefficients ``beta`` plus the
B-spline coefficients of the functional parameters). It is fast (seconds).

Standard errors are deliberately *not* computed here: the inference step
(closed-form Fisher inversion, closed-form frailty adjustment, or resampling)
can be orders of magnitude slower than the estimation itself, and many uses
(bias studies, exploratory fits, model comparison) never need it. Call
:func:`ode_unify.inference.inference` on the returned :class:`Estimate` when you
do, or use the one-call convenience wrapper :func:`ode_unify.fit`.

Each estimation core replicates the corresponding per-model ``main()``
orchestration exactly (same knot construction and optimiser settings), so the
point estimates are numerically identical to the standalone modules.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ._engine.common import augknt

from ._engine.cox.objective_func import objective_func as _cox_obj
from ._engine.aft.cox_rec import cox_rec as _aft_cox_rec
from ._engine.aft.mle import mle as _aft_mle
from ._engine.ltm.mle import mle as _ltm_mle
from ._engine.npmle.objective_func import objective_func as _npmle_obj
from ._engine.random_effect.cox.objective_func import objective_func as _re_cox_obj
from ._engine.random_effect.aft_rec.cox_rec import cox_rec as _re_aft_cox_rec
from ._engine.random_effect.aft_rec.mle import mle as _re_aft_mle
from ._engine.random_effect.ltm.mle import mle as _re_ltm_mle

_DEFAULT_KNOTS = {'cox': None, 'aft': 'quantile', 'npmle': 'equal', 'ltm': 'K4'}


@dataclass
class Estimate:
    """Point estimates plus everything the inference step needs.

    ``beta`` is the length-``p`` coefficient vector (index 0 pinned to 1.0 for
    the LTM family). ``spline`` holds the functional-parameter pieces
    (single-spline models: ``knots``/``k``/``coefs``; LTM: two splines).
    ``se``/``ci_lower``/``ci_upper`` are ``None`` until
    :func:`ode_unify.inference.inference` fills them. ``raw`` mirrors the
    per-model result-file schema (``est_r``, knots, orders, ...).
    """
    beta: np.ndarray
    spline: dict
    estimator: str
    random_effect: bool
    knots_setting: Optional[str]
    seed: int
    runtime: float
    layout: str = 'uniform'
    success: bool = True
    se: Optional[np.ndarray] = None
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None
    se_all: Optional[np.ndarray] = None
    raw: dict = field(default_factory=dict)


def _unpack(data, order='C'):
    # The memory order of x must match what the reference mains get from
    # np.load of their canonical generator files -- 'C' for the non-frailty
    # generators, 'F' for the frailty ones -- because BLAS results (and thus
    # optimizer paths) are layout-sensitive at the last-bit level.
    x = np.asarray(data['x'], dtype=float)
    x = np.asfortranarray(x) if order == 'F' else np.ascontiguousarray(x)
    time = np.asarray(data['time'], dtype=float).ravel()
    delta = np.asarray(data['delta'], dtype=float).ravel()
    id_vec = np.asarray(data['id']).ravel()
    return x, time, delta, id_vec


# --------------------------------------------------------------------------- #
# estimation cores (each mirrors its main() exactly)
# --------------------------------------------------------------------------- #

def _est_cox(data, knots_setting, order):
    x, time, delta, _ = _unpack(data, order)
    p = x.shape[1]
    k = 3
    l = x.shape[0] ** (1.0 / 7.0)
    num_breaks = int(np.floor(l)) + 1
    knots = augknt(np.linspace(0.0, float(np.max(time)), num_breaks), k)
    q = len(knots) - k
    t0 = _time.time()
    res = minimize(lambda r: _cox_obj(r, x, time, delta, knots, k),
                   np.zeros(p + q), jac=True, method='BFGS',
                   options={'maxiter': 500, 'gtol': 1e-6})
    runtime = _time.time() - t0
    return dict(est_r=res.x, p=p, k=k, q=q, l=l, knots=knots, runtime=runtime)


def _est_aft(data, knots_setting, order):
    x, time, delta, id_vec = _unpack(data, order)
    N = int(np.unique(id_vec).size)
    temp, beta0 = _aft_cox_rec(x, time, delta)
    k = 4
    l = int(np.ceil(N ** (1.0 / 5.0))) + 1
    if knots_setting == 'quantile':
        interior = np.quantile(temp, np.arange(1, l) / l)
        breaks = np.concatenate([[0.0], interior, [float(np.max(temp))]])
    elif knots_setting == 'equal':
        breaks = np.linspace(0.0, 2.0 * float(np.max(temp)), l + 1)
    else:
        raise ValueError(f'unknown knots_setting={knots_setting}')
    knots = augknt(breaks, k)
    p = x.shape[1]
    q = len(knots) - k
    r0 = np.zeros(p + q)
    r0[:p] = beta0
    est_r, runtime = _aft_mle(x, time, delta, id_vec, knots, k, r0, True)
    return dict(est_r=est_r, p=p, k=k, q=q, l=l, knots=knots, runtime=runtime)


def _est_npmle(data, knots_setting, order):
    x, time, delta, id_vec = _unpack(data, order)
    rho1 = float(np.asarray(data.get('rho1', 0.5)).ravel()[0])
    r1 = float(np.asarray(data.get('r1', 1.0)).ravel()[0])
    N = int(np.unique(id_vec).size)
    p = x.shape[1]
    k = 3
    l = int(np.ceil(N ** (1.0 / 5.0)) + 2)
    temp = time[delta > 0]
    if knots_setting == 'quantile':
        interior = np.quantile(temp, np.arange(1, l) / l)
        knots = augknt(np.concatenate([[0.0], interior,
                                       [float(np.max(time))]]), k)
    elif knots_setting == 'equal':
        knots = augknt(np.linspace(0.0, float(np.max(time)), l + 1), k)
    else:
        raise ValueError(f'unknown knots_setting={knots_setting}')
    q = knots.size - k
    t0 = _time.time()
    res = minimize(
        lambda r: _npmle_obj(r, x, time, delta, id_vec, q, knots, k,
                             rho1, r1, ci=False),
        np.zeros(p + q), jac=True, method='BFGS',
        options={'maxiter': 10000, 'gtol': 1e-3})
    runtime = _time.time() - t0
    return dict(est_r=res.x, p=p, k=k, q=q, l=l, knots=knots,
                rho1=rho1, r1=r1, runtime=runtime)


def _est_ltm(data, knots_setting, order):
    x, time, delta, id_vec = _unpack(data, order)
    t0 = _time.time()
    est, knots_0, knots_q, k0, kq, q_0, q_q = _ltm_mle(
        x, time, delta, id_vec, knots_setting)
    runtime = _time.time() - t0
    return dict(est_r=est[:-1], succ_ind=int(est[-1]), p=x.shape[1],
                q_0=q_0, q_q=q_q, knots_0=knots_0, knots_q=knots_q,
                k0=k0, kq=kq, runtime=runtime)


def _est_re_cox(data, knots_setting, order):
    x, time, delta, _ = _unpack(data, order)
    p = x.shape[1]
    k = 3
    l = x.shape[0] ** (1.0 / 7.0)
    knots = augknt(np.linspace(0.0, float(np.max(time)), int(l) + 1), k)
    q = len(knots) - k
    t0 = _time.time()
    res = minimize(lambda r: _re_cox_obj(r, x, time, delta, knots, k),
                   np.zeros(p + q), jac=True, method='BFGS',
                   options={'maxiter': 500})
    runtime = _time.time() - t0
    return dict(est_r=res.x, p=p, k=k, q=q, l=l, knots=knots, runtime=runtime)


def _est_re_aft(data, knots_setting, order):
    x, time, delta, id_vec = _unpack(data, order)
    temp, beta0 = _re_aft_cox_rec(x, time, delta)
    k = 4
    l = int(np.ceil(x.shape[0] ** (1.0 / 5.0)))
    if knots_setting == 'quantile':
        interior = np.quantile(temp, np.arange(1, l) / l)
        knots = augknt(np.concatenate([[0.0], interior, [float(np.max(temp))]]), k)
    elif knots_setting == 'equal':
        knots = augknt(np.linspace(0.0, 2.0 * float(np.max(temp)), l + 1), k)
    else:
        raise ValueError(f'unknown knots_setting={knots_setting}')
    p = x.shape[1]
    q = knots.size - k
    r0 = np.zeros(p + q)
    r0[:p] = beta0
    est_r, runtime = _re_aft_mle(x, time, delta, id_vec, knots, k, r0, True)
    return dict(est_r=est_r, p=p, k=k, q=q, l=l, knots=knots, runtime=runtime)


def _est_re_ltm(data, knots_setting, order):
    x, time, delta, _ = _unpack(data, order)
    t0 = _time.time()
    est, p, q_q, q_0, knots_0, knots_q, k0, kq, l0, lq = _re_ltm_mle(
        x, time, delta, knots_setting)
    runtime = _time.time() - t0
    return dict(est_r=est[:-1], succ_ind=int(est[-1]), p=p,
                q_q=q_q, q_0=q_0, knots_0=knots_0, knots_q=knots_q,
                k0=k0, kq=kq, l0=l0, lq=lq, runtime=runtime)


_CORES = {
    ('cox', False): (_est_cox, False),
    ('aft', False): (_est_aft, False),
    ('npmle', False): (_est_npmle, False),
    ('ltm', False): (_est_ltm, True),
    ('cox', True): (_est_re_cox, False),
    ('aft', True): (_est_re_aft, False),
    ('ltm', True): (_est_re_ltm, True),
}


# Memory order of x that each legacy per-model pipeline consumed (its
# generator file's np.load layout). Only used with layout='legacy'.
_LEGACY_ORDER = {
    ('cox', False): 'C', ('aft', False): 'C', ('ltm', False): 'C',
    ('npmle', False): 'F',
    ('cox', True): 'F', ('aft', True): 'F', ('ltm', True): 'F',
}


def estimate(data, *, estimator, random_effect=False, knots=None, seed=0,
             layout='uniform'):
    """Fit the model and return point estimates only (fast; no SEs).

    Parameters
    ----------
    data : dict
        Long-format data with ``x`` (n_rows, p), ``time``, ``delta``
        (1 event / 0 censoring) and ``id``.
    estimator : {'cox', 'aft', 'npmle', 'ltm'}
    random_effect : bool
        Fit the gamma-frailty version.
    knots : str or None
        Knot scheme: ``'quantile'``/``'equal'`` for aft, ``'K1'``..``'K4'`` for
        ltm (defaults: aft ``'quantile'``, ltm ``'K4'``). Ignored for cox.
    seed : int
        Stored on the result; used by the resampling inference of the frailty
        models (match it to the data seed to reproduce the reference pipelines).
    layout : {'uniform', 'legacy'}
        Memory layout of ``x`` fed to the numerics. ``'uniform'`` (default)
        uses C order for every estimator, so all ode_unify results are
        mutually consistent. ``'legacy'`` mirrors each old per-model pipeline's
        layout (C for the non-frailty models, Fortran for the frailty ones),
        reproducing the standalone modules bit-for-bit. The two layouts agree
        to optimizer tolerance (~1e-6 on estimates, far below the SEs).
    """
    key = (estimator, bool(random_effect))
    if key not in _CORES:
        raise ValueError(f'unsupported (estimator={estimator!r}, '
                         f'random_effect={random_effect})')
    if layout not in ('uniform', 'legacy'):
        raise ValueError(f'unknown layout={layout!r}')
    core, is_ltm = _CORES[key]
    if knots is None:
        knots = _DEFAULT_KNOTS[estimator]
    order = 'C' if layout == 'uniform' else _LEGACY_ORDER[key]

    raw = core(data, knots, order)

    est_r = np.asarray(raw['est_r']).ravel()
    p = int(raw['p'])
    beta = est_r[:p].copy()
    if is_ltm:
        beta[0] = 1.0
        spline = {'knots_0': raw['knots_0'], 'knots_q': raw['knots_q'],
                  'k0': int(raw['k0']), 'kq': int(raw['kq']),
                  'q_0': int(raw['q_0']), 'q_q': int(raw['q_q']),
                  'coefs_q': est_r[p:p + int(raw['q_q'])],
                  'coefs_alpha': est_r[p + int(raw['q_q']):]}
    else:
        spline = {'knots': raw['knots'], 'k': int(raw['k']),
                  'coefs': est_r[p:]}

    return Estimate(
        beta=beta, spline=spline, estimator=estimator,
        random_effect=bool(random_effect), knots_setting=knots,
        seed=int(seed), runtime=float(raw.get('runtime', 0.0)),
        layout=layout,
        success=bool(raw.get('succ_ind', 1)),
        raw={k: np.asarray(v) for k, v in raw.items()},
    )
