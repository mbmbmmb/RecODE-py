"""Inference (standard errors) for :func:`ode_unify.estimator.estimate` fits.

Kept separate from estimation because it dominates the runtime: point
estimation takes seconds, while the frailty resampling below runs 150-800
perturbed score evaluations. The method is chosen automatically from the
fitted model:

* **no random effect** (cox / aft / ltm) -- closed form: invert the empirical
  Fisher information built from per-subject scores.
* **random_effect cox** -- closed-form sandwich adjustment for ``beta``
  (``inference_beta``); the spline-coefficient SEs (needed only for
  functional-parameter bands) come from the resampling routine and are computed
  when ``spline_se=True``.
* **random_effect aft / ltm** -- resampling (Zeng & Lin 2008) automatically;
  it yields ``beta`` and spline SEs together.

Each branch replicates the corresponding per-model ``inference()`` exactly
(same score kernels, reductions and resampling RNG seeds), so the SEs are
numerically identical to the standalone modules.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from ._engine.cox.objective_func_inf import objective_func_inf as _cox_obj_inf
from ._engine.aft.objective_func_beta import objective_func_beta as _aft_obj_beta
from ._engine.aft.objective_func_sieve import objective_func_sieve as _aft_obj_sieve
from ._engine.npmle.objective_func import objective_func as _npmle_obj
from ._engine.ltm.objective_func_beta import objective_func_beta as _ltm_obj_beta
from ._engine.ltm.objective_func_sieve import objective_func_sieve as _ltm_obj_sieve
from ._engine.random_effect.cox.inference import inference as _re_cox_inf
from ._engine.random_effect.cox.inference_beta import inference_beta as _re_cox_inf_beta
from ._engine.random_effect.aft_rec.inference import inference as _re_aft_inf
from ._engine.random_effect.ltm.inference import inference as _re_ltm_inf

from .estimator import Estimate, _unpack


# --------------------------------------------------------------------------- #
# closed-form branches (no random effect)
# --------------------------------------------------------------------------- #

def _inf_cox(est, data, order):
    x, time, delta, id_vec = _unpack(data, order)
    r = est.raw
    _, grad = _cox_obj_inf(r['est_r'].ravel(), x, time, delta, id_vec,
                           r['knots'].ravel(), int(r['k']))
    return np.sqrt(np.diag(np.linalg.inv(grad.T @ grad)))


def _inf_aft(est, data, order):
    x, time, delta, id_vec = _unpack(data, order)
    r = est.raw
    est_r = r['est_r'].ravel()
    p = int(r['p'])
    knots, k = r['knots'].ravel(), int(r['k'])
    beta, theta = est_r[:p], est_r[p:]
    _, gb = _aft_obj_beta(beta, x, time, delta, id_vec, theta, knots, k, ci=True)
    _, gs = _aft_obj_sieve(theta, x, time, delta, id_vec, beta, knots, k,
                           ci=True, forward=True)
    grad = np.concatenate([gb, gs], axis=1)
    nz = np.any(grad != 0, axis=0)
    try:
        se_r = np.sqrt(np.diag(np.linalg.inv(grad[:, nz].T @ grad[:, nz])))
        se = np.zeros(grad.shape[1])
        se[nz] = se_r
    except np.linalg.LinAlgError:
        se = np.full(grad.shape[1], np.nan)
    return se


def _inf_npmle(est, data, order):
    x, time, delta, id_vec = _unpack(data, order)
    r = est.raw
    rho1 = float(np.asarray(r.get('rho1', 0.5)).ravel()[0])
    r1 = float(np.asarray(r.get('r1', 1.0)).ravel()[0])
    q = int(r['q'])
    _, grad = _npmle_obj(r['est_r'].ravel(), x, time, delta, id_vec, q,
                         r['knots'].ravel(), int(r['k']), rho1, r1, ci=True)
    nz = np.any(grad != 0.0, axis=0)
    try:
        inv = np.linalg.inv(grad[:, nz].T @ grad[:, nz])
        se_r = np.sqrt(np.abs(np.diag(inv)))
        se = np.zeros(grad.shape[1])
        se[nz] = se_r
    except np.linalg.LinAlgError:
        se = np.full(grad.shape[1], np.nan)
    return se


def _inf_ltm(est, data, order):
    x, time, delta, id_vec = _unpack(data, order)
    r = est.raw
    est_r = r['est_r'].ravel()
    p = int(r['p'])
    q_q = int(r['q_q'])
    knots_0, knots_q = r['knots_0'].ravel(), r['knots_q'].ravel()
    k0, kq = int(r['k0']), int(r['kq'])
    beta = est_r[1:p]
    theta = est_r[p:p + q_q]
    alpha = est_r[p + q_q:]
    _, gb = _ltm_obj_beta(beta, x, time, delta, id_vec, theta, alpha,
                          knots_0, knots_q, k0, kq, ci=True)
    _, gs = _ltm_obj_sieve(np.concatenate([theta, alpha]), x, time, delta,
                           id_vec, beta, knots_0, knots_q, k0, kq, ci=True)
    grad = np.concatenate([gb, gs], axis=1)
    M = grad.T @ grad
    eigvals, V = np.linalg.eigh(M)
    inv_fish = (V * (1.0 / np.maximum(eigvals, 1.0))) @ V.T
    return np.sqrt(np.abs(np.diag(inv_fish)))


# --------------------------------------------------------------------------- #
# random-effect branches (scratch-dir shims around the vendored inference)
# --------------------------------------------------------------------------- #

def _write_npz(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrs)


def _inf_re_cox(est, data, seed, data_setting, spline_se, order):
    x, time, delta, id_vec = _unpack(data, order)
    r = est.raw
    N = int(np.unique(id_vec).size)
    with tempfile.TemporaryDirectory(prefix='ode_unify_recox_') as tmp:
        _write_npz(os.path.join(tmp, 'data',
                   f'simudata_N{N}_seed{seed}_setting{data_setting}.npz'),
                   x=x, time=time, delta=delta, id=id_vec)
        _write_npz(os.path.join(tmp, 'res',
                   f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz'),
                   est_r=r['est_r'].reshape(-1, 1), k=int(r['k']),
                   l=float(r['l']), p=int(r['p']), q=int(r['q']),
                   knots=r['knots'].ravel())
        fish_beta = _re_cox_inf_beta(N, seed, data_setting, root=tmp)
        se_beta = np.sqrt(np.abs(np.diag(fish_beta)))
        se_all = None
        if spline_se:
            fish_spline = _re_cox_inf(N, seed, data_setting, root=tmp)
            se_all = np.sqrt(np.abs(np.diag(fish_spline)))
    return se_beta, se_all


def _inf_re_aft(est, data, seed, data_setting, order):
    x, time, delta, id_vec = _unpack(data, order)
    r = est.raw
    N = int(np.unique(id_vec).size)
    ks = est.knots_setting
    with tempfile.TemporaryDirectory(prefix='ode_unify_reaft_') as tmp:
        _write_npz(os.path.join(tmp, 'data',
                   f'simudata_N{N}_seed{seed}_setting{data_setting}.npz'),
                   x=x, time=time, delta=delta, id=id_vec)
        _write_npz(os.path.join(tmp, 'res',
                   f'res_aft_N{N}_seed{seed}_setting{data_setting}_knots{ks}.npz'),
                   est_r=r['est_r'].reshape(-1, 1), k=int(r['k']),
                   l=int(r['l']), p=int(r['p']), q=int(r['q']),
                   knots=r['knots'].ravel())
        se_all = _re_aft_inf(N, seed, data_setting, ks, root=tmp).ravel()
    return se_all


def _inf_re_ltm(est, data, seed, data_setting, order):
    x, time, delta, id_vec = _unpack(data, order)
    r = est.raw
    N = int(np.unique(id_vec).size)
    ks = est.knots_setting
    sub, prefix = (('cox_rec_rd', 'res_cox_N') if data_setting == 1
                   else ('aft_rd', 'res_aft_N'))
    with tempfile.TemporaryDirectory(prefix='ode_unify_reltm_') as tmp:
        _write_npz(os.path.join(tmp, 'data', sub,
                   f'simudata_N{N}_seed{seed}_setting{data_setting}.npz'),
                   x=x, time=time, delta=delta, id=id_vec)
        _write_npz(os.path.join(tmp, 'res', sub,
                   f'{prefix}{N}_seed{seed}_setting{data_setting}_knots{ks}.npz'),
                   est_r=r['est_r'].reshape(-1, 1), p=int(r['p']),
                   q_q=int(r['q_q']), q_0=int(r['q_0']),
                   knots_0=r['knots_0'].ravel(), knots_q=r['knots_q'].ravel(),
                   k0=int(r['k0']), kq=int(r['kq']),
                   l0=int(r['l0']), lq=int(r['lq']))
        fish = _re_ltm_inf(N, seed, data_setting, ks, root=tmp)
    return np.sqrt(np.abs(np.diag(fish)))


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def inference(est: Estimate, data, *, seed=None, data_setting=None,
              spline_se=True) -> Estimate:
    """Fill in standard errors / 95% Wald CIs on a fitted :class:`Estimate`.

    Parameters
    ----------
    est : Estimate
        Output of :func:`ode_unify.estimator.estimate` (modified in place and
        returned).
    data : dict
        The same data the model was fit on.
    seed : int or None
        RNG seed for the frailty resampling (defaults to ``est.seed``; the
        random-effect cox resampling uses its reference-fixed seed 0
        internally). Irrelevant for the closed-form branches.
    data_setting : int or None
        Only used by the random-effect ltm resampling, whose number of
        perturbations is setting-specific (800 for setting 1, 1000 for
        setting 2). Defaults: cox-type 1, aft-type 2, ltm 1.
    spline_se : bool
        For the random-effect cox only: also compute the resampling SEs of the
        spline coefficients (needed for baseline-hazard bands; slower). The
        closed-form ``beta`` SEs are always computed.
    """
    if not est.success:
        return est
    seed = est.seed if seed is None else int(seed)
    key = (est.estimator, est.random_effect)
    p = est.beta.size
    is_ltm = est.estimator == 'ltm'
    from .estimator import _LEGACY_ORDER
    order = 'C' if est.layout == 'uniform' else _LEGACY_ORDER[key]

    if key == ('cox', False):
        se_all = _inf_cox(est, data, order)
    elif key == ('aft', False):
        se_all = _inf_aft(est, data, order)
    elif key == ('npmle', False):
        se_all = _inf_npmle(est, data, order)
    elif key == ('ltm', False):
        se_all = _inf_ltm(est, data, order)
    elif key == ('cox', True):
        ds = 1 if data_setting is None else int(data_setting)
        se_beta, se_all = _inf_re_cox(est, data, seed, ds, spline_se, order)
        est.raw['se_beta'] = se_beta
        if se_all is not None:
            est.raw['se_all'] = se_all
            est.se_all = se_all
        est.se = se_beta[:p]
        est.ci_lower = est.beta - 1.96 * est.se
        est.ci_upper = est.beta + 1.96 * est.se
        return est
    elif key == ('aft', True):
        ds = 2 if data_setting is None else int(data_setting)
        se_all = _inf_re_aft(est, data, seed, ds, order)
    elif key == ('ltm', True):
        ds = 1 if data_setting is None else int(data_setting)
        se_all = _inf_re_ltm(est, data, seed, ds, order)
    else:
        raise ValueError(f'unsupported {key}')

    est.raw['se_all'] = np.asarray(se_all)
    est.se_all = np.asarray(se_all).ravel()
    if is_ltm:
        # se_all layout: [se_b2..se_bp, se_theta, se_alpha] -- no entry for the
        # identifiability-fixed first beta.
        est.se = np.concatenate([[np.nan], est.se_all[:p - 1]])
    else:
        est.se = est.se_all[:p]
    est.ci_lower = est.beta - 1.96 * est.se
    est.ci_upper = est.beta + 1.96 * est.se
    return est


def fit(data, *, estimator, random_effect=False, knots=None, ci=True,
        seed=0, data_setting=None, spline_se=True,
        layout='uniform') -> Estimate:
    """Convenience wrapper: :func:`estimate` then (if ``ci``) :func:`inference`."""
    from .estimator import estimate
    est = estimate(data, estimator=estimator, random_effect=random_effect,
                   knots=knots, seed=seed, layout=layout)
    if ci:
        est = inference(est, data, seed=seed, data_setting=data_setting,
                        spline_se=spline_se)
    return est
