"""MIMIC-III analysis of Section 6 of the paper: ODE-Cox and ODE-Flex fits, the likelihood
ratio statistic, and the coefficient table, at a fixed administrative horizon (two years).

Specification:

* cohort: every patient's first ICU admission, censored at min(horizon, death);
  20 covariates (averaged max/min pairs, single-valued clinical variables,
  comorbidities, admission-type dummies, missingness indicators);
* time on the raw scale (days); quadratic B-splines (order 3) for log alpha and
  log q, ceil(m^{1/7}) intervals each, knots at quantiles (event times for alpha,
  ODE-Cox mu-hat for q);
* ODE-Cox: q == 1, every regression coefficient free, closed-form sandwich SEs
  with subject-level score aggregation (gamma-frailty robust);
* ODE-Flex: the anchor covariate is RESCALED by its own ODE-Cox estimate c0, so
  beta_anchor == 1 reproduces the Cox linear predictor, and log alpha is pinned at
  its own ODE-Cox value at the median event time. Both constraints are satisfied
  by the ODE-Cox solution, so the null is a feasible point and LR >= 0;
* LR = 2m (negll_cox - negll_flex); the paper calibrates it by the parametric
  bootstrap in bootstrap_final_lr.py (the chi-square p-value stored here is not used);
* Zeng-Lin resampling sandwich for ODE-Flex, restricted to the null space of the
  alpha anchor, giving pointwise 95% bands for log q and log alpha.

Usage:  python fit_final_horizon.py [tag] [B]
        tag: suffix of <OUT>/cohort_long_scaled_<tag>.npz (default 2y);
        B: resampling draws for the bands (default 4000; the paper re-estimates the
        bands with resample_bands.py, B = 8000, seed 1).
"""
from __future__ import annotations

import os
import sys
import time as _time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, minimize
from scipy.stats import chi2, norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from ode_unify._engine.common import augknt, spcol, solve_sieve_step     # noqa: E402
from ode_unify._engine.random_effect.cox.baseline_hazard_func import (   # noqa: E402
    baseline_hazard_func)
from ode_unify._engine.random_effect.cox.baseline_hazard_func_v import (  # noqa: E402
    baseline_hazard_func_v)
from ode_unify._engine.random_effect.cox.inference_beta import _v_at_times  # noqa: E402
from ode_unify._engine.random_effect.ltm.inference_objective_func_sieve import (  # noqa: E402,E501
    inference_objective_func_sieve as IOFS)
from ode_unify._engine.random_effect.ltm.objective_func_beta import (   # noqa: E402
    objective_func_beta as OFB)
from ode_unify._engine.random_effect.ltm.objective_func_sieve import (  # noqa: E402
    objective_func_sieve as OFS)

# The engine integrates its ODEs adaptively at rtol=1e-6. Summed over ~35,000
# patients that leaves ~0.05 units of noise in the log pseudo-likelihood, enough to
# stall line searches several LR units short of the optimum (finite differences of
# the objective diverged as the step shrank). Tighten the tolerance for these fits.
import ode_unify._engine.common as _common                                # noqa: E402
import ode_unify._engine.random_effect.ltm.inference_objective_func_sieve as _m_iofs  # noqa: E402,E501
import ode_unify._engine.random_effect.ltm.objective_func_beta as _m_ofb  # noqa: E402
import ode_unify._engine.random_effect.ltm.objective_func_sieve as _m_ofs  # noqa: E402

_ode0, _ivp0 = _common.solve_ode, solve_ivp


def _tight_ode(*a, **k):
    k.update(rtol=1e-10, atol=1e-12)
    return _ode0(*a, **k)


def _tight_ivp(*a, **k):
    k.update(rtol=1e-10, atol=1e-12)
    return _ivp0(*a, **k)


_m_ofs.solve_ode = _m_ofb.solve_ode = _tight_ode
_m_iofs.solve_ivp = _tight_ivp

from config import OUT                                                  # noqa: E402
ANCHOR = 'bun_mean'
K = 3
ITERS = 150
TOL = 1e-5
SEED = 0
STARTS = 6


def load(tag):
    """20-covariate design on raw time (days); anchor covariate first."""
    z = np.load(os.path.join(OUT, f'cohort_long_scaled_{tag}.npz'), allow_pickle=True)
    cols = [str(c) for c in z['x_cols']]
    x = np.asarray(z['x'], float)
    keep = [i for i, c in enumerate(cols)
            if not c.endswith('_spread') and c != 'Adm_type']
    names = [cols[i] for i in keep]
    X = x[:, keep]
    a = x[:, cols.index('Adm_type')]
    X = np.column_stack([X, (a == 1).astype(float), (a == 2).astype(float)])
    names += ['Adm_type_scheduled', 'Adm_type_unscheduled']
    for j in range(X.shape[1]):
        m = ~np.isfinite(X[:, j])
        if m.any():
            X[m, j] = np.nanmedian(X[:, j])
    ai = names.index(ANCHOR)
    order = [ai] + [j for j in range(len(names)) if j != ai]
    X = np.ascontiguousarray(X[:, order])
    names = [names[j] for j in order]
    t = np.asarray(z['time'], float).ravel()
    delta = np.asarray(z['delta'], int).ravel()
    idv = np.asarray(z['id'], int).ravel()
    # consecutive 1..N ids, as the per-subject score scatter requires
    _, idv = np.unique(idv, return_inverse=True)
    return X, t, delta, idv + 1, names


def quantile_knots(values, lo, hi, n_int):
    inner = np.quantile(values, np.linspace(0, 1, n_int + 1))[1:-1]
    return augknt(np.concatenate([[lo], inner, [hi]]), K)


def alternate(X, t, delta, beta, theta, alpha, kn0, knq, free_theta, cons_row=None,
              label=''):
    """Alternating sieve (SLSQP / trust-constr) and beta (BFGS) steps."""
    q_q = theta.size
    # Work on the total (not per-patient) scale: one LR unit is a change of
    # 1/(2m) ~ 1e-5 in the per-patient objective, below what SLSQP/BFGS resolve
    # with their default tolerances, which let the fit stop a few LR units short
    # (or not leave a null starting point at all).
    sc = float((np.asarray(delta) == 0).sum())
    use_trust = False
    t0 = _time.time()
    for it in range(ITERS):
        bp, tp, ap = beta.copy(), theta.copy(), alpha.copy()

        def f(v):
            th = v[:q_q] if free_theta else theta
            al = v[q_q:] if free_theta else v
            val, g = OFS(np.concatenate([th, al]), X, t, delta, beta,
                         kn0, knq, K, K)
            return sc * val, sc * (g if free_theta else g[q_q:])

        v0 = np.concatenate([theta, alpha]) if free_theta else alpha.copy()
        cons = []
        if cons_row is not None:
            row = np.zeros(v0.size)
            row[q_q:] = cons_row[0]
            cons = [LinearConstraint(row.reshape(1, -1), cons_row[1], cons_row[1])]
        r, use_trust = solve_sieve_step(f, v0, cons, prefer_trust=use_trust,
                                        slsqp_options={'maxiter': 200, 'ftol': 1e-10})
        if free_theta:
            theta, alpha = r.x[:q_q], r.x[q_q:]
        else:
            alpha = r.x
        def fb(bb):
            v, gb = OFB(bb, X, t, delta, theta, alpha, kn0, knq, K, K)
            return sc * v, sc * gb
        rb = minimize(fb, beta, jac=True, method='BFGS', options={'maxiter': 500, 'gtol': 1e-5})
        beta = rb.x
        step = max(np.abs(beta - bp).max(), np.abs(theta - tp).max(),
                   np.abs(alpha - ap).max())
        if it % 5 == 0:
            val, _ = OFS(np.concatenate([theta, alpha]), X, t, delta, beta,
                         kn0, knq, K, K)
            print(f'  [{label}] it {it:3d} negll={val:.9f} step={step:.2e} '
                  f'{_time.time() - t0:.0f}s', flush=True)
        val_now, _ = OFS(np.concatenate([theta, alpha]), X, t, delta, beta,
                         kn0, knq, K, K)
        if it > 0 and step < TOL and abs(val_prev - val_now) * sc < 1e-4:
            break
        val_prev = val_now

    val, _ = OFS(np.concatenate([theta, alpha]), X, t, delta, beta, kn0, knq, K, K)
    return dict(beta=beta, theta=theta, alpha=alpha, negll=float(val),
                iters=it + 1, converged=bool(step < TOL), step=float(step))


def _flex_start(args):
    X, t, delta, beta0, alpha0, kn0, knq, cons_row, k = args
    q_q = len(knq) - K
    th0 = (np.zeros(q_q) if k == 0
           else np.random.default_rng(k).normal(0.0, 0.15, q_q))
    return alternate(X, t, delta, beta0.copy(), th0, alpha0.copy(), kn0, knq,
                     free_theta=True, cons_row=cons_row, label=f'flex-s{k}')


def fit_flex(X, t, delta, beta0, alpha0, kn0, knq, cons_row, starts=1, workers=1):
    """ODE-Flex from the null (k=0) and starts-1 perturbed theta starts; best fit.

    Different starts end within ~0.5 LR units of each other once the ODE
    tolerance is tight, but a start at theta = 0 occasionally stops early, so
    the observed fit and every bootstrap replicate take the best of several.
    """
    jobs = [(X, t, delta, beta0, alpha0, kn0, knq, cons_row, k) for k in range(starts)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(workers, starts)) as ex:
            fits = list(ex.map(_flex_start, jobs))
    else:
        fits = [_flex_start(j) for j in jobs]
    best = min(fits, key=lambda r: r['negll'])
    best['all_negll'] = [r['negll'] for r in fits]
    return best


def cox_sandwich(X, t, delta, idv, beta, alpha, kn0):
    """Closed-form sandwich for ODE-Cox beta (random/cox/inference_beta.py body)."""
    N = int(idv.max())
    p = X.shape[1]
    idx = np.where(delta == 0)[0]
    tau, x_tau = t[idx], X[idx]
    mc = np.exp(x_tau @ beta)
    v_val = _v_at_times(t, tau, x_tau, mc)
    u, bin_idx = np.unique(tau, return_inverse=True)
    tspan = np.concatenate([[1e-12], u])
    sol = solve_ivp(lambda s, y: baseline_hazard_func(s, alpha, kn0, K),
                    (tspan[0], tspan[-1]), [0.0], t_eval=tspan, method='RK45',
                    rtol=1e-8, atol=1e-10)
    cum = sol.y[0, 1:][bin_idx]
    sol_v = solve_ivp(lambda s, y: baseline_hazard_func_v(
        np.atleast_1d(s), tau, x_tau, beta, alpha, kn0, K).ravel(),
        (tspan[0], tspan[-1]), np.zeros(p), t_eval=tspan, method='RK45',
        rtol=1e-8, atol=1e-10)
    cum_v = sol_v.y[:, 1:].T[bin_idx]
    b1 = np.zeros((N, p))
    np.add.at(b1, idv - 1, (X - v_val) * delta[:, None])
    b2 = mc[:, None] * (-x_tau * cum[:, None] + cum_v)
    Bm = np.zeros((N, p))
    np.add.at(Bm, idv[idx] - 1, b2)
    Bm = b1 + Bm
    Bm = Bm.T @ Bm / N
    A = (-(x_tau * (mc * cum)[:, None]).T @ x_tau
         + cum_v.T @ (x_tau * mc[:, None])) / N
    return np.linalg.solve(A, Bm @ np.linalg.inv(A.T)) / N


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else '2y'
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    X, t, delta, idv, names = load(tag)
    m = int((delta == 0).sum())
    p = X.shape[1]
    n_int = int(np.ceil(m ** (1 / 7)))
    horizon = float(t.max())
    kn0 = quantile_knots(t[delta == 1], 0.0, horizon, n_int)
    q_0 = len(kn0) - K
    print(f'[{tag}] m={m} rows={len(t)} events={int(delta.sum())} p={p} '
          f'intervals={n_int} q_0={q_0}', flush=True)

    # ---- ODE-Cox, every coefficient free: prepend a zero column so that the
    # fixed leading coefficient (=1) multiplies nothing.
    X0 = np.column_stack([np.zeros(len(t)), X])
    knq_dummy = augknt(np.array([0.0, 1.0]), K)          # theta == 0 -> q == 1
    th0 = np.zeros(len(knq_dummy) - K)
    tm = time_t = _time.time()
    cox = alternate(X0, t, delta, np.zeros(p), th0, np.zeros(q_0), kn0, knq_dummy,
                    free_theta=False, label='cox')
    b_cox, a_cox = cox['beta'], cox['alpha']
    print(f'[{tag}] ODE-Cox negll={cox["negll"]:.9f} conv={cox["converged"]} '
          f'iters={cox["iters"]} {_time.time() - time_t:.0f}s', flush=True)
    S_cox = cox_sandwich(X, t, delta, idv, b_cox, a_cox, kn0)
    se_cox = np.sqrt(np.diag(S_cox))

    # ODE-Cox mean function at the censoring rows -> q knots
    mu_cox = np.zeros(len(t))
    u, bi = np.unique(t, return_inverse=True)
    sol = solve_ivp(lambda s, y: baseline_hazard_func(s, a_cox, kn0, K),
                    (0.0, u[-1]), [0.0], t_eval=u, method='RK45', rtol=1e-8, atol=1e-10)
    mu_cox = sol.y[0][bi] * np.exp(X @ b_cox)
    mu_c = mu_cox[delta == 0]
    knq = quantile_knots(mu_c, 0.0, 1.05 * float(mu_cox.max()), n_int)
    q_q = len(knq) - K

    # ---- ODE-Flex under the rescaled anchor
    c0 = float(b_cox[0])
    Xs = X.copy()
    Xs[:, 0] *= c0
    t_anchor = float(np.median(t[delta == 1]))
    row = spcol(kn0, K, np.array([t_anchor]))[0]
    anchor_val = float(row @ a_cox)
    beta0 = b_cox[1:].copy()
    # the Cox solution is feasible: check its objective in the Flex parameterisation
    nl_seed, _ = OFS(np.concatenate([np.zeros(q_q), a_cox]), Xs, t, delta, beta0,
                     kn0, knq, K, K)
    print(f'[{tag}] c0={c0:.4f} t_anchor={t_anchor:.2f}d  seed negll={nl_seed:.9f} '
          f'(cox {cox["negll"]:.9f})', flush=True)
    time_t = _time.time()
    flex = fit_flex(Xs, t, delta, beta0, a_cox, kn0, knq, (row, anchor_val),
                    starts=STARTS, workers=STARTS)
    print(f'[{tag}] LR by start: '
          f'{np.round(2 * m * (nl_seed - np.array(flex["all_negll"])), 2)}', flush=True)
    LR = 2 * m * (nl_seed - flex['negll'])
    df = q_q - 1
    pval = float(chi2.sf(LR, df))
    print(f'[{tag}] ODE-Flex negll={flex["negll"]:.9f} conv={flex["converged"]} '
          f'iters={flex["iters"]} {_time.time() - time_t:.0f}s', flush=True)
    print(f'[{tag}] LR={LR:.3f} df={df} p={pval:.3g}', flush=True)

    # ---- Zeng-Lin resampling sandwich for ODE-Flex, on the constraint null space
    th, al, be = flex['theta'], flex['alpha'], flex['beta']
    est = np.concatenate([be, th, al])                     # (p-1)+q_q+q_0
    d = est.size
    crow = np.zeros(d)
    crow[p - 1 + q_q:] = row
    Nul = null_space(crow.reshape(1, -1))                  # d x (d-1)
    Nsub = int(idv.max())
    g = IOFS(np.concatenate([th, al]), Xs, t, delta, idv, be, kn0, knq, K, K, ci=True)
    V = g.T @ g / Nsub
    rng = np.random.default_rng(SEED)
    Zr = rng.standard_normal((Nul.shape[1], B))
    Y = np.zeros((B, d))
    time_t = _time.time()
    for i in range(B):
        e = est + Nul @ Zr[:, i] / np.sqrt(Nsub)
        Y[i] = IOFS(np.concatenate([e[p - 1:p - 1 + q_q], e[p - 1 + q_q:]]), Xs, t,
                    delta, idv, e[:p - 1], kn0, knq, K, K, ci=False) / np.sqrt(Nsub)
        if (i + 1) % 250 == 0:
            el = _time.time() - time_t
            print(f'  resample {i + 1}/{B}  {el:.0f}s  (eta {el / (i + 1) * (B - i - 1):.0f}s)',
                  flush=True)
    Y -= Y.mean(0)
    Ar = np.linalg.solve(Zr @ Zr.T, Zr @ Y)                # (d-1) x d  ~ (D N)'
    H = Ar @ Nul
    H = 0.5 * (H + H.T)
    Hi = np.linalg.inv(H)
    S_red = Hi @ (Nul.T @ V @ Nul) @ Hi / Nsub
    Sigma = Nul @ S_red @ Nul.T
    se_flex = np.sqrt(np.abs(np.diag(Sigma)))

    # bands
    gu = np.linspace(0.0, float(np.quantile(mu_c, 0.995)), 300)
    Bq = spcol(knq, K, gu)
    S_th = Sigma[p - 1:p - 1 + q_q, p - 1:p - 1 + q_q]
    lq = Bq @ th
    se_q = np.sqrt(np.abs(np.einsum('ij,jk,ik->i', Bq, S_th, Bq)))
    gt = np.linspace(0.0, horizon, 400)
    B0 = spcol(kn0, K, gt)
    S_al = Sigma[p - 1 + q_q:, p - 1 + q_q:]
    la = B0 @ al
    se_a = np.sqrt(np.abs(np.einsum('ij,jk,ik->i', B0, S_al, B0)))
    la_cox = B0 @ a_cox
    z = norm.ppf(0.975)
    excl = ((lq - z * se_q) > 0) | ((lq + z * se_q) < 0)
    print(f'[{tag}] log q band excludes 0 on {100 * excl.mean():.1f}% of '
          f'[0, q99.5(mu)={gu[-1]:.3f}]; median half-width {z * np.median(se_q):.3f}',
          flush=True)

    # coefficient tables (Flex beta reported on the original covariate scale:
    # beta_anchor = c0 and the free betas unchanged, since only x_anchor was rescaled)
    b_flex = np.concatenate([[c0], be])
    se_bflex = np.concatenate([[np.nan], se_flex[:p - 1]])
    tab = pd.DataFrame(dict(feature=names, cox_est=b_cox, cox_se=se_cox,
                            cox_z=b_cox / se_cox,
                            cox_p=2 * norm.sf(np.abs(b_cox / se_cox)),
                            flex_est=b_flex, flex_se=se_bflex,
                            flex_p=2 * norm.sf(np.abs(b_flex / se_bflex))))
    tab.to_csv(os.path.join(OUT, f'final_{tag}_coef.csv'), index=False)
    np.savez_compressed(
        os.path.join(OUT, f'final_{tag}.npz'),
        names=np.array(names), knots_0=kn0, knots_q=knq, k=K, c0=c0,
        t_anchor=t_anchor, beta_cox=b_cox, alpha_cox=a_cox, se_cox=se_cox,
        beta_flex=be, theta=th, alpha_flex=al, Sigma=Sigma, se_flex=se_flex,
        negll_cox=cox['negll'], negll_seed=nl_seed, negll_flex=flex['negll'],
        LR=LR, df=df, p=pval, m=m, events=int(delta.sum()), B=B,
        gu=gu, log_q=lq, se_q=se_q, gt=gt, log_alpha=la, se_alpha=se_a,
        log_alpha_cox=la_cox, mu_c=mu_c, horizon=horizon,
        cox_conv=cox['converged'], flex_conv=flex['converged'])
    print(tab.sort_values('cox_p').to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print(f'wrote final_{tag}.npz and final_{tag}_coef.csv  total {_time.time() - tm:.0f}s')


if __name__ == '__main__':
    main()
