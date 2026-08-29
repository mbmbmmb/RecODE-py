"""Regression test for the constrained sieve step's solver cascade.

``solve_sieve_step`` tries SLSQP first and falls back to trust-constr when the
line search breaks down numerically. After the time-transform clamp and the
``ode_guard`` signature fix, no fit in any current study still triggers that
fallback -- which is the point of the fixes, but leaves the fallback path
unexercised by the studies themselves. These tests drive each branch directly on
a synthetic quadratic under the same shape of linear equality constraint that
``alpha(2) = 1`` imposes, so a regression in the plumbing fails here rather than
silently reverting to "one solver, no recovery".

Run either way (no pytest required):
    python ode_unify/tests/test_solve_sieve_step.py
    python -m pytest ode_unify/tests/test_solve_sieve_step.py
"""
import os
import sys

import numpy as np
from scipy.optimize import LinearConstraint

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ode_unify._engine.common import (ODEIntegrationError,      # noqa: E402
                                      SIEVE_COEF_BOUND, solve_sieve_step)

N = 6
A = np.ones((1, N))                       # sum(x) == 0
TARGET = np.arange(1.0, N + 1)
TARGET -= TARGET.mean()                   # feasible: sums to zero
CON = LinearConstraint(A, 0.0, 0.0)
X0 = np.zeros(N)


def _fun(r):
    d = np.asarray(r, float) - TARGET
    return float(d @ d), 2 * d


def _residual(x):
    return abs(float(np.ravel(A @ x)[0]))


def test_healthy_objective_uses_slsqp():
    res, used_trust = solve_sieve_step(_fun, X0, CON)
    assert not used_trust
    assert np.allclose(res.x, TARGET, atol=1e-6)
    assert _residual(res.x) < 1e-8


def _falls_back(exc):
    """Both failure modes seen in practice must switch solvers, not propagate."""
    calls = [0]

    def flaky(r):
        calls[0] += 1
        if calls[0] == 2:                 # inside SLSQP, not during the retry
            raise exc
        return _fun(r)

    res, used_trust = solve_sieve_step(flaky, X0, CON)
    assert used_trust, f'fallback did not fire on {type(exc).__name__}'
    assert np.allclose(res.x, TARGET, atol=1e-6)
    assert _residual(res.x) < 1e-8


def test_falls_back_on_ode_integration_error():
    _falls_back(ODEIntegrationError('synthetic integration failure'))


def test_falls_back_on_t_eval_value_error():
    _falls_back(ValueError('Values in `t_eval` are not within `t_span`.'))


def test_prefer_trust_skips_slsqp():
    """The flag is sticky: once a fit needs the robust solver it keeps it."""
    res, used_trust = solve_sieve_step(_fun, X0, CON, prefer_trust=True)
    assert used_trust
    assert np.allclose(res.x, TARGET, atol=1e-6)


def test_slsqp_path_honours_the_coefficient_box():
    far = 1e4 * np.arange(1.0, N + 1)
    far -= far.mean()                     # optimum far outside the box

    def wide(r):
        d = np.asarray(r, float) - far
        return float(d @ d), 2 * d

    res, _ = solve_sieve_step(wide, X0, CON)
    assert np.abs(res.x).max() <= SIEVE_COEF_BOUND + 1e-6


def test_guard_resolves_ci_from_each_signature():
    """objective_func_beta takes ci 11th, objective_func_sieve 10th. A fixed
    positional index reads kq for the beta objective -- a non-zero int, i.e.
    always 'ci=True' -- which silently disabled the guard on that step."""
    import inspect

    from ode_unify._engine.ltm.objective_func_beta import objective_func_beta
    from ode_unify._engine.ltm.objective_func_sieve import objective_func_sieve

    for fn, expected in ((objective_func_sieve, 9), (objective_func_beta, 10)):
        params = list(inspect.signature(fn).parameters)
        assert params.index('ci') - 1 == expected


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS  {name}')
            except AssertionError as e:
                fails += 1
                print(f'FAIL  {name}: {e}')
    print('\n%s' % ('all cascade paths verified' if not fails
                    else f'{fails} test(s) failed'))
    sys.exit(1 if fails else 0)
