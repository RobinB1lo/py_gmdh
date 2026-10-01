"""Penalized logistic regression for the logistic neurons and Platt scaling.

The solver fits the coefficients of a logistic model whose design matrix
already contains any constant column, minimizing

    f(w) = sum_i [log(1 + exp(z_i)) - t_i z_i] + ridge / 2 * ||w||^2,   z = X w,

where the targets ``t_i`` lie in ``[0, 1]`` (0/1 labels, or smoothed targets
for Platt scaling). Every coefficient, including the one of the constant
column, is penalized. The objective is strictly convex, so its minimizer is
unique.

The fit has two stages:

1. Newton's method with the Hessian ``X^T diag(p(1 - p)) X + ridge I``, a
   backtracking (Armijo) line search extended by step doubling, and columns
   rescaled to unit maximum magnitude. This converges in a few iterations for
   ordinary designs.
2. If the gradient with respect to the original coefficients is still large
   afterwards, which happens when a column mixes values of very different
   magnitude (for example negative powers of inputs close to zero), the
   result is refined by limited-memory BFGS in the original coordinates.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

try:  # keep warning filters written for scikit-learn's class working
    from sklearn.exceptions import ConvergenceWarning as _WarningBase
except ImportError:  # pragma: no cover
    _WarningBase = UserWarning

_MIN_RIDGE = 1e-12
_ARMIJO = 1e-4
_MIN_STEP = 1e-10
_MAX_DOUBLINGS = 80
_GRADIENT_TOL = 1e-6
_LBFGS_MEMORY = 10
_LBFGS_MAX_ITER = 1000
_LBFGS_STALL = 5


class ConvergenceWarning(_WarningBase):
    """Issued when an iterative fit stops before reaching its tolerance."""


def sigmoid(z):
    """Numerically stable logistic function ``1 / (1 + exp(-z))``."""
    with np.errstate(over="ignore"):  # exp(-z) = inf correctly gives 0
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class _Problem:
    """Objective and gradient of the penalized log-likelihood."""

    def __init__(self, X: np.ndarray, t: np.ndarray, penalty: np.ndarray) -> None:
        self.X, self.t, self.penalty = X, t, penalty

    def value(self, w: np.ndarray) -> float:
        z = self.X @ w
        return float(np.sum(np.logaddexp(0.0, z) - self.t * z) + 0.5 * (self.penalty * w) @ w)

    def gradient(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = sigmoid(self.X @ w)
        return self.X.T @ (p - self.t) + self.penalty * w, p


def _line_search(problem: _Problem, w: np.ndarray, d: np.ndarray, f: float,
                 slope: float) -> Tuple[np.ndarray, float]:
    """Step along ``-d``, where ``slope = g @ d > 0`` is the directional derivative.

    If the unit step does not increase the objective, the step is doubled for
    as long as the objective keeps decreasing, or does not change measurably
    yet. This handles plateaus on which a few observations with large values
    dominate the curvature while the minimum lies many orders of magnitude
    further along the same direction.
    Otherwise the step is halved until the Armijo condition holds. Returns
    ``(w, f)`` unchanged if no decrease is found.
    """
    f_new = problem.value(w - d)
    if f_new <= f:
        step = 1.0
        for _ in range(_MAX_DOUBLINGS):
            f_next = problem.value(w - 2.0 * step * d)
            # keep doubling while improving, or while the change is still
            # below floating-point resolution
            if not (f_next < f_new or f_next == f_new == f):
                break
            step, f_new = 2.0 * step, f_next
        if f_new < f:
            return w - step * d, f_new
    step = 0.5
    while step >= _MIN_STEP:
        f_new = problem.value(w - step * d)
        if f_new <= f - _ARMIJO * step * slope:
            return w - step * d, f_new
        step *= 0.5
    return w, f


def _newton_direction(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Solve ``H d = g`` after symmetric diagonal equilibration of ``H``."""
    diag = np.sqrt(np.clip(np.diag(H), np.finfo(float).tiny, None))
    Hs = H / diag[:, None] / diag[None, :]
    gs = g / diag
    try:
        L = np.linalg.cholesky(Hs)
        ds = np.linalg.solve(L.T, np.linalg.solve(L, gs))
    except np.linalg.LinAlgError:
        ds = np.linalg.lstsq(Hs, gs, rcond=None)[0]
    return ds / diag


def _newton(problem: _Problem, v: np.ndarray, max_iter: int,
            tol: float) -> Tuple[np.ndarray, float, bool]:
    X = problem.X
    f = problem.value(v)
    for _ in range(max_iter):
        g, p = problem.gradient(v)
        H = (X * (p * (1.0 - p))[:, None]).T @ X + np.diag(problem.penalty)
        d = _newton_direction(H, g)
        decrement = float(g @ d)
        if not np.isfinite(decrement) or decrement <= 0:
            d, decrement = g, float(g @ g)
        v_new, f_new = _line_search(problem, v, d, f, decrement)
        threshold = tol * (1.0 + abs(f))
        if f - f_new <= threshold:
            converged = 0.5 * decrement <= threshold or not np.any(g)
            return v_new, f_new, converged
        v, f = v_new, f_new
    return v, f, False


def _lbfgs(problem: _Problem, w: np.ndarray, f: float, tol: float) -> Tuple[np.ndarray, float]:
    """Limited-memory BFGS (two-loop recursion) from ``w``; never increases ``f``."""
    g, _ = problem.gradient(w)
    s_hist, y_hist = [], []
    stalled = 0
    for _ in range(_LBFGS_MAX_ITER):
        if np.max(np.abs(g)) <= _GRADIENT_TOL * len(problem.t):
            break
        q = g.copy()
        alphas = []
        for s, y in zip(reversed(s_hist), reversed(y_hist)):
            alpha = (s @ q) / (y @ s)
            alphas.append(alpha)
            q -= alpha * y
        if s_hist:
            q *= (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1])
        else:
            reach = np.max(np.abs(problem.X @ q))
            if not reach > 0:
                break
            q /= reach  # first trial step changes no logit by more than one
        for (s, y), alpha in zip(zip(s_hist, y_hist), reversed(alphas)):
            q += s * (alpha - (y @ q) / (y @ s))
        slope = float(g @ q)
        if not (np.isfinite(slope) and slope > 0):
            if not s_hist:
                break
            s_hist, y_hist = [], []
            continue
        w_new, f_new = _line_search(problem, w, q, f, slope)
        if f_new >= f:
            if not s_hist:
                break
            s_hist, y_hist = [], []
            continue
        g_new, _ = problem.gradient(w_new)
        s, y = w_new - w, g_new - g
        if s @ y > 1e-12 * np.sqrt((s @ s) * (y @ y)):
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > _LBFGS_MEMORY:
                s_hist.pop(0)
                y_hist.pop(0)
        stalled = stalled + 1 if f - f_new <= tol * (1.0 + abs(f)) else 0
        w, f, g = w_new, f_new, g_new
        if stalled >= _LBFGS_STALL:
            break
    return w, f


def logistic_fit(X: np.ndarray, t: np.ndarray, ridge: float, max_iter: int = 100,
                 tol: float = 1e-12, start: Optional[np.ndarray] = None) -> np.ndarray:
    """Coefficients of the L2-penalized logistic regression of ``t`` on ``X``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_coefficients)
        Design matrix, including the constant column if one is wanted.
    t : ndarray of shape (n_samples,)
        Targets in ``[0, 1]``.
    ridge : float
        L2 penalty; values below ``1e-12`` are raised to ``1e-12`` so that the
        problem always has a unique, finite solution.
    max_iter : int, default=100
        Maximum number of Newton iterations.
    tol : float, default=1e-12
        Relative tolerance. Newton's method stops when both the decrease
        predicted by the quadratic model (half the Newton decrement
        ``g^T H^{-1} g``) and the decrease achieved by the line search are
        below ``tol * (1 + f)``.
    start : ndarray, optional
        Initial coefficients (zeros by default).

    Returns
    -------
    w : ndarray of shape (n_coefficients,)
    """
    n, q = X.shape
    ridge = max(float(ridge), _MIN_RIDGE)
    # Newton iterations on columns scaled to unit maximum magnitude, v = scale * w;
    # the penalty becomes ridge / scale**2, so the minimizer is unchanged.
    scale = np.max(np.abs(X), axis=0)
    scale[~(scale > 0)] = 1.0
    scaled = _Problem(X / scale, t, ridge / scale ** 2)
    v = np.zeros(q) if start is None else np.asarray(start, dtype=float) * scale
    v, f, converged = _newton(scaled, v, max_iter, tol)
    w = v / scale

    original = _Problem(X, t, np.full(q, ridge))
    g, _ = original.gradient(w)
    if np.max(np.abs(g)) > _GRADIENT_TOL * n:
        f_start = original.value(w)
        w_polished, f_polished = _lbfgs(original, w, f_start, tol)
        if f_polished < f_start - tol * (1.0 + abs(f_start)):
            v, _, converged = _newton(scaled, w_polished * scale, max_iter, tol)
            w = v / scale
            if original.value(w) > f_polished:
                w = w_polished
            g, _ = original.gradient(w)
            converged = converged or np.max(np.abs(g)) <= _GRADIENT_TOL * n
    if not converged:
        warnings.warn("Logistic fit stopped before converging (iteration limit reached or no "
                      "further decrease possible).", ConvergenceWarning, stacklevel=2)
    return w
