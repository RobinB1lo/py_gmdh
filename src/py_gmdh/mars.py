"""GMDH with MARS (multivariate adaptive regression spline) neurons.

Each neuron is a small MARS model (Friedman, 1991) of its two inputs: hinge
functions ``max(0, ±(x - t))`` and their pairwise products are added by a
greedy forward pass and pruned by generalized cross-validation (GCV).
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from ._base import GMDHClassifierBase, GMDHRegressorBase, Neuron

Factor = Tuple[int, float, int]
Basis = Tuple[Factor, ...]

_DEGENERACY_TOL = 1e-10


def _hinge(x, knot: float, sign: int, lib):
    if lib is np:
        return np.maximum(0.0, sign * (x - knot))
    return sp.Max(0, sign * (x - sp.Float(knot)), evaluate=False)


def _evaluate(basis: Basis, X: np.ndarray) -> np.ndarray:
    column = np.ones(X.shape[0])
    for variable, knot, sign in basis:
        column = column * _hinge(X[:, variable], knot, sign, np)
    return column


def _candidate_knots(x: np.ndarray, n_knots: int) -> np.ndarray:
    lo, hi = x.min(), x.max()
    knots = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_knots + 2)[1:-1]))
    return knots[(knots > lo) & (knots < hi)]


def _gcv(rss: float, n_samples: int, n_terms: int, penalty: float) -> float:
    """Friedman's GCV with ``n_terms + penalty * n_knots`` effective parameters."""
    effective = n_terms + penalty * (n_terms - 1) / 2.0
    if effective >= n_samples:
        return np.inf
    return rss / n_samples / (1.0 - effective / n_samples) ** 2


def _rss(B: np.ndarray, y: np.ndarray) -> float:
    w = np.linalg.lstsq(B, y, rcond=None)[0]
    return float(np.sum((y - B @ w) ** 2))


def _forward_pass(X, y, max_terms, max_degree, n_knots) -> Tuple[List[Basis], np.ndarray]:
    n, p = X.shape
    knots = [_candidate_knots(X[:, v], n_knots) for v in range(p)]
    basis: List[Basis] = [()]
    B = np.ones((n, 1))
    total = float(np.sum((y - y.mean()) ** 2)) + 1e-300

    while len(basis) < max_terms:
        Q, _ = np.linalg.qr(B)
        residual = y - Q @ (Q.T @ y)
        if residual @ residual <= 1e-12 * total:
            break

        best_gain, best_terms, best_columns = 0.0, None, None
        for m, parent in enumerate(basis):
            if len(parent) >= max_degree:
                continue
            used = {factor[0] for factor in parent}
            for v in range(p):
                if v in used or knots[v].size == 0:
                    continue
                x, T = X[:, v], knots[v]
                raw = [B[:, m, None] * np.maximum(0.0, x[:, None] - T),
                       B[:, m, None] * np.maximum(0.0, T - x[:, None])]
                perp = [H - Q @ (Q.T @ H) for H in raw]
                norms = [np.sum(H * H, axis=0) for H in perp]
                valid = [nrm > _DEGENERACY_TOL * (np.sum(H * H, axis=0) + 1e-300)
                         for nrm, H in zip(norms, raw)]
                proj = [H.T @ residual for H in perp]
                cross = np.sum(perp[0] * perp[1], axis=0)

                single = [np.where(ok, b ** 2 / np.where(ok, nrm, 1.0), 0.0)
                          for ok, b, nrm in zip(valid, proj, norms)]
                det = norms[0] * norms[1] - cross ** 2
                pair_ok = valid[0] & valid[1] & (det > _DEGENERACY_TOL * norms[0] * norms[1])
                pair = np.where(pair_ok,
                                (norms[1] * proj[0] ** 2 - 2 * cross * proj[0] * proj[1]
                                 + norms[0] * proj[1] ** 2) / np.where(pair_ok, det, 1.0),
                                0.0)

                for k in range(T.size):
                    options = []
                    if pair_ok[k] and len(basis) + 2 <= max_terms:
                        options.append((pair[k], (+1, -1)))
                    options += [(single[0][k], (+1,)), (single[1][k], (-1,))]
                    gain, signs = max(options, key=lambda o: o[0])
                    if gain > best_gain:
                        best_gain = gain
                        best_terms = [parent + ((v, float(T[k]), s),) for s in signs]
                        best_columns = [raw[0 if s > 0 else 1][:, k] for s in signs]

        if best_terms is None or best_gain <= 1e-12 * total:
            break
        basis.extend(best_terms)
        B = np.column_stack([B, *best_columns])
    return basis, B


def _backward_pass(B, y, basis, penalty) -> List[Basis]:
    n = len(y)
    active = list(range(len(basis)))
    best_active = list(active)
    best_gcv = _gcv(_rss(B, y), n, len(active), penalty)
    while len(active) > 1:
        rss, drop = min((_rss(B[:, [c for c in active if c != k]], y), k)
                        for k in active if k != 0)
        active.remove(drop)
        score = _gcv(rss, n, len(active), penalty)
        if score < best_gcv:
            best_gcv, best_active = score, list(active)
    return [basis[k] for k in best_active]


def fit_mars(X: np.ndarray, y: np.ndarray, max_terms: int = 11, max_degree: int = 2,
             penalty: float = 3.0, n_knots: int = 20) -> List[Basis]:
    """Select MARS basis functions for ``X`` and ``y``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    max_terms : int, default=11
        Maximum number of basis functions (including the intercept) in the
        forward pass.
    max_degree : int, default=2
        Maximum number of hinge factors in one basis function.
    penalty : float, default=3.0
        GCV cost per knot used by the backward pass.
    n_knots : int, default=20
        Number of candidate knots per variable, placed at quantiles.

    Returns
    -------
    basis : list of tuple
        Selected basis functions; each is a tuple of ``(variable, knot, sign)``
        hinge factors, and the empty tuple is the intercept.
    """
    basis, B = _forward_pass(X, y, max_terms, max_degree, n_knots)
    return _backward_pass(B, y, basis, penalty)


def evaluate_basis(basis: Sequence[Basis], X: np.ndarray) -> np.ndarray:
    """Design matrix of MARS basis functions evaluated on ``X``."""
    return np.column_stack([_evaluate(b, X) for b in basis])


class MARSNeuron(Neuron):
    """Neuron whose basis is selected by a two-input MARS fit.

    The hinge terms are always selected by least squares; the coefficients
    are then refitted with the neuron's link (ridge regression or logistic
    regression).
    """

    __slots__ = ("max_terms", "max_degree", "penalty", "n_knots", "basis")

    def __init__(self, i: int, j: int, max_terms: int = 11, max_degree: int = 2,
                 penalty: float = 3.0, n_knots: int = 20) -> None:
        super().__init__(i, j)
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.n_knots = n_knots
        self.basis: List[Basis] = [()]

    def terms(self, a, b, lib) -> list:
        inputs = (a, b)
        columns = []
        for basis in self.basis:
            term = 1
            for variable, knot, sign in basis:
                term = term * _hinge(inputs[variable], knot, sign, lib)
            columns.append(term)
        return columns

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray, ridge: float) -> "MARSNeuron":
        self.basis = fit_mars(np.column_stack([a, b]), y, self.max_terms, self.max_degree,
                              self.penalty, self.n_knots)
        self.w = self._solve(self.design_matrix(a, b), y, ridge)
        return self


class _MARSBasis:
    _expandable = False

    def _check_params(self) -> None:
        super()._check_params()
        if not isinstance(self.max_terms, (int, np.integer)) or self.max_terms < 2:
            raise ValueError(f"max_terms must be an integer >= 2, got {self.max_terms!r}.")
        if self.max_degree not in (1, 2):
            raise ValueError(f"max_degree must be 1 or 2, got {self.max_degree!r}.")
        if self.penalty < 0:
            raise ValueError(f"penalty must be non-negative, got {self.penalty!r}.")
        if not isinstance(self.n_knots, (int, np.integer)) or self.n_knots < 1:
            raise ValueError(f"n_knots must be a positive integer, got {self.n_knots!r}.")

    def _make_neuron(self, i: int, j: int) -> MARSNeuron:
        return MARSNeuron(i, j, self.max_terms, self.max_degree, self.penalty, self.n_knots)


class MARSGMDH(_MARSBasis, GMDHRegressorBase):
    """GMDH regressor with MARS neurons.

    Each neuron fits a small MARS model to its two inputs: a forward pass
    greedily adds hinge functions ``max(0, ±(z - t))``, optionally multiplied
    by an existing term, with knots ``t`` chosen among quantiles of the
    training inputs; a backward pass then prunes terms by GCV. Candidates are
    ranked by their RMSE on the selection set, as in :class:`GMDH`.

    Parameters
    ----------
    n_keep : int, default=10
        Number of neurons kept in each layer and passed on to the next.
    max_layers : int, default=10
        Maximum number of layers to grow.
    ridge : float, default=1e-6
        L2 regularization used when refitting the selected basis of each
        neuron.
    training_split : float, default=0.5
        Fraction of samples used to fit neurons; the remainder forms the
        selection set used to rank them.
    patience : int, default=0
        Number of consecutive non-improving layers tolerated before growth
        stops.
    threshold : float, optional
        Stop as soon as the best selection RMSE (on the standardized target)
        falls to or below this value.
    max_terms : int, default=11
        Maximum number of basis functions per neuron, including the
        intercept, before pruning.
    max_degree : {1, 2}, default=2
        Maximum interaction order inside a neuron; ``1`` gives additive
        neurons.
    penalty : float, default=3.0
        GCV cost per knot used when pruning.
    n_knots : int, default=20
        Number of candidate knots per input, placed at quantiles.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer. The
        selected hinge terms of each neuron are available as ``basis``.
    best_score_ : float
        Selection RMSE of the output neuron on the standardized target.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during :meth:`fit`, when available.
    x_mean_, x_scale_ : ndarray of shape (n_features_in_,)
        Per-feature standardization statistics.
    y_mean_, y_scale_ : float
        Target standardization statistics.
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None, max_terms: int = 11,
                 max_degree: int = 2, penalty: float = 3.0, n_knots: int = 20,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.n_knots = n_knots
        self.random_state = random_state


class MARSGMDHClassifier(_MARSBasis, GMDHClassifierBase):
    """GMDH binary classifier with MARS neurons.

    Each neuron selects hinge functions of its two inputs with a least-squares
    MARS forward pass and GCV pruning on the 0/1 target, then refits their
    coefficients by logistic regression. Candidates are ranked by selection
    log loss.

    Parameters
    ----------
    n_keep : int, default=10
        Number of neurons kept in each layer and passed on to the next.
    max_layers : int, default=10
        Maximum number of layers to grow.
    ridge : float, default=1e-6
        L2 penalty of each logistic neuron (the inverse of scikit-learn's
        ``C``).
    training_split : float, default=0.5
        Fraction of samples used to fit neurons; the remainder forms the
        selection set used to rank them.
    patience : int, default=0
        Number of consecutive non-improving layers tolerated before growth
        stops.
    threshold : float, optional
        Stop as soon as the best selection log loss falls to or below this
        value.
    max_terms : int, default=11
        Maximum number of basis functions per neuron, including the
        intercept, before pruning.
    max_degree : {1, 2}, default=2
        Maximum interaction order inside a neuron; ``1`` gives additive
        neurons.
    penalty : float, default=3.0
        GCV cost per knot used when pruning.
    n_knots : int, default=20
        Number of candidate knots per input, placed at quantiles.
    calibrate : bool, default=True
        Recalibrate the output probability with Platt scaling fitted on
        out-of-fold predictions. This refits the network ``calibration_cv``
        additional times.
    calibration_cv : int, default=3
        Number of stratified folds used to fit the calibration.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer. The
        selected hinge terms of each neuron are available as ``basis``.
    best_score_ : float
        Selection log loss of the output neuron.
    classes_ : ndarray of shape (2,)
        Class labels; ``classes_[1]`` is the positive class.
    calibration_slope_, calibration_intercept_ : float
        Platt scaling parameters ``a`` and ``b`` of ``sigmoid(a * z + b)``,
        where ``z`` is the output neuron's logit; ``1`` and ``0`` when
        ``calibrate=False``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during :meth:`fit`, when available.
    x_mean_, x_scale_ : ndarray of shape (n_features_in_,)
        Per-feature standardization statistics.
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None,
                 max_terms: int = 11,
                 max_degree: int = 2,
                 penalty: float = 3.0,
                 n_knots: int = 20,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.n_knots = n_knots
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
