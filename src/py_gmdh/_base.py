"""Shared building blocks for the GMDH estimators.

A GMDH network is built layer by layer. Every layer fits one two-input
partial model (a *neuron*) for each pair of its inputs on a training split,
ranks the candidates by an external criterion computed on a separate
selection split, and passes the outputs of the best ``n_keep`` neurons on as
the inputs of the next layer. Growth stops once the criterion no longer
improves, and the network is truncated to its best layer.

Variants differ only in the neuron they fit and the criterion they rank
with, so both are exposed as overridable hooks on :class:`BaseGMDH`.
Regressors use neurons with an identity output; classifiers wrap the same
neurons in a logistic link and fit them by penalized maximum likelihood.
"""

from __future__ import annotations

import re
import warnings
from itertools import combinations
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._lazy import sp
from ._logistic import ConvergenceWarning, logistic_fit, sigmoid

_SCALE_EPS = 1e-12
_RANGE_EPS = 1e-10
_PROBA_EPS = 1e-8
_LOGIT_CLIP = 30.0
_AUTO_EXPAND_MAX_DEPTH = 3
_DEFAULT_MAX_LENGTH = 2000
_UNSTABLE_EXPANSION = 1e-3

IDENTITY = "identity"
LOGISTIC = "logistic"


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _ridge_solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = X.T @ X + ridge * np.eye(X.shape[1])
    try:
        return np.linalg.solve(gram, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, X.T @ y, rcond=None)[0]


def _logistic_solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """L2-penalized logistic regression without a separate intercept.

    The design matrix already contains a constant column, which is penalized
    like every other coefficient.
    """
    return logistic_fit(X, y, ridge)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return value if np.isfinite(value) else np.inf


def _log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    p = np.clip(proba, _PROBA_EPS, 1 - _PROBA_EPS)
    value = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    return value if np.isfinite(value) else np.inf


def _fit_platt(scores: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    """Fit Platt scaling ``P = sigmoid(a * score + b)`` by maximum likelihood.

    Uses Platt's (1999) smoothed targets ``(N+ + 1) / (N+ + 2)`` and
    ``1 / (N- + 2)`` to avoid overfitting the two parameters.
    """
    n_pos = float(t.sum())
    n_neg = float(len(t) - n_pos)
    target = np.where(t == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    start = np.array([1.0, np.log((n_pos + 1) / (n_neg + 1))])
    a, b = logistic_fit(np.column_stack([scores, np.ones_like(scores)]), target, ridge=0.0,
                        start=start)
    return float(a), float(b)


def _unit_scale(x, lo: float, hi: float):
    """Map ``x`` from ``[lo, hi]`` onto ``[0, 1]`` (NumPy or SymPy input)."""
    return (x - lo) / (hi - lo + _RANGE_EPS)


class EquationTooLongError(ValueError):
    """Raised by ``equation()`` when the closed form would be too long to be useful."""


def _check_precision(precision) -> None:
    if precision is not None and (isinstance(precision, bool)
                                  or not isinstance(precision, (int, np.integer))
                                  or precision < 1):
        raise ValueError(f"precision must be a positive integer or None, got {precision!r}.")


def _number(value, digits: Optional[int] = None) -> sp.Float:
    """SymPy constant for a fitted coefficient, rounded to ``digits`` significant digits.

    Only fitted coefficients are rounded for display; constants derived from
    the data (means, scales, ranges, knots) are always kept at full
    precision, because small relative errors in them can change the output
    by a large amount.
    """
    value = float(value)
    if digits is not None:
        value = float(f"{value:.{digits}g}")
    return sp.Float(value)


def _round_coefficients(expr: sp.Expr, digits: Optional[int],
                        keep_constant: bool = True) -> sp.Expr:
    """Round the coefficient of every term of an expanded sum.

    With ``keep_constant`` the constant term is left at full precision; for
    regressors it absorbs the target mean, which may be large compared with
    the variation of the output.
    """
    if digits is None:
        return expr
    terms = []
    for term in sp.Add.make_args(expr):
        if term.is_Number:
            terms.append(term if keep_constant else _number(term, digits))
        else:
            coefficient, rest = term.as_coeff_Mul()
            terms.append(_number(coefficient, digits) * rest)
    return sp.Add(*terms)


def _sigmoid_expression(z: sp.Expr) -> sp.Expr:
    """``1 / (1 + exp(-z))`` kept in this form rather than simplified by SymPy."""
    with sp.evaluate(False):
        return 1 / (1 + sp.exp(-z))


def _expand_sigmoid_arguments(expr: sp.Expr, digits: Optional[int] = None) -> sp.Expr:
    """Multiply out the innermost sigmoid arguments, which are functions of the inputs only."""
    replacements = {node: sp.exp(_round_coefficients(sp.expand(node.args[0]), digits,
                                                     keep_constant=False), evaluate=False)
                    for node in expr.atoms(sp.exp) if not node.args[0].has(sp.exp)}
    with sp.evaluate(False):
        return expr.xreplace(replacements)


def _scaled_term(coefficient: sp.Float, term) -> sp.Expr:
    """``coefficient * term`` without distributing the coefficient over a sum.

    Distributing would multiply rounded coefficients together and print more
    digits than requested; the result is mathematically the same.
    """
    if term == 1:
        return coefficient
    return sp.Mul(coefficient, term, evaluate=False)


def _definition(z: str, name: str, mean: float, scale: float) -> str:
    sign = "-" if mean >= 0 else "+"
    return f"  {z} = ({name} {sign} {abs(mean)!r}) / {scale!r}"


# ---------------------------------------------------------------------------
# Neurons
# ---------------------------------------------------------------------------

class Neuron:
    """Two-input partial model built on a basis ``phi(a, b)``.

    With the identity link the neuron predicts ``w · phi(a, b)`` and is fitted
    by ridge regression. With the logistic link it predicts
    ``sigmoid(w · phi(a, b))`` and is fitted by L2-penalized logistic
    regression.

    Subclasses define the basis through :meth:`terms`, which must work for
    both NumPy arrays (``lib=numpy``) and SymPy expressions (``lib=sympy``).
    Using a single definition for both guarantees that the symbolic equation
    reproduces the numerical predictions exactly.

    Parameters
    ----------
    i, j : int
        Indices of the two input columns this neuron combines.
    """

    __slots__ = ("i", "j", "w", "link")

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.w: Optional[np.ndarray] = None
        self.link = IDENTITY

    def terms(self, a, b, lib) -> list:
        raise NotImplementedError

    def _setup(self, a: np.ndarray, b: np.ndarray) -> None:
        """Hook for data-dependent basis parameters, called before fitting."""

    def design_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        columns = self.terms(a, b, np)
        return np.column_stack([np.broadcast_to(np.asarray(c, dtype=float), a.shape)
                                for c in columns])

    def _solve(self, X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
        if self.link == LOGISTIC:
            return _logistic_solve(X, y, ridge)
        return _ridge_solve(X, y, ridge)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.link == LOGISTIC:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -_LOGIT_CLIP, _LOGIT_CLIP)))
        return z

    def loss(self, y: np.ndarray, output: np.ndarray) -> float:
        """RMSE for the identity link, mean log loss for the logistic link."""
        return _log_loss(y, output) if self.link == LOGISTIC else _rmse(y, output)

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray, ridge: float) -> "Neuron":
        self._setup(a, b)
        self.w = self._solve(self.design_matrix(a, b), y, ridge)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._activate(self.linear_predict(a, b))

    def linear_predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Output before the link function (the logit for logistic neurons)."""
        return self.design_matrix(a, b) @ self.w

    def _linear_expression(self, a, b, digits: Optional[int] = None, scale: float = 1.0,
                           offset: float = 0.0) -> sp.Expr:
        """``scale * (w · phi(a, b)) + offset``, coefficients rounded to ``digits``.

        The first basis function is the constant, so ``offset`` (kept at full
        precision) is added to the intercept.
        """
        expr = sp.Add(*[_scaled_term(_number(scale * w, digits), t)
                        for w, t in zip(self.w, self.terms(a, b, sp))])
        return expr + sp.Float(float(offset)) if offset else expr

    def expression(self, a: sp.Expr, b: sp.Expr, digits: Optional[int] = None) -> sp.Expr:
        """Symbolic form of the fitted neuron with inputs ``a`` and ``b``."""
        linear = self._linear_expression(a, b, digits)
        if self.link == LOGISTIC:
            return _sigmoid_expression(linear)
        return linear

    def _input_multiplicity(self) -> Tuple[int, int]:
        """Number of times each input occurs in the neuron's symbolic expression."""
        a, b = sp.Dummy("a"), sp.Dummy("b")
        nodes = list(sp.preorder_traversal(self.expression(a, b)))
        return sum(node == a for node in nodes), sum(node == b for node in nodes)

    def _describe_linear(self, a_name: str, b_name: str, precision: Optional[int]) -> str:
        return sp.sstr(self._linear_expression(sp.Symbol(a_name), sp.Symbol(b_name), precision))

    def describe(self, a_name: str, b_name: str, precision: Optional[int] = 4) -> str:
        """Human-readable formula of the neuron in terms of named inputs."""
        formula = self._describe_linear(a_name, b_name, precision)
        return f"sigmoid({formula})" if self.link == LOGISTIC else formula


class PolynomialNeuron(Neuron):
    """Ivakhnenko polynomial ``w0 + w1 a + w2 b + w3 ab + w4 a² + w5 b²``."""

    __slots__ = ()

    def terms(self, a, b, lib) -> list:
        return [1, a, b, a * b, a ** 2, b ** 2]


class TensorBasisNeuron(Neuron):
    """Neuron built from a univariate basis applied to each input.

    The design matrix is ``[1, phi(a), phi(b), phi(a) ⊗ phi(b)]``, where each
    input is first rescaled to ``[0, 1]`` using the range seen during fitting.
    Subclasses implement :meth:`basis`.
    """

    __slots__ = ("a_range", "b_range")

    def __init__(self, i: int, j: int) -> None:
        super().__init__(i, j)
        self.a_range = (0.0, 1.0)
        self.b_range = (0.0, 1.0)

    def basis(self, u, lib) -> list:
        """Evaluate the univariate basis on ``u`` in ``[0, 1]``."""
        raise NotImplementedError

    def _setup(self, a: np.ndarray, b: np.ndarray) -> None:
        self.a_range = (float(a.min()), float(a.max()))
        self.b_range = (float(b.min()), float(b.max()))

    def terms(self, a, b, lib) -> list:
        phi_a = self.basis(_unit_scale(a, *self.a_range), lib)
        phi_b = self.basis(_unit_scale(b, *self.b_range), lib)
        return [1, *phi_a, *phi_b, *[p * q for p in phi_a for q in phi_b]]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Layer:
    """Surviving neurons of one GMDH layer, ordered best first.

    Attributes
    ----------
    neurons : list of Neuron
        Selected neurons; the first one is the layer's best model.
    metrics : dict
        Selection metrics of the best neuron.
    """

    __slots__ = ("neurons", "metrics")

    def __init__(self, neurons: List[Neuron], metrics: Dict[str, float]) -> None:
        self.neurons = neurons
        self.metrics = metrics

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.column_stack([n.predict(Z[:, n.i], Z[:, n.j]) for n in self.neurons])

    def __repr__(self) -> str:
        return f"Layer(n_neurons={len(self.neurons)}, metrics={self.metrics})"


class BaseGMDH(BaseEstimator):
    """Base class implementing the layered GMDH search.

    Subclasses may override:

    * :meth:`_make_neuron` - the partial model fitted for each input pair;
    * :meth:`_score` - the external criterion used to rank candidates;
    * :meth:`_layer_context` - per-layer state shared by all candidates;
    * :meth:`_check_params` - validation of variant-specific parameters.

    Class attributes:

    * ``_link`` - output link of every neuron (identity or logistic);
    * ``_base_metric`` - name of the default selection metric;
    * ``_criterion`` - metric that drives early stopping and ``threshold``;
    * ``_improvement_tol`` - minimal decrease of the criterion that counts
      as an improvement;
    * ``_expandable`` - whether the symbolic equation is polynomial and
      therefore worth expanding by default.
    """

    _link = IDENTITY
    _base_metric = "rmse"
    _improvement_tol = 1e-12
    _expandable = True
    _fitted_attr = "layers_"

    @property
    def _criterion(self) -> str:
        return self._base_metric

    # -- hooks --------------------------------------------------------------

    def _make_neuron(self, i: int, j: int) -> Neuron:
        return PolynomialNeuron(i, j)

    def _create_neuron(self, i: int, j: int) -> Neuron:
        neuron = self._make_neuron(i, j)
        neuron.link = self._link
        return neuron

    def _layer_context(self, n_selection: int, rng: np.random.Generator) -> Any:
        return None

    def _score(self, neuron: Neuron, a_tr, b_tr, y_tr, a_se, b_se, y_se,
               context: Any) -> Optional[Tuple[Any, Dict[str, float]]]:
        """Return ``(sort_key, metrics)`` for a fitted candidate (lower key is better).

        Returning ``None`` discards the candidate.
        """
        value = neuron.loss(y_se, neuron.predict(a_se, b_se))
        return value, {self._base_metric: value}

    def _check_params(self) -> None:
        self._check_search_params()
        if not 0.0 < self.training_split < 1.0:
            raise ValueError(f"training_split must lie in (0, 1), got {self.training_split!r}.")
        if self.ridge < 0:
            raise ValueError(f"ridge must be non-negative, got {self.ridge!r}.")

    def _check_search_params(self) -> None:
        if not isinstance(self.n_keep, (int, np.integer)) or self.n_keep < 1:
            raise ValueError(f"n_keep must be a positive integer, got {self.n_keep!r}.")
        if not isinstance(self.max_layers, (int, np.integer)) or self.max_layers < 1:
            raise ValueError(f"max_layers must be a positive integer, got {self.max_layers!r}.")
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience!r}.")

    # -- data handling ------------------------------------------------------

    def _validate_fit_data(self, X, y, y_numeric: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        self._check_params()
        columns = getattr(X, "columns", None)
        X, y = check_X_y(X, y, y_numeric=y_numeric)
        if X.shape[1] < 2:
            raise ValueError(f"GMDH requires at least 2 features, got {X.shape[1]} feature(s).")
        self.n_features_in_ = X.shape[1]
        if columns is not None and all(isinstance(c, str) for c in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return X, y

    def _validate_predict_data(self, X) -> np.ndarray:
        check_is_fitted(self, self._fitted_attr)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but {type(self).__name__} "
                             f"is expecting {self.n_features_in_} features as input.")
        return X

    def _fit_input_scaling(self, X: np.ndarray) -> np.ndarray:
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0) + _SCALE_EPS
        return self._scale_inputs(X)

    def _scale_inputs(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean_) / self.x_scale_

    def _split(self, n_samples: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        order = rng.permutation(n_samples)
        n_train = int(n_samples * self.training_split)
        if n_train < 1 or n_train >= n_samples:
            raise ValueError(f"training_split={self.training_split} leaves an empty training "
                             f"or selection set for {n_samples} samples.")
        return order[:n_train], order[n_train:]

    def _feature_symbols(self, feature_names: Optional[Sequence[str]]) -> List[sp.Symbol]:
        if feature_names is None:
            if hasattr(self, "feature_names_in_"):
                feature_names = list(self.feature_names_in_)
            else:
                feature_names = [f"x{k}" for k in range(self.n_features_in_)]
        feature_names = [str(name) for name in feature_names]
        if len(feature_names) != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} feature names, "
                             f"got {len(feature_names)}.")
        return [sp.Symbol(name) for name in feature_names]

    def _scaled_input_expressions(self, symbols) -> List[sp.Expr]:
        return [(s - float(m)) / float(d)
                for s, m, d in zip(symbols, self.x_mean_, self.x_scale_)]

    @staticmethod
    def _standardized_symbols(symbols) -> List[sp.Symbol]:
        """``z0, z1, ...`` for default names, otherwise ``z_<name>`` (made identifier-safe)."""
        names = [s.name for s in symbols]
        if names == [f"x{k}" for k in range(len(names))]:
            return [sp.Symbol(f"z{k}") for k in range(len(names))]
        taken = set(names)
        result = []
        for name in names:
            base = "z_" + (re.sub(r"\W", "_", name) or "_")
            candidate, k = base, 1
            while candidate in taken:
                candidate, k = f"{base}_{k}", k + 1
            taken.add(candidate)
            result.append(sp.Symbol(candidate))
        return result

    # -- network search -----------------------------------------------------

    def _fit_layer(self, Z_tr, Z_se, y_tr, y_se, rng):
        context = self._layer_context(len(y_se), rng)
        candidates = []
        for i, j in combinations(range(Z_tr.shape[1]), 2):
            neuron = self._create_neuron(i, j).fit(Z_tr[:, i], Z_tr[:, j], y_tr, self.ridge)
            scored = self._score(neuron, Z_tr[:, i], Z_tr[:, j], y_tr,
                                 Z_se[:, i], Z_se[:, j], y_se, context)
            if scored is not None:
                candidates.append((scored[0], scored[1], neuron))
        if not candidates:
            raise RuntimeError("No candidate neuron could be scored in this layer.")

        candidates.sort(key=lambda c: c[0])
        survivors = candidates[:self.n_keep]
        layer = Layer([c[2] for c in survivors], survivors[0][1])
        return layer, layer.predict(Z_tr), layer.predict(Z_se)

    def _fit_network(self, Z_tr, Z_se, y_tr, y_se, rng) -> None:
        layers: List[Layer] = []
        best_score = np.inf
        best_depth = -1
        stalled = 0

        for depth in range(self.max_layers):
            if Z_tr.shape[1] < 2:
                break
            layer, Z_tr, Z_se = self._fit_layer(Z_tr, Z_se, y_tr, y_se, rng)
            layers.append(layer)

            score = layer.metrics[self._criterion]
            if score < best_score - self._improvement_tol:
                best_score, best_depth, stalled = score, depth, 0
            else:
                stalled += 1

            if self.threshold is not None and best_score <= self.threshold:
                break
            if stalled > self.patience:
                break

        if best_depth < 0:
            raise RuntimeError("No neuron produced a finite selection score.")
        self.layers_ = layers[:best_depth + 1]
        self.best_score_ = best_score

    def _forward(self, Z: np.ndarray, linear: bool = False) -> np.ndarray:
        """Network output; with ``linear=True`` the output neuron's value before its link."""
        for layer in self.layers_[:-1]:
            Z = layer.predict(Z)
        neuron = self.layers_[-1].neurons[0]
        a, b = Z[:, neuron.i], Z[:, neuron.j]
        return neuron.linear_predict(a, b) if linear else neuron.predict(a, b)

    def _output_expression(self, inputs: List[sp.Expr], linear: bool = False,
                           digits: Optional[int] = None, scale: float = 1.0,
                           offset: float = 0.0) -> sp.Expr:
        """Symbolic network output; ``scale`` and ``offset`` apply to a linear output."""
        memo: Dict[Tuple[int, int], sp.Expr] = {}

        def arguments(depth: int, neuron: Neuron):
            if depth == 0:
                return inputs[neuron.i], inputs[neuron.j]
            return node(depth - 1, neuron.i), node(depth - 1, neuron.j)

        def node(depth: int, index: int) -> sp.Expr:
            key = (depth, index)
            if key not in memo:
                neuron = self.layers_[depth].neurons[index]
                memo[key] = neuron.expression(*arguments(depth, neuron), digits)
            return memo[key]

        depth = len(self.layers_) - 1
        output = self.layers_[depth].neurons[0]
        if linear or self._link == IDENTITY:
            return output._linear_expression(*arguments(depth, output), digits, scale, offset)
        return node(depth, 0)

    def _depth(self) -> int:
        return len(self.layers_)

    def _polynomial_degree(self) -> int:
        """Upper bound on the degree of the expanded equation of a polynomial network."""
        return 2 ** self._depth()

    def _equation_size(self) -> Tuple[int, set]:
        """Variable occurrences in the nested equation and the features it uses.

        Each neuron repeats an input as often as the input appears in its own
        formula (three times for the quadratic neuron), so the count can grow
        exponentially with depth. It is computed from the network structure
        without building the equation.
        """
        memo: Dict[Tuple[int, int], Tuple[int, set]] = {}

        def visit(depth: int, index: int) -> Tuple[int, set]:
            key = (depth, index)
            if key not in memo:
                neuron = self.layers_[depth].neurons[index]
                count_a, count_b = neuron._input_multiplicity()
                if depth == 0:
                    features = {k for k, c in ((neuron.i, count_a), (neuron.j, count_b)) if c}
                    memo[key] = (count_a + count_b, features)
                else:
                    size_a, feat_a = visit(depth - 1, neuron.i) if count_a else (0, set())
                    size_b, feat_b = visit(depth - 1, neuron.j) if count_b else (0, set())
                    memo[key] = (count_a * size_a + count_b * size_b, feat_a | feat_b)
            return memo[key]

        return visit(len(self.layers_) - 1, 0)

    def _summary_body(self, names: List[str], precision: Optional[int]) -> List[str]:
        lines = []
        inputs = names
        for depth, layer in enumerate(self.layers_):
            metrics = ", ".join(f"{k}={v:.6g}" for k, v in layer.metrics.items())
            lines.append(f"Layer {depth + 1} ({len(layer.neurons)} neurons; best: {metrics})")
            outputs = []
            for index, neuron in enumerate(layer.neurons):
                name = f"n{depth + 1}_{index}"
                formula = neuron.describe(inputs[neuron.i], inputs[neuron.j], precision)
                lines.append(f"  {name} = {formula}")
                outputs.append(name)
            inputs = outputs
        lines.append(f"Output: {inputs[0]}")
        return lines

    # -- shared public helpers ------------------------------------------------

    def _build_expression(self, inputs, digits, expand, scale=1.0, offset=0.0,
                          calibration=None) -> sp.Expr:
        if calibration is None:
            expr = self._output_expression(inputs, digits=digits, scale=scale, offset=offset)
        else:
            slope, intercept = calibration
            logit = self._output_expression(inputs, linear=True, digits=digits)
            with sp.evaluate(False):
                clipped = sp.Min(sp.Max(logit, -_LOGIT_CLIP), _LOGIT_CLIP)
                score = _number(slope, digits) * clipped + _number(intercept, digits)
            expr = _sigmoid_expression(score)
        if expand:
            if self._link == LOGISTIC:
                expr = _expand_sigmoid_arguments(expr, digits if self._expandable else None)
            else:
                expr = sp.expand(expr)
                if self._expandable:
                    expr = _round_coefficients(expr, digits)
        return expr

    def _check_equation_size(self, expand: bool, max_length: Optional[int]) -> set:
        size, used = self._equation_size()
        what = "variable occurrences"
        if expand and self._expandable and self._link == IDENTITY:
            degree = self._polynomial_degree()
            terms = comb(len(used) + degree, degree)
            if terms > size:
                size, what = terms, "terms (upper bound for the expanded polynomial)"
        if max_length is not None and size > max_length:
            raise EquationTooLongError(
                f"The equation is too long to be useful: it would contain about {size:,} "
                f"{what}, more than max_length={max_length:,}. Reduce the number of "
                "parameters of the model, for example by fitting with a smaller max_layers or "
                "n_keep, or with fewer input features (keeping the most significant ones), and "
                "refit. summary() describes the current model layer by layer. Pass "
                "max_length=None to build the equation anyway.")
        return used

    def _warn_if_unstable(self, used: set, digits: Optional[int], degree: int) -> None:
        """Warn when a raw-variable expansion is numerically ill-conditioned.

        Expanding a polynomial of degree ``d`` in ``x = mean + scale * z``
        produces coefficients of size about ``(1 + |mean| / scale) ** d`` that
        cancel when the equation is evaluated, so relative errors of
        ``10**-digits`` in the printed coefficients are amplified by that factor.
        """
        if not used:
            return
        ratio = max(abs(float(self.x_mean_[k])) / float(self.x_scale_[k]) for k in used)
        amplification = (1.0 + ratio) ** degree
        if amplification * 10.0 ** -(digits if digits is not None else 15) > _UNSTABLE_EXPANSION:
            warnings.warn(
                "The expanded equation in raw variables is numerically ill-conditioned: some "
                f"features have |mean| / scale up to {ratio:.3g}, so its coefficients nearly "
                "cancel and small rounding errors are amplified about "
                f"{amplification:.2g}-fold. Use variables='standardized' (the default) or "
                "expand=False for an accurate equation.", RuntimeWarning, stacklevel=3)

    def _equation(self, lhs: str, feature_names, precision, expand, as_sympy, variables,
                  max_length, scale=1.0, offset=0.0, calibration=None):
        check_is_fitted(self, self._fitted_attr)
        _check_precision(precision)
        if variables not in ("standardized", "raw"):
            raise ValueError(f"variables must be 'standardized' or 'raw', got {variables!r}.")
        if max_length is not None and max_length < 1:
            raise ValueError(f"max_length must be a positive integer or None, got {max_length!r}.")
        symbols = self._feature_symbols(feature_names)
        if expand is None:
            expand = self._expandable and self._depth() <= _AUTO_EXPAND_MAX_DEPTH
        used = self._check_equation_size(expand, max_length)

        standardized = variables == "standardized"
        z_symbols = self._standardized_symbols(symbols) if standardized else None
        inputs = z_symbols if standardized else self._scaled_input_expressions(symbols)
        digits = None if as_sympy else precision
        expr = self._build_expression(inputs, digits, expand, scale, offset, calibration)
        if not standardized and expand and self._expandable:
            degree = self._polynomial_degree() if self._link == IDENTITY else 2
            self._warn_if_unstable(used, digits, degree)

        present = expr.free_symbols
        indices = [k for k in range(len(symbols)) if standardized and z_symbols[k] in present]
        if as_sympy:
            if not standardized:
                return expr
            definitions = {z_symbols[k]: (symbols[k] - float(self.x_mean_[k]))
                           / float(self.x_scale_[k]) for k in indices}
            return expr, definitions
        text = f"{lhs} = {sp.sstr(expr)}"
        if indices:
            text += "\nwhere\n" + "\n".join(
                _definition(z_symbols[k].name, symbols[k].name, float(self.x_mean_[k]),
                            float(self.x_scale_[k])) for k in indices)
        return text

    def _summary_text(self, feature_names, precision, footer: str) -> str:
        check_is_fitted(self, self._fitted_attr)
        _check_precision(precision)
        symbols = self._feature_symbols(feature_names)
        z_symbols = self._standardized_symbols(symbols)
        header = [f"{type(self).__name__} (selection criterion: {self._criterion}, "
                  f"best score: {self.best_score_:.6g})",
                  "Inputs are standardized:"]
        header += [_definition(z.name, x.name, float(m), float(d))
                   for z, x, m, d in zip(z_symbols, symbols, self.x_mean_, self.x_scale_)]
        body = self._summary_body([z.name for z in z_symbols], precision)
        return "\n".join(header + [""] + body + ["", footer])


class GMDHRegressorBase(RegressorMixin, BaseGMDH):
    """Regression interface shared by all GMDH regressors.

    Inputs and target are standardized before the network is grown; the
    scaling is undone in :meth:`predict` and folded into :meth:`equation`.
    """

    def fit(self, X, y) -> "GMDHRegressorBase":
        """Grow the GMDH network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs. At least two features are required.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_fit_data(X, y)
        Z = self._fit_input_scaling(X)
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std() + _SCALE_EPS)
        t = (y - self.y_mean_) / self.y_scale_

        rng = np.random.default_rng(self.random_state)
        train, select = self._split(len(y), rng)
        self._fit_network(Z[train], Z[select], t[train], t[select], rng)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        Z = self._scale_inputs(self._validate_predict_data(X))
        return self._forward(Z) * self.y_scale_ + self.y_mean_

    def equation(self, feature_names: Optional[Sequence[str]] = None,
                 precision: Optional[int] = 4, expand: Optional[bool] = None,
                 as_sympy: bool = False, variables: str = "standardized",
                 max_length: Optional[int] = _DEFAULT_MAX_LENGTH):
        """Closed-form equation of the fitted network.

        All neurons are substituted into a single equation. By default it is
        written in the standardized inputs ``z_k = (x_k - mean_k) / scale_k``,
        whose definitions are listed below the equation; this keeps the
        coefficients well conditioned when a feature's mean is large compared
        with its spread.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names used for the input variables. Defaults to the column names
            seen during :meth:`fit`, or ``x0, x1, ...``.
        precision : int or None, default=4
            Significant digits shown for the fitted coefficients in the
            returned string. Rounding is for display only: constants derived
            from the data (means, scales, ranges, knots) are always shown in
            full, and ``as_sympy=True`` always returns full precision.
            ``None`` shows every coefficient in full.
        expand : bool or None, default=None
            Whether to multiply the nested expression out. By default the
            equation is expanded only for polynomial networks with at most
            three layers; deeper networks grow combinatorially when expanded.
        as_sympy : bool, default=False
            Return SymPy objects instead of a string: the full-precision
            right-hand side and, with standardized variables, a dict mapping
            each ``z_k`` symbol to its definition in the original feature
            (``expr.subs(definitions)`` gives the equation in raw variables).
        variables : {"standardized", "raw"}, default="standardized"
            Write the equation in the standardized inputs ``z_k`` or directly
            in the original features. An expanded equation in raw variables
            can be numerically ill-conditioned for features whose mean is
            large relative to their scale; a ``RuntimeWarning`` is issued
            when that is likely.
        max_length : int or None, default=2000
            Largest equation that is built, measured in variable occurrences
            (or terms, for an expanded polynomial). Larger equations raise
            :class:`EquationTooLongError`; ``None`` removes the limit.

        Returns
        -------
        equation : str, sympy.Expr or (sympy.Expr, dict)
            ``"y = ..."`` followed by the ``z_k`` definitions, or the SymPy
            objects described under ``as_sympy``.

        Raises
        ------
        EquationTooLongError
            If the equation would exceed ``max_length``.
        """
        return self._equation("y", feature_names, precision, expand, as_sympy, variables,
                              max_length, scale=self.y_scale_, offset=self.y_mean_)

    def summary(self, feature_names: Optional[Sequence[str]] = None,
                precision: Optional[int] = 4) -> str:
        """Layer-by-layer description of the fitted network.

        Each neuron is written in terms of the *standardized* outputs of the
        previous layer. Use :meth:`equation` for the fully substituted model.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names used for the input variables.
        precision : int or None, default=4
            Number of significant digits shown for coefficients.

        Returns
        -------
        summary : str
        """
        check_is_fitted(self, self._fitted_attr)
        sign = "-" if self.y_mean_ < 0 else "+"
        footer = f"y = output * {self.y_scale_:.6g} {sign} {abs(self.y_mean_):.6g}"
        return self._summary_text(feature_names, precision, footer)


class GMDHClassifierBase(ClassifierMixin, BaseGMDH):
    """Binary classification interface shared by all GMDH classifiers.

    Every neuron is a logistic model ``sigmoid(w · phi(a, b))`` fitted by
    L2-penalized maximum likelihood, and candidates are ranked by their log
    loss on the selection set. Neurons of later layers take the predicted
    probabilities of the previous layer as inputs, and the probability of the
    network's output neuron is the predicted probability of the positive
    class, ``classes_[1]``.

    With ``calibrate=True`` the output probability ``sigmoid(z)`` of the
    network is recalibrated by Platt scaling to ``sigmoid(a * z + b)``, where
    ``z`` is the output neuron's logit clipped to ``[-30, 30]`` as in every
    neuron. The two parameters are fitted on
    out-of-fold predictions: the network is refitted ``calibration_cv`` times
    on stratified folds of the data and predicts the held-out fold, so
    calibration never reuses data a network was built on. The final network
    is the one fitted on all data.
    """

    _link = LOGISTIC
    _base_metric = "log_loss"
    _expandable = False

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _more_tags(self):
        return {"binary_only": True}

    def _check_params(self) -> None:
        super()._check_params()
        if not isinstance(self.calibrate, (bool, np.bool_)):
            raise ValueError(f"calibrate must be a boolean, got {self.calibrate!r}.")
        if not isinstance(self.calibration_cv, (int, np.integer)) or self.calibration_cv < 2:
            raise ValueError("calibration_cv must be an integer >= 2, "
                             f"got {self.calibration_cv!r}.")

    def fit(self, X, y) -> "GMDHClassifierBase":
        """Grow the GMDH network and, if enabled, calibrate its probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs. At least two features are required.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_fit_data(X, y, y_numeric=False)
        check_classification_targets(y)
        target_type = type_of_target(y, input_name="y")
        if target_type != "binary":
            raise ValueError("Only binary classification is supported. The type of the "
                             f"target is {target_type}.")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"{type(self).__name__} requires two classes in y; "
                             f"got {len(self.classes_)} class(es).")
        t = (y == self.classes_[1]).astype(float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self._fit_binary(X, t)
            self._fit_calibration(X, t)

        n_unconverged = 0
        reported = set()
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                n_unconverged += 1
            elif (str(w.message), w.category) not in reported:
                reported.add((str(w.message), w.category))
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if n_unconverged:
            warnings.warn(f"{n_unconverged} neuron fit(s) did not converge; increasing "
                          "ridge usually helps.", ConvergenceWarning, stacklevel=2)
        return self

    def _fit_binary(self, X: np.ndarray, t: np.ndarray) -> None:
        """Fit the uncalibrated network on inputs ``X`` and 0/1 targets ``t``."""
        Z = self._fit_input_scaling(X)
        rng = np.random.default_rng(self.random_state)
        train, select = self._split(len(t), rng)
        if np.unique(t[train]).size < 2:
            raise ValueError("The training split contains a single class; use more data "
                             "or a different training_split or random_state.")
        self._fit_network(Z[train], Z[select], t[train], t[select], rng)

    def _fit_calibration(self, X: np.ndarray, t: np.ndarray) -> None:
        """Fit Platt scaling on out-of-fold probabilities of refitted networks."""
        self.calibration_slope_, self.calibration_intercept_ = 1.0, 0.0
        if not self.calibrate:
            return
        if np.bincount(t.astype(int)).min() < self.calibration_cv:
            warnings.warn("Too few samples of the minority class for "
                          f"calibration_cv={self.calibration_cv}; probabilities are left "
                          "uncalibrated.", UserWarning, stacklevel=3)
            return

        folds = StratifiedKFold(self.calibration_cv, shuffle=True,
                                random_state=self.random_state)
        scores = np.empty(len(t))
        try:
            for fit_idx, val_idx in folds.split(X, t):
                fold_model = clone(self)
                fold_model._fit_binary(X[fit_idx], t[fit_idx])
                scores[val_idx] = fold_model._forward(fold_model._scale_inputs(X[val_idx]),
                                                      linear=True)
        except (ValueError, RuntimeError) as error:
            warnings.warn(f"Calibration was skipped because a calibration fold could not be "
                          f"fitted ({error}); probabilities are left uncalibrated.",
                          UserWarning, stacklevel=3)
            return
        self.calibration_slope_, self.calibration_intercept_ = _fit_platt(
            np.clip(scores, -_LOGIT_CLIP, _LOGIT_CLIP), t)

    @property
    def _is_calibrated(self) -> bool:
        return (self.calibration_slope_, self.calibration_intercept_) != (1.0, 0.0)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probabilities of ``classes_[0]`` and ``classes_[1]``.
        """
        Z = self._scale_inputs(self._validate_predict_data(X))
        if self._is_calibrated:
            z = np.clip(self._forward(Z, linear=True), -_LOGIT_CLIP, _LOGIT_CLIP)
            p = sigmoid(self.calibration_slope_ * z + self.calibration_intercept_)
        else:
            p = self._forward(Z)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        """Predict class labels for ``X`` (positive class when its probability is at least 0.5).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def equation(self, feature_names: Optional[Sequence[str]] = None,
                 precision: Optional[int] = 4, expand: Optional[bool] = None,
                 as_sympy: bool = False, variables: str = "standardized",
                 max_length: Optional[int] = _DEFAULT_MAX_LENGTH):
        """Closed-form equation for the probability of the positive class.

        All neurons are substituted in, so the equation is a nested
        composition of sigmoids. By default it is written in the standardized
        inputs ``z_k = (x_k - mean_k) / scale_k``, whose definitions are
        listed below the equation.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names used for the input variables. Defaults to the column names
            seen during :meth:`fit`, or ``x0, x1, ...``.
        precision : int or None, default=4
            Significant digits shown for the fitted coefficients in the
            returned string. Rounding is for display only: constants derived
            from the data (means, scales, ranges, knots) are always shown in
            full, and ``as_sympy=True`` always returns full precision.
            ``None`` shows every coefficient in full.
        expand : bool or None, default=None
            Whether to multiply out the arguments of the first-layer
            sigmoids, which are polynomials of the inputs for polynomial
            neurons. Off by default.
        as_sympy : bool, default=False
            Return SymPy objects instead of a string: the full-precision
            right-hand side and, with standardized variables, a dict mapping
            each ``z_k`` symbol to its definition in the original feature
            (``expr.subs(definitions)`` gives the equation in raw variables).
        variables : {"standardized", "raw"}, default="standardized"
            Write the equation in the standardized inputs ``z_k`` or directly
            in the original features. With ``expand=True`` an equation in raw
            variables can be numerically ill-conditioned for features whose
            mean is large relative to their scale; a ``RuntimeWarning`` is
            issued when that is likely.
        max_length : int or None, default=2000
            Largest equation that is built, measured in variable
            occurrences. Larger equations raise
            :class:`EquationTooLongError`; ``None`` removes the limit.

        Returns
        -------
        equation : str, sympy.Expr or (sympy.Expr, dict)
            ``"P(y = <positive class>) = ..."`` followed by the ``z_k``
            definitions, or the SymPy objects described under ``as_sympy``.

        Raises
        ------
        EquationTooLongError
            If the equation would exceed ``max_length``.
        """
        calibration = ((self.calibration_slope_, self.calibration_intercept_)
                       if self._is_calibrated else None)
        return self._equation(f"P(y = {self.classes_[1]})", feature_names, precision, expand,
                              as_sympy, variables, max_length, calibration=calibration)

    def summary(self, feature_names: Optional[Sequence[str]] = None,
                precision: Optional[int] = 4) -> str:
        """Layer-by-layer description of the fitted network.

        Neurons of the first layer take standardized inputs; later neurons
        take the probabilities produced by the previous layer. Use
        :meth:`equation` for the fully substituted model.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names used for the input variables.
        precision : int or None, default=4
            Number of significant digits shown for coefficients.

        Returns
        -------
        summary : str
        """
        check_is_fitted(self, self._fitted_attr)
        footer = f"P(y = {self.classes_[1]}) = output"
        if self._is_calibrated:
            footer = (f"P(y = {self.classes_[1]}) = sigmoid({self.calibration_slope_:.6g} * z "
                      f"+ {self.calibration_intercept_:.6g}), where output = sigmoid(z) and z "
                      f"is clipped to [-{_LOGIT_CLIP:g}, {_LOGIT_CLIP:g}]  [Platt scaling]")
        return self._summary_text(feature_names, precision, footer)
