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

import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

_SCALE_EPS = 1e-12
_RANGE_EPS = 1e-10
_PROBA_EPS = 1e-8
_LOGIT_CLIP = 30.0
_AUTO_EXPAND_MAX_DEPTH = 3

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

    The design matrix already contains a constant column. ``ridge`` is the
    inverse of scikit-learn's ``C``.
    """
    model = LogisticRegression(C=1.0 / max(ridge, 1e-12), fit_intercept=False,
                               solver="lbfgs", max_iter=1000)
    return model.fit(X, y).coef_[0]


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

    def objective(params):
        z = params[0] * scores + params[1]
        residual = expit(z) - target
        loss = float(np.sum(np.logaddexp(0.0, z) - target * z))
        return loss, np.array([residual @ scores, residual.sum()])

    start = np.array([1.0, np.log((n_pos + 1) / (n_neg + 1))])
    result = minimize(objective, start, jac=True, method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def _unit_scale(x, lo: float, hi: float):
    """Map ``x`` from ``[lo, hi]`` onto ``[0, 1]`` (NumPy or SymPy input)."""
    return (x - lo) / (hi - lo + _RANGE_EPS)


def _round_floats(expr: sp.Expr, digits: int) -> sp.Expr:
    """Round every floating-point constant in ``expr`` to ``digits`` significant digits."""
    rounded = {f: sp.Float(float(f"{float(f):.{digits}g}")) for f in expr.atoms(sp.Float)}
    with sp.evaluate(False):
        return expr.xreplace(rounded)


def _sigmoid_expression(z: sp.Expr) -> sp.Expr:
    """``1 / (1 + exp(-z))`` kept in this form rather than simplified by SymPy."""
    with sp.evaluate(False):
        return 1 / (1 + sp.exp(-z))


def _expand_sigmoid_arguments(expr: sp.Expr) -> sp.Expr:
    """Multiply out the innermost sigmoid arguments, which are functions of the inputs only."""
    replacements = {node: sp.exp(sp.expand(node.args[0]), evaluate=False)
                    for node in expr.atoms(sp.exp) if not node.args[0].has(sp.exp)}
    with sp.evaluate(False):
        return expr.xreplace(replacements)


def _format_expr(expr: sp.Expr, precision: Optional[int]) -> str:
    if precision is not None:
        expr = _round_floats(expr, precision)
    return sp.sstr(expr)


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

    def _linear_expression(self, a, b) -> sp.Expr:
        return sp.Add(*[sp.Float(float(w)) * t
                        for w, t in zip(self.w, self.terms(a, b, sp))])

    def expression(self, a: sp.Expr, b: sp.Expr) -> sp.Expr:
        """Symbolic form of the fitted neuron with inputs ``a`` and ``b``."""
        linear = self._linear_expression(a, b)
        if self.link == LOGISTIC:
            return _sigmoid_expression(linear)
        return linear

    def _describe_linear(self, a_name: str, b_name: str, precision: Optional[int]) -> str:
        return _format_expr(self._linear_expression(sp.Symbol(a_name), sp.Symbol(b_name)),
                            precision)

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

    def _scaled_input_expressions(self, feature_names) -> List[sp.Expr]:
        symbols = self._feature_symbols(feature_names)
        return [(s - float(m)) / float(d)
                for s, m, d in zip(symbols, self.x_mean_, self.x_scale_)]

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

    def _output_expression(self, inputs: List[sp.Expr], linear: bool = False) -> sp.Expr:
        memo: Dict[Tuple[int, int], sp.Expr] = {}

        def arguments(depth: int, neuron: Neuron):
            if depth == 0:
                return inputs[neuron.i], inputs[neuron.j]
            return node(depth - 1, neuron.i), node(depth - 1, neuron.j)

        def node(depth: int, index: int) -> sp.Expr:
            key = (depth, index)
            if key not in memo:
                neuron = self.layers_[depth].neurons[index]
                memo[key] = neuron.expression(*arguments(depth, neuron))
            return memo[key]

        depth = len(self.layers_) - 1
        output = self.layers_[depth].neurons[0]
        if linear:
            return output._linear_expression(*arguments(depth, output))
        return node(depth, 0)

    def _depth(self) -> int:
        return len(self.layers_)

    def _summary_body(self, names: List[str], precision: Optional[int]) -> List[str]:
        lines = []
        inputs = names
        for depth, layer in enumerate(self.layers_):
            metrics = ", ".join(f"{k}={v:.6g}" for k, v in layer.metrics.items())
            lines.append(f"Layer {depth + 1} ({len(layer.neurons)} neurons; best: {metrics})")
            outputs = []
            for index, neuron in enumerate(layer.neurons):
                name = f"z{depth + 1}_{index}"
                formula = neuron.describe(inputs[neuron.i], inputs[neuron.j], precision)
                lines.append(f"  {name} = {formula}")
                outputs.append(name)
            inputs = outputs
        lines.append(f"Output: {inputs[0]}")
        return lines

    # -- shared public helpers ------------------------------------------------

    def _network_expression(self, feature_names, precision, expand, scale=None, offset=None,
                            calibration=None):
        check_is_fitted(self, self._fitted_attr)
        inputs = self._scaled_input_expressions(feature_names)
        if calibration is None:
            expr = self._output_expression(inputs)
        else:
            slope, intercept = calibration
            logit = self._output_expression(inputs, linear=True)
            with sp.evaluate(False):
                score = sp.Float(slope) * logit + sp.Float(intercept)
            expr = _sigmoid_expression(score)
        if scale is not None:
            expr = expr * sp.Float(scale) + sp.Float(offset)
        if expand is None:
            expand = self._expandable and self._depth() <= _AUTO_EXPAND_MAX_DEPTH
        if expand:
            expr = _expand_sigmoid_arguments(expr) if self._link == LOGISTIC else sp.expand(expr)
        if precision is not None:
            expr = _round_floats(expr, precision)
        return expr

    def _summary_text(self, feature_names, precision, footer: str) -> str:
        check_is_fitted(self, self._fitted_attr)
        names = [s.name for s in self._feature_symbols(feature_names)]
        header = [f"{type(self).__name__} (selection criterion: {self._criterion}, "
                  f"best score: {self.best_score_:.6g})",
                  "Inputs are standardized: z = (x - mean) / scale.", ""]
        return "\n".join(header + self._summary_body(names, precision) + ["", footer])


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
                 as_sympy: bool = False):
        """Closed-form equation of the fitted network.

        The equation is written in terms of the original, unstandardized
        features; all intermediate neurons are substituted in.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names used for the input variables. Defaults to the column names
            seen during :meth:`fit`, or ``x0, x1, ...``.
        precision : int or None, default=4
            Number of significant digits kept for every constant. ``None``
            keeps full precision, which reproduces :meth:`predict` exactly.
        expand : bool or None, default=None
            Whether to multiply the nested expression out. By default the
            equation is expanded only for polynomial networks with at most
            three layers; deeper networks grow combinatorially when expanded.
        as_sympy : bool, default=False
            Return a :class:`sympy.Expr` for the right-hand side instead of a
            string.

        Returns
        -------
        equation : str or sympy.Expr
            ``"y = ..."`` or the SymPy expression of the right-hand side.
        """
        check_is_fitted(self, self._fitted_attr)
        expr = self._network_expression(feature_names, precision, expand,
                                        scale=self.y_scale_, offset=self.y_mean_)
        return expr if as_sympy else f"y = {sp.sstr(expr)}"

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
        footer = f"y = output * {self.y_scale_:.6g} + {self.y_mean_:.6g}"
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
    ``z`` is the output neuron's logit. The two parameters are fitted on
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
        self.calibration_slope_, self.calibration_intercept_ = _fit_platt(scores, t)

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
            p = expit(self.calibration_slope_ * self._forward(Z, linear=True)
                      + self.calibration_intercept_)
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
                 as_sympy: bool = False):
        """Closed-form equation for the probability of the positive class.

        The equation is written in terms of the original, unstandardized
        features; all intermediate neurons are substituted in, so it is a
        nested composition of sigmoids.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names used for the input variables. Defaults to the column names
            seen during :meth:`fit`, or ``x0, x1, ...``.
        precision : int or None, default=4
            Number of significant digits kept for every constant. ``None``
            keeps full precision, which reproduces :meth:`predict_proba`.
        expand : bool or None, default=None
            Whether to multiply out the arguments of the first-layer
            sigmoids, which are polynomials of the inputs for polynomial
            neurons. Off by default.
        as_sympy : bool, default=False
            Return a :class:`sympy.Expr` for the right-hand side instead of a
            string.

        Returns
        -------
        equation : str or sympy.Expr
            ``"P(y = <positive class>) = ..."`` or the SymPy expression of the
            right-hand side.
        """
        check_is_fitted(self, self._fitted_attr)
        calibration = ((self.calibration_slope_, self.calibration_intercept_)
                       if self._is_calibrated else None)
        expr = self._network_expression(feature_names, precision, expand,
                                        calibration=calibration)
        return expr if as_sympy else f"P(y = {self.classes_[1]}) = {sp.sstr(expr)}"

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
                      f"+ {self.calibration_intercept_:.6g}), where output = sigmoid(z) "
                      "[Platt scaling]")
        return self._summary_text(feature_names, precision, footer)
