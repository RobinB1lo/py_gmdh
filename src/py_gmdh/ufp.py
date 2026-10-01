"""GMDH with unconstrained fractional polynomial neurons."""

from typing import Optional, Sequence

import numpy as np

from ._base import GMDHClassifierBase, GMDHRegressorBase, Neuron
from .cfp import DEFAULT_POWERS, _check_powers, fractional_power


class UFPNeuron(Neuron):
    """Neuron ``w0 + w1 a^p + w2 b^q`` with powers chosen during fitting.

    The pair ``(p, q)`` is selected from ``candidate_powers`` by grid search
    on the training data, minimizing the neuron's own loss (RMSE or log
    loss). A power of ``0`` removes that input from the neuron.
    """

    __slots__ = ("candidate_powers", "power_a", "power_b")

    def __init__(self, i: int, j: int,
                 candidate_powers: Sequence[float] = DEFAULT_POWERS) -> None:
        super().__init__(i, j)
        self.candidate_powers = tuple(float(p) for p in candidate_powers)
        self.power_a = 1.0
        self.power_b = 1.0

    def terms(self, a, b, lib) -> list:
        return [1, fractional_power(a, self.power_a, lib), fractional_power(b, self.power_b, lib)]

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray, ridge: float) -> "UFPNeuron":
        ones = np.ones_like(a)
        a_terms = {p: fractional_power(a, p, np) for p in self.candidate_powers}
        b_terms = {p: fractional_power(b, p, np) for p in self.candidate_powers}

        best_error = np.inf
        best = (self.candidate_powers[0], self.candidate_powers[0])
        for p in self.candidate_powers:
            for q in self.candidate_powers:
                X = np.column_stack([ones, a_terms[p], b_terms[q]])
                error = self.loss(y, self._activate(X @ self._solve(X, y, ridge)))
                if error < best_error:
                    best_error, best = error, (p, q)

        self.power_a, self.power_b = best
        self.w = self._solve(self.design_matrix(a, b), y, ridge)
        return self


class _UFPBasis:
    _expandable = False

    def _check_params(self) -> None:
        super()._check_params()
        _check_powers(self.candidate_powers)

    def _make_neuron(self, i: int, j: int) -> UFPNeuron:
        return UFPNeuron(i, j, self.candidate_powers)


class UFPGMDH(_UFPBasis, GMDHRegressorBase):
    """GMDH regressor with unconstrained fractional polynomial neurons.

    Each neuron has the form ``w0 + w1 a^p + w2 b^q``, where the powers
    ``p`` and ``q`` are chosen independently for every neuron from
    ``candidate_powers`` by grid search on the training split. Integer powers
    keep the sign of the input, fractional powers act on its absolute value
    and a power of ``0`` drops the input.

    Parameters
    ----------
    n_keep : int, default=10
        Number of neurons kept in each layer and passed on to the next.
    max_layers : int, default=10
        Maximum number of layers to grow.
    ridge : float, default=1e-6
        L2 regularization used when fitting each neuron.
    training_split : float, default=0.5
        Fraction of samples used to fit neurons; the remainder forms the
        selection set used to rank them.
    patience : int, default=0
        Number of consecutive non-improving layers tolerated before growth
        stops.
    threshold : float, optional
        Stop as soon as the best selection RMSE (on the standardized target)
        falls to or below this value.
    candidate_powers : sequence of float, default=(-2, -1, -0.5, 0, 0.5, 1, 2)
        Powers searched for each input. Pass ``(1,)`` for linear neurons.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer. The
        selected powers are available as ``power_a`` and ``power_b`` on each
        neuron.
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

    See Also
    --------
    CFPGMDH : Uses every power of a fixed set in each neuron instead.
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None,
                 candidate_powers: Sequence[float] = DEFAULT_POWERS,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.candidate_powers = candidate_powers
        self.random_state = random_state


class UFPGMDHClassifier(_UFPBasis, GMDHClassifierBase):
    """GMDH binary classifier with unconstrained fractional polynomial neurons.

    Each neuron has the form ``sigmoid(w0 + w1 a^p + w2 b^q)``, where the
    powers ``p`` and ``q`` are chosen independently for every neuron from
    ``candidate_powers`` by minimizing the training log loss. Integer powers
    keep the sign of the input, fractional powers act on its absolute value
    and a power of ``0`` drops the input.

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
    candidate_powers : sequence of float, default=(-2, -1, -0.5, 0, 0.5, 1, 2)
        Powers searched for each input. Pass ``(1,)`` for linear neurons.
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
        selected powers are available as ``power_a`` and ``power_b`` on each
        neuron.
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

    See Also
    --------
    CFPGMDHClassifier : Uses every power of a fixed set in each neuron instead.
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None,
                 candidate_powers: Sequence[float] = DEFAULT_POWERS,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.candidate_powers = candidate_powers
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
