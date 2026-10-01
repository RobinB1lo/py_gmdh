"""GMDH with trigonometric (Fourier) basis neurons."""

from typing import Optional

import numpy as np

from ._base import GMDHClassifierBase, GMDHRegressorBase, TensorBasisNeuron


class FourierNeuron(TensorBasisNeuron):
    """Neuron with basis ``sin(2πhu), cos(2πhu)`` for ``h = 1..n_harmonics``.

    Each input is rescaled to ``u`` in ``[0, 1]`` over its fitted range, so
    harmonic ``h`` completes ``h`` full periods across that range.
    """

    __slots__ = ("n_harmonics",)

    def __init__(self, i: int, j: int, n_harmonics: int = 3) -> None:
        super().__init__(i, j)
        self.n_harmonics = n_harmonics

    def basis(self, u, lib) -> list:
        t = 2.0 * np.pi * u
        functions = []
        for h in range(1, self.n_harmonics + 1):
            functions.extend([lib.sin(h * t), lib.cos(h * t)])
        return functions


class _FourierBasis:
    _expandable = False

    def _check_params(self) -> None:
        super()._check_params()
        if not isinstance(self.n_harmonics, (int, np.integer)) or self.n_harmonics < 1:
            raise ValueError(f"n_harmonics must be a positive integer, got {self.n_harmonics!r}.")

    def _make_neuron(self, i: int, j: int) -> FourierNeuron:
        return FourierNeuron(i, j, self.n_harmonics)


class FourierGMDH(_FourierBasis, GMDHRegressorBase):
    """GMDH regressor with Fourier-series neurons.

    Each neuron expands both of its inputs in ``n_harmonics`` sine/cosine
    pairs and fits the constant, the univariate terms and all pairwise
    products. Suited to targets with periodic or oscillatory structure.

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
    n_harmonics : int, default=3
        Number of sine/cosine pairs per input. A neuron has
        ``(2 * n_harmonics + 1) ** 2`` coefficients.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer.
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
                 threshold: Optional[float] = None, n_harmonics: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.n_harmonics = n_harmonics
        self.random_state = random_state


class FourierGMDHClassifier(_FourierBasis, GMDHClassifierBase):
    """GMDH binary classifier with Fourier-series neurons.

    Each neuron is a logistic model over ``n_harmonics`` sine/cosine pairs
    of both inputs, their constant and all pairwise products.

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
    n_harmonics : int, default=3
        Number of sine/cosine pairs per input. A neuron has
        ``(2 * n_harmonics + 1) ** 2`` coefficients.
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
        Layers of the fitted network, truncated at the best layer.
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
                 n_harmonics: int = 3,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.n_harmonics = n_harmonics
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
