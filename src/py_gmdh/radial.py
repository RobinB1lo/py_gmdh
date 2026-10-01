"""GMDH with Gaussian radial basis function neurons."""

from typing import Optional

import numpy as np

from ._base import GMDHClassifierBase, GMDHRegressorBase, TensorBasisNeuron


class RadialNeuron(TensorBasisNeuron):
    """Neuron with Gaussian basis ``exp(-gamma (u - c)²)``.

    Centers ``c`` are spaced evenly over ``[0, 1]``, the range to which each
    input is rescaled.
    """

    __slots__ = ("n_centers", "gamma")

    def __init__(self, i: int, j: int, n_centers: int = 5, gamma: float = 5.0) -> None:
        super().__init__(i, j)
        self.n_centers = n_centers
        self.gamma = gamma

    def basis(self, u, lib) -> list:
        return [lib.exp(-self.gamma * (u - float(c)) ** 2)
                for c in np.linspace(0.0, 1.0, self.n_centers)]


class _RadialBasis:
    _expandable = False

    def _check_params(self) -> None:
        super()._check_params()
        if not isinstance(self.n_centers, (int, np.integer)) or self.n_centers < 1:
            raise ValueError(f"n_centers must be a positive integer, got {self.n_centers!r}.")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma!r}.")

    def _make_neuron(self, i: int, j: int) -> RadialNeuron:
        return RadialNeuron(i, j, self.n_centers, self.gamma)


class RadialGMDH(_RadialBasis, GMDHRegressorBase):
    """GMDH regressor with Gaussian radial basis function neurons.

    Each neuron expands both inputs in ``n_centers`` Gaussian bumps and fits
    the constant, the univariate terms and all pairwise products, giving a
    smooth, localized approximation.

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
    n_centers : int, default=5
        Number of Gaussian centers per input.
    gamma : float, default=5.0
        Width parameter ``1 / (2 sigma²)`` of each Gaussian, measured on the
        unit-rescaled input.
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
                 threshold: Optional[float] = None, n_centers: int = 5,
                 gamma: float = 5.0, random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.n_centers = n_centers
        self.gamma = gamma
        self.random_state = random_state


class RadialGMDHClassifier(_RadialBasis, GMDHClassifierBase):
    """GMDH binary classifier with Gaussian radial basis function neurons.

    Each neuron is a logistic model over ``n_centers`` Gaussian bumps of
    both inputs, their constant and all pairwise products.

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
    n_centers : int, default=5
        Number of Gaussian centers per input.
    gamma : float, default=5.0
        Width parameter ``1 / (2 sigma²)`` of each Gaussian, measured on the
        unit-rescaled input.
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
                 n_centers: int = 5,
                 gamma: float = 5.0,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.n_centers = n_centers
        self.gamma = gamma
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
