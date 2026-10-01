"""Classical GMDH with Ivakhnenko polynomial neurons."""

from typing import Optional

from ._base import GMDHClassifierBase, GMDHRegressorBase


class GMDH(GMDHRegressorBase):
    """Multilayer GMDH regressor with quadratic polynomial neurons.

    Each neuron models a pair of inputs with the Ivakhnenko polynomial
    ``w0 + w1 a + w2 b + w3 ab + w4 a² + w5 b²``. Candidates are fitted on a
    training split and ranked by their RMSE on a held-out selection split.

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
        Feature names seen during :meth:`fit`, when ``X`` has string column
        names.
    x_mean_, x_scale_ : ndarray of shape (n_features_in_,)
        Per-feature standardization statistics.
    y_mean_, y_scale_ : float
        Target standardization statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from py_gmdh.gmdh import GMDH
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, size=(200, 3))
    >>> y = X[:, 0] * X[:, 1] + X[:, 2] ** 2
    >>> model = GMDH(n_keep=6, random_state=0).fit(X, y)
    >>> model.predict(X[:3]).shape
    (3,)
    >>> equation = model.equation(feature_names=["a", "b", "c"])
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.random_state = random_state


class GMDHClassifier(GMDHClassifierBase):
    """Multilayer GMDH binary classifier with quadratic polynomial neurons.

    Each neuron is a logistic model of the Ivakhnenko polynomial,
    ``sigmoid(w0 + w1 a + w2 b + w3 ab + w4 a² + w5 b²)``. Candidates are
    fitted on a training split and ranked by their log loss on a held-out
    selection split; later layers combine the probabilities produced by the
    previous layer.

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

    Examples
    --------
    >>> import numpy as np
    >>> from py_gmdh.gmdh import GMDHClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 3))
    >>> y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)
    >>> model = GMDHClassifier(n_keep=4, random_state=0).fit(X, y)
    >>> model.predict_proba(X[:3]).shape
    (3, 2)
    >>> equation = model.equation(feature_names=["a", "b", "c"])
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
