"""GMDH with neuron selection by the Akaike information criterion."""

from typing import Optional

import numpy as np

from ._base import _PROBA_EPS, GMDHClassifierBase, GMDHRegressorBase


class _AICSelection:
    """Rank candidates by ``AIC = n ln(RSS / n) + 2k`` on the selection set."""

    _criterion = "aic"

    def _score(self, neuron, a_tr, b_tr, y_tr, a_se, b_se, y_se, context):
        residual = y_se - neuron.predict(a_se, b_se)
        n = len(y_se)
        rss = float(np.sum(residual ** 2))
        aic = n * np.log(max(rss, 1e-12) / n) + 2 * neuron.w.size
        rmse = float(np.sqrt(rss / n))
        if not np.isfinite(aic):
            aic = np.inf
        return aic, {"aic": aic, "rmse": rmse}


class _AICClassification:
    """Rank candidates by ``AIC = 2k + 2 NLL`` on the selection set.

    The AIC grows with the size of the selection set, so early stopping uses
    the mean log loss instead, which is comparable across layers.
    """

    _criterion = "log_loss"
    _improvement_tol = 1e-6

    def _score(self, neuron, a_tr, b_tr, y_tr, a_se, b_se, y_se, context):
        p = np.clip(neuron.predict(a_se, b_se), _PROBA_EPS, 1 - _PROBA_EPS)
        nll = float(-np.sum(y_se * np.log(p) + (1 - y_se) * np.log(1 - p)))
        aic = 2 * neuron.w.size + 2 * nll
        if not np.isfinite(aic):
            return np.inf, {"aic": np.inf, "log_loss": np.inf}
        return aic, {"aic": aic, "log_loss": nll / len(y_se)}


class AICGMDH(_AICSelection, GMDHRegressorBase):
    """GMDH regressor that selects neurons by the Akaike information criterion.

    The AIC ``n ln(RSS / n) + 2k`` is evaluated on the selection set, where
    ``k`` is the number of neuron coefficients. Both neuron ranking and early
    stopping use the AIC.

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
        Stop as soon as the best selection AIC falls to or below this value.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer. Each
        layer's ``metrics`` holds ``aic`` and ``rmse`` of its best neuron.
    best_score_ : float
        Selection AIC of the output neuron.
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
                 threshold: Optional[float] = None,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.random_state = random_state


class AICGMDHClassifier(_AICClassification, GMDHClassifierBase):
    """GMDH binary classifier that selects neurons by the Akaike information criterion.

    The AIC generalized to the Bernoulli likelihood, ``2k + 2 NLL``, is
    evaluated on the selection set, where ``k`` is the number of neuron
    coefficients and ``NLL`` the negative log-likelihood. Neurons are ranked
    by AIC; early stopping uses the mean selection log loss, and a layer
    counts as an improvement only if it lowers it by more than ``1e-6``.

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
        Layers of the fitted network, truncated at the best layer. Each
        layer's ``metrics`` holds ``aic`` and ``log_loss`` of its best neuron.
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
