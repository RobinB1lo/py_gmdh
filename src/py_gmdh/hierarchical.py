"""GMDH with lexicographic (hierarchical) multi-metric neuron selection."""

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from ._base import _PROBA_EPS, GMDHClassifierBase, GMDHRegressorBase


class _HierarchicalSelection:
    """Rank candidates by RMSE, breaking near-ties with MAE, R², AIC and adjusted R²."""

    def _check_params(self) -> None:
        super()._check_params()
        if not isinstance(self.rmse_decimals, (int, np.integer)) or self.rmse_decimals < 0:
            raise ValueError("rmse_decimals must be a non-negative integer, "
                             f"got {self.rmse_decimals!r}.")

    def _score(self, neuron, a_tr, b_tr, y_tr, a_se, b_se, y_se, context):
        residual = y_se - neuron.predict(a_se, b_se)
        n = len(y_se)
        k = neuron.w.size

        rss = float(np.sum(residual ** 2))
        tss = float(np.sum((y_se - y_se.mean()) ** 2)) + 1e-12
        rmse = float(np.sqrt(rss / n))
        mae = float(np.mean(np.abs(residual)))
        r2 = 1.0 - rss / tss
        aic = n * np.log(max(rss, 1e-12) / n) + 2 * k
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2

        metrics = {"rmse": rmse, "mae": mae, "r2": r2, "aic": aic, "adj_r2": adj_r2}
        if not all(np.isfinite(v) for v in metrics.values()):
            return (np.inf,), {**metrics, "rmse": np.inf}
        key = (round(rmse, self.rmse_decimals), mae, -r2, aic, -adj_r2)
        return key, metrics


class _HierarchicalClassification:
    """Rank candidates by log loss, breaking near-ties with Brier score, AUC,
    AIC and adjusted McFadden R²."""

    _improvement_tol = 1e-6

    def _check_params(self) -> None:
        super()._check_params()
        if (not isinstance(self.log_loss_decimals, (int, np.integer))
                or self.log_loss_decimals < 0):
            raise ValueError("log_loss_decimals must be a non-negative integer, "
                             f"got {self.log_loss_decimals!r}.")

    def _score(self, neuron, a_tr, b_tr, y_tr, a_se, b_se, y_se, context):
        p = np.clip(neuron.predict(a_se, b_se), _PROBA_EPS, 1 - _PROBA_EPS)
        n = len(y_se)
        k = neuron.w.size

        nll = float(-np.sum(y_se * np.log(p) + (1 - y_se) * np.log(1 - p)))
        log_loss = nll / n
        brier = float(np.mean((p - y_se) ** 2))
        auc = float(roc_auc_score(y_se, p)) if np.unique(y_se).size == 2 else 0.5
        aic = 2 * k + 2 * nll
        base_rate = float(np.clip(y_se.mean(), _PROBA_EPS, 1 - _PROBA_EPS))
        null_nll = max(-n * (base_rate * np.log(base_rate)
                             + (1 - base_rate) * np.log(1 - base_rate)), 1e-12)
        adj_mcfadden_r2 = 1.0 - (nll + k) / null_nll

        metrics = {"log_loss": log_loss, "brier": brier, "auc": auc, "aic": aic,
                   "adj_mcfadden_r2": adj_mcfadden_r2}
        if not all(np.isfinite(v) for v in metrics.values()):
            return (np.inf,), {**metrics, "log_loss": np.inf}
        key = (round(log_loss, self.log_loss_decimals), brier, -auc, aic, -adj_mcfadden_r2)
        return key, metrics


class HierarchicalGMDH(_HierarchicalSelection, GMDHRegressorBase):
    """GMDH regressor with hierarchical multi-metric neuron selection.

    Candidates are ranked lexicographically on the selection set: first by
    RMSE rounded to ``rmse_decimals`` decimals, then by MAE, R², AIC and
    adjusted R². Neurons whose RMSE is practically indistinguishable are
    therefore separated by the secondary criteria rather than by noise.
    Early stopping uses the (unrounded) selection RMSE.

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
    rmse_decimals : int, default=3
        Decimals to which RMSE is rounded before the tie-breaking metrics
        are consulted.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer. Each
        layer's ``metrics`` holds ``rmse``, ``mae``, ``r2``, ``aic`` and
        ``adj_r2`` of its best neuron.
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
                 threshold: Optional[float] = None, rmse_decimals: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.rmse_decimals = rmse_decimals
        self.random_state = random_state


class HierarchicalGMDHClassifier(_HierarchicalClassification, GMDHClassifierBase):
    """GMDH binary classifier with hierarchical multi-metric neuron selection.

    Candidates are ranked lexicographically on the selection set: first by
    log loss rounded to ``log_loss_decimals`` decimals, then by Brier score,
    AUC, AIC (``2k + 2 NLL``) and adjusted McFadden pseudo-R²
    (``1 - (NLL + k) / NLL_null``). Early stopping uses the (unrounded)
    selection log loss, and a layer counts as an improvement only if it
    lowers the log loss by more than ``1e-6``.

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
    log_loss_decimals : int, default=3
        Decimals to which log loss is rounded before the tie-breaking
        metrics are consulted.
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
        layer's ``metrics`` holds ``log_loss``, ``brier``, ``auc``, ``aic`` and
        ``adj_mcfadden_r2`` of its best neuron.
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
                 log_loss_decimals: int = 3,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.log_loss_decimals = log_loss_decimals
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
