"""GMDH with k-fold cross-validated neuron selection."""

from typing import List, Optional, Tuple

import numpy as np

from ._base import LOGISTIC, GMDHClassifierBase, GMDHRegressorBase


class _CrossValidatedSelection:
    """Rank candidates by k-fold cross-validated loss on the selection set.

    For every candidate pair of inputs, the same neuron type is refitted on
    ``n_folds - 1`` folds of the selection set and scored on the held-out
    fold. The neuron that is kept is the one fitted on the training set.
    For classifiers, folds whose fitting part contains a single class are
    skipped, and a candidate without any usable fold is discarded.
    """

    @property
    def _criterion(self) -> str:
        return f"cv_{self._base_metric}"

    def _check_params(self) -> None:
        super()._check_params()
        if not isinstance(self.n_folds, (int, np.integer)) or self.n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {self.n_folds!r}.")

    def _layer_context(self, n_selection: int,
                       rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
        if n_selection < self.n_folds:
            raise ValueError(f"The selection set has {n_selection} samples, fewer than "
                             f"n_folds={self.n_folds}.")
        order = rng.permutation(n_selection)
        folds = np.array_split(order, self.n_folds)
        return [(np.concatenate(folds[:k] + folds[k + 1:]), folds[k])
                for k in range(self.n_folds)]

    def _score(self, neuron, a_tr, b_tr, y_tr, a_se, b_se, y_se, folds):
        errors = []
        for fit_idx, val_idx in folds:
            if self._link == LOGISTIC and np.unique(y_se[fit_idx]).size < 2:
                continue
            fold_neuron = self._create_neuron(neuron.i, neuron.j).fit(
                a_se[fit_idx], b_se[fit_idx], y_se[fit_idx], self.ridge)
            errors.append(fold_neuron.loss(
                y_se[val_idx], fold_neuron.predict(a_se[val_idx], b_se[val_idx])))
        if not errors:
            return None
        cv_error = float(np.mean(errors))
        return cv_error, {self._criterion: cv_error}


class CVGMDH(_CrossValidatedSelection, GMDHRegressorBase):
    """GMDH regressor with cross-validated neuron selection.

    The data are split once into training and selection sets. Each candidate
    neuron is fitted on the training set, while its ranking score is the
    mean RMSE of k-fold cross-validation performed on the selection set. This
    gives a lower-variance external criterion than a single hold-out score.

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
        selection set used for cross-validation.
    patience : int, default=0
        Number of consecutive non-improving layers tolerated before growth
        stops.
    threshold : float, optional
        Stop as soon as the best cross-validated RMSE (on the standardized
        target) falls to or below this value.
    n_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, optional
        Seed controlling the train/selection split and the fold assignment.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer.
    best_score_ : float
        Cross-validated RMSE of the output neuron on the standardized target.
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
                 threshold: Optional[float] = None, n_folds: int = 5,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.n_folds = n_folds
        self.random_state = random_state


class CVGMDHClassifier(_CrossValidatedSelection, GMDHClassifierBase):
    """GMDH binary classifier with cross-validated neuron selection.

    The data are split once into training and selection sets. Each
    candidate neuron is fitted on the training set, while its ranking score
    is the mean log loss of k-fold cross-validation performed on the
    selection set.

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
        Stop as soon as the best selection cross-validated log loss falls to or below this
        value.
    n_folds : int, default=5
        Number of cross-validation folds.
    calibrate : bool, default=True
        Recalibrate the output probability with Platt scaling fitted on
        out-of-fold predictions. This refits the network ``calibration_cv``
        additional times.
    calibration_cv : int, default=3
        Number of stratified folds used to fit the calibration.
    random_state : int, optional
        Seed controlling the train/selection split and the fold assignment.

    Attributes
    ----------
    layers_ : list of Layer
        Layers of the fitted network, truncated at the best layer.
    best_score_ : float
        Cross-validated log loss of the output neuron.
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
                 n_folds: int = 5,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.n_folds = n_folds
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
