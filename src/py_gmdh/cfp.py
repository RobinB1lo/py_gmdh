"""GMDH with constrained fractional polynomial neurons."""

from typing import Optional, Sequence

import numpy as np
import sympy as sp

from ._base import GMDHClassifierBase, GMDHRegressorBase, Neuron

DEFAULT_POWERS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_ZERO_EPS = 1e-10


def fractional_power(x, p: float, lib):
    """Fractional-polynomial transform of ``x`` with power ``p``.

    ``p = 0`` yields a constant term, integer powers keep the sign of ``x``
    and non-integer powers act on ``|x|``. Numerically, values of ``|x|``
    below ``1e-10`` are clipped to keep negative powers finite.
    """
    p = float(p)
    if lib is np:
        x = np.where(np.abs(x) < _ZERO_EPS, _ZERO_EPS, x)
        if p == 0.0:
            return np.ones_like(x)
        if p.is_integer():
            return x ** int(p)
        return np.abs(x) ** p
    if p == 0.0:
        return 1
    if p.is_integer():
        return x ** int(p)
    return sp.Abs(x) ** sp.Float(p)


class CFPNeuron(Neuron):
    """Additive neuron ``w0 + Σ_p (w_p a^p + v_p b^p)`` over a fixed set of powers."""

    __slots__ = ("powers",)

    def __init__(self, i: int, j: int, powers: Sequence[float] = DEFAULT_POWERS) -> None:
        super().__init__(i, j)
        self.powers = tuple(powers)

    def terms(self, a, b, lib) -> list:
        return [1,
                *[fractional_power(a, p, lib) for p in self.powers],
                *[fractional_power(b, p, lib) for p in self.powers]]


def _check_powers(powers) -> None:
    if len(powers) == 0:
        raise ValueError("At least one power is required.")
    if not all(np.isfinite(float(p)) for p in powers):
        raise ValueError(f"Powers must be finite numbers, got {powers!r}.")


class _CFPBasis:
    _expandable = False

    def _check_params(self) -> None:
        super()._check_params()
        _check_powers(self.powers)

    def _make_neuron(self, i: int, j: int) -> CFPNeuron:
        return CFPNeuron(i, j, self.powers)


class CFPGMDH(_CFPBasis, GMDHRegressorBase):
    """GMDH regressor with constrained fractional polynomial neurons.

    Every neuron is an additive fractional polynomial containing *all*
    powers from a fixed set for both of its inputs,
    ``w0 + Σ_p (w_p a^p + v_p b^p)``.

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
    powers : sequence of float, default=(-2, -1, -0.5, 0, 0.5, 1, 2)
        Powers included for each input.
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

    See Also
    --------
    UFPGMDH : Selects one power per input for each neuron instead.
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None,
                 powers: Sequence[float] = DEFAULT_POWERS,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.powers = powers
        self.random_state = random_state


class CFPGMDHClassifier(_CFPBasis, GMDHClassifierBase):
    """GMDH binary classifier with constrained fractional polynomial neurons.

    Every neuron is a logistic model of an additive fractional polynomial
    containing all powers from a fixed set for both inputs,
    ``sigmoid(w0 + Σ_p (w_p a^p + v_p b^p))``.

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
    powers : sequence of float, default=(-2, -1, -0.5, 0, 0.5, 1, 2)
        Powers included for each input.
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

    See Also
    --------
    UFPGMDHClassifier : Selects one power per input for each neuron instead.
    """

    def __init__(self, n_keep: int = 10, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 threshold: Optional[float] = None,
                 powers: Sequence[float] = DEFAULT_POWERS,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.powers = powers
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
