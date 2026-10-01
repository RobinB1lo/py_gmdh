"""Combinatorial GMDH (COMBI).

COMBI is Ivakhnenko's single-layer GMDH algorithm. A reference function,
here a polynomial of the inputs, is expanded into its terms, and every
partial model made of a subset of those terms is fitted on the training
split. Models are generated with steadily increasing complexity (number of
terms) and ranked by an external criterion; the model with the lowest
criterion over all complexity levels is selected.
"""

import warnings
from itertools import combinations
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.special import expit

from ._base import (_LOGIT_CLIP, _PROBA_EPS, LOGISTIC, GMDHClassifierBase, GMDHRegressorBase,
                    _format_expr, _logistic_solve, _ridge_solve, _sigmoid_expression)

Term = Tuple[int, ...]

REFERENCES = ("linear", "interaction", "quadratic")
CRITERIA = ("regularity", "press")
_BATCH_ELEMENTS = 4_000_000
_NEWTON_ITERATIONS = 25


def reference_terms(n_features: int, reference: str) -> List[Term]:
    """Terms of the reference polynomial, excluding the intercept.

    ``"linear"`` gives ``x_i``; ``"interaction"`` adds ``x_i x_j`` for
    ``i < j``; ``"quadratic"`` (the Ivakhnenko polynomial generalized to all
    inputs) also adds the squares ``x_i²``.
    """
    terms: List[Term] = [(k,) for k in range(n_features)]
    if reference == "interaction":
        terms += [(i, j) for i, j in combinations(range(n_features), 2)]
    elif reference == "quadratic":
        terms += [(i, j) for i in range(n_features) for j in range(i, n_features)]
    return terms


def _design(Z: np.ndarray, terms: List[Term]) -> np.ndarray:
    columns = [np.ones(len(Z))]
    columns += [np.prod(Z[:, list(term)], axis=1) for term in terms]
    return np.column_stack(columns)


def _batch_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(a, v, rcond=None)[0] for a, v in zip(A, b)])


class _CombinatorialSearch:
    """Exhaustive single-layer search over subsets of reference terms."""

    _fitted_attr = "coef_"

    @property
    def _criterion(self) -> str:
        if self._link == LOGISTIC:
            return "log_loss"
        return "press_rmse" if self.criterion == "press" else "rmse"

    def _check_search_params(self) -> None:
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {self.reference!r}.")
        if self._link != LOGISTIC and self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {self.criterion!r}.")
        if self.max_terms is not None and (not isinstance(self.max_terms, (int, np.integer))
                                           or self.max_terms < 1):
            raise ValueError(f"max_terms must be a positive integer or None, "
                             f"got {self.max_terms!r}.")
        if self.patience is not None and self.patience < 0:
            raise ValueError(f"patience must be non-negative or None, got {self.patience!r}.")
        if not isinstance(self.max_models, (int, np.integer)) or self.max_models < 1:
            raise ValueError(f"max_models must be a positive integer, got {self.max_models!r}.")

    # -- search -------------------------------------------------------------

    def _complexity_levels(self, n_terms: int) -> int:
        limit = n_terms if self.max_terms is None else min(self.max_terms, n_terms)
        total = 1
        for k in range(1, limit + 1):
            if total + comb(n_terms, k) > self.max_models:
                warnings.warn(f"The search was limited to models with at most {k - 1} terms "
                              f"to stay within max_models={self.max_models}; set max_terms "
                              "or increase max_models to change this.", UserWarning,
                              stacklevel=4)
                return k - 1
            total += comb(n_terms, k)
        return limit

    def _fit_network(self, Z_tr, Z_se, y_tr, y_se, rng) -> None:
        terms = reference_terms(Z_tr.shape[1], self.reference)
        use_press = self._link != LOGISTIC and self.criterion == "press"
        if use_press:
            Z_all, y_all = np.vstack([Z_tr, Z_se]), np.concatenate([y_tr, y_se])
            scorer = self._press_scorer(_design(Z_all, terms), y_all)
        elif self._link == LOGISTIC:
            scorer = self._logistic_scorer(_design(Z_tr, terms), y_tr,
                                           _design(Z_se, terms), y_se)
        else:
            scorer = self._regularity_scorer(_design(Z_tr, terms), y_tr,
                                             _design(Z_se, terms), y_se)

        levels: List[Dict] = []
        best_score, best_subset, stalled = np.inf, (0,), 0
        for k in range(0, self._complexity_levels(len(terms)) + 1):
            level_score, level_subset = np.inf, None
            subsets = combinations(range(1, len(terms) + 1), k)
            batch_size = max(1, _BATCH_ELEMENTS // ((k + 1) ** 2 * scorer.rows))
            while True:
                chunk = [(0,) + s for s in _take(subsets, batch_size)]
                if not chunk:
                    break
                scores = scorer(np.array(chunk))
                scores[~np.isfinite(scores)] = np.inf
                j = int(np.argmin(scores))
                if scores[j] < level_score:
                    level_score, level_subset = float(scores[j]), chunk[j]
            if level_subset is None:
                break
            levels.append({"n_terms": k, "score": level_score,
                           "terms": [terms[c - 1] for c in level_subset[1:]]})

            if level_score < best_score - self._improvement_tol:
                best_score, best_subset, stalled = level_score, level_subset, 0
            else:
                stalled += 1
            if self.threshold is not None and best_score <= self.threshold:
                break
            if self.patience is not None and stalled > self.patience:
                break

        if not np.isfinite(best_score):
            raise RuntimeError("No model produced a finite selection score.")

        self.terms_ = [terms[c - 1] for c in best_subset[1:]]
        self.levels_ = levels
        self.best_score_ = best_score
        if self.refit or use_press:
            Z_fit, y_fit = np.vstack([Z_tr, Z_se]), np.concatenate([y_tr, y_se])
        else:
            Z_fit, y_fit = Z_tr, y_tr
        design = _design(Z_fit, self.terms_)
        if self._link == LOGISTIC:
            self.coef_ = _logistic_solve(design, y_fit, self.ridge)
        else:
            self.coef_ = _ridge_solve(design, y_fit, self.ridge)

    def _regularity_scorer(self, P_tr, y_tr, P_se, y_se):
        gram, rhs = P_tr.T @ P_tr, P_tr.T @ y_tr
        gram_se, rhs_se, yy_se = P_se.T @ P_se, P_se.T @ y_se, float(y_se @ y_se)
        ridge, n_se = self.ridge, len(y_se)

        def score(idx: np.ndarray) -> np.ndarray:
            A = gram[idx[:, :, None], idx[:, None, :]] + ridge * np.eye(idx.shape[1])
            w = _batch_solve(A, rhs[idx])
            sse = (yy_se - 2 * np.sum(w * rhs_se[idx], axis=1)
                   + np.einsum("bi,bij,bj->b", w, gram_se[idx[:, :, None], idx[:, None, :]], w))
            return np.sqrt(np.maximum(sse, 0.0) / n_se)

        score.rows = 1
        return score

    def _press_scorer(self, P, y):
        gram, rhs, ridge, n = P.T @ P, P.T @ y, self.ridge, len(y)

        def score(idx: np.ndarray) -> np.ndarray:
            A = gram[idx[:, :, None], idx[:, None, :]] + ridge * np.eye(idx.shape[1])
            A_inv = np.linalg.pinv(A)
            w = np.einsum("bij,bj->bi", A_inv, rhs[idx])
            X = np.transpose(P[:, idx], (1, 0, 2))
            residual = y - np.einsum("bnm,bm->bn", X, w)
            leverage = np.einsum("bnm,bmk,bnk->bn", X, A_inv, X)
            loo = residual / np.maximum(1.0 - leverage, 1e-12)
            return np.sqrt(np.mean(loo ** 2, axis=1))

        score.rows = n
        return score

    def _logistic_scorer(self, P_tr, y_tr, P_se, y_se):
        ridge = self.ridge

        def score(idx: np.ndarray) -> np.ndarray:
            X = np.transpose(P_tr[:, idx], (1, 0, 2))
            Xt = np.transpose(X, (0, 2, 1))
            m = idx.shape[1]
            w = np.zeros((len(idx), m))
            active = np.ones(len(idx), dtype=bool)
            for _ in range(_NEWTON_ITERATIONS):
                Xa, Xta, wa = X[active], Xt[active], w[active]
                p = expit(np.clip((Xa @ wa[:, :, None])[:, :, 0], -_LOGIT_CLIP, _LOGIT_CLIP))
                grad = (Xta @ (p - y_tr)[:, :, None])[:, :, 0] + ridge * wa
                hess = Xta @ (Xa * (p * (1 - p))[:, :, None]) + ridge * np.eye(m)
                step = np.clip(_batch_solve(hess, grad), -10.0, 10.0)
                w[active] = wa - step
                converged = np.max(np.abs(step), axis=1) < 1e-7
                active[np.flatnonzero(active)[converged]] = False
                if not active.any():
                    break
            z = np.clip((np.transpose(P_se[:, idx], (1, 0, 2)) @ w[:, :, None])[:, :, 0],
                        -_LOGIT_CLIP, _LOGIT_CLIP)
            p = np.clip(expit(z), _PROBA_EPS, 1 - _PROBA_EPS)
            return -np.mean(y_se * np.log(p) + (1 - y_se) * np.log(1 - p), axis=1)

        score.rows = len(y_tr)
        return score

    # -- prediction and equations ---------------------------------------------

    def _forward(self, Z: np.ndarray, linear: bool = False) -> np.ndarray:
        z = _design(Z, self.terms_) @ self.coef_
        if self._link == LOGISTIC and not linear:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -_LOGIT_CLIP, _LOGIT_CLIP)))
        return z

    def _linear_expression(self, inputs: List[sp.Expr]) -> sp.Expr:
        monomials = [1] + [sp.Mul(*[inputs[k] for k in term]) for term in self.terms_]
        return sp.Add(*[sp.Float(float(w)) * m for w, m in zip(self.coef_, monomials)])

    def _output_expression(self, inputs: List[sp.Expr], linear: bool = False) -> sp.Expr:
        expr = self._linear_expression(inputs)
        if self._link == LOGISTIC and not linear:
            return _sigmoid_expression(expr)
        return expr

    def _depth(self) -> int:
        return 1

    def _summary_body(self, names: List[str], precision: Optional[int]) -> List[str]:
        def monomial(term):
            if len(term) == 2 and term[0] == term[1]:
                return f"{names[term[0]]}**2"
            return "*".join(names[k] for k in term)

        def label(terms):
            return " + ".join(monomial(term) for term in terms) or "(intercept)"

        best = min(self.levels_, key=lambda level: level["score"])
        lines = [f"Best model per complexity level (reference: {self.reference}):"]
        for level in self.levels_:
            marker = "*" if level is best else " "
            lines.append(f" {marker} {level['n_terms']:2d} terms  {self._criterion}="
                         f"{level['score']:.6g}  [{label(level['terms'])}]")
        formula = _format_expr(self._linear_expression([sp.Symbol(n) for n in names]), precision)
        if self._link == LOGISTIC:
            formula = f"sigmoid({formula})"
        lines += ["", f"Selected model: output = {formula}"]
        return lines


def _take(iterator, n: int) -> list:
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == n:
            break
    return chunk


class CombiGMDH(_CombinatorialSearch, GMDHRegressorBase):
    """Combinatorial (single-layer) GMDH regressor.

    The reference polynomial of all inputs is expanded into its terms, and
    every model made of the intercept plus a subset of ``k`` terms is fitted
    by ridge regression on the training split, for ``k = 0, 1, 2, ...``
    (progressive complication). Each model is scored by an external
    criterion and the best model over all complexity levels is selected.
    Unlike the multilayer GMDH, the result is a single explicit polynomial.

    The number of candidate models grows as ``2 ** n_terms``; the search is
    limited to ``max_terms`` terms and to at most ``max_models`` models.

    Parameters
    ----------
    reference : {"linear", "interaction", "quadratic"}, default="quadratic"
        Reference polynomial whose terms are searched: linear terms only;
        linear terms and pairwise products; or linear terms, pairwise
        products and squares.
    max_terms : int, optional
        Maximum number of terms (besides the intercept) in a model. By
        default all complexity levels are searched that fit within
        ``max_models``.
    criterion : {"regularity", "press"}, default="regularity"
        External criterion. ``"regularity"`` is the RMSE on the selection
        split of models fitted on the training split. ``"press"`` is the
        leave-one-out RMSE (predicted residual sum of squares) computed on
        all data, which needs no split.
    ridge : float, default=1e-6
        L2 regularization used when fitting each model.
    training_split : float, default=0.5
        Fraction of samples used to fit models under the regularity
        criterion; the remainder forms the selection set.
    patience : int, optional
        Stop after this many consecutive complexity levels without
        improvement. By default all levels are searched, since the criterion
        can have local minima.
    threshold : float, optional
        Stop as soon as the best criterion value (on the standardized target)
        falls to or below this value.
    refit : bool, default=True
        Refit the coefficients of the selected model on all data.
    max_models : int, default=1_000_000
        Maximum number of candidate models evaluated.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    terms_ : list of tuple of int
        Terms of the selected model; each is a tuple of feature indices
        whose standardized values are multiplied.
    coef_ : ndarray of shape (len(terms_) + 1,)
        Coefficients of the intercept and of ``terms_`` on the standardized
        scale.
    levels_ : list of dict
        Best model (``n_terms``, ``score``, ``terms``) at each complexity
        level.
    best_score_ : float
        Criterion value of the selected model on the standardized target.
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
    GMDH : Multilayer GMDH, which grows pairwise neurons layer by layer.
    """

    def __init__(self, reference: str = "quadratic", max_terms: Optional[int] = None,
                 criterion: str = "regularity", ridge: float = 1e-6,
                 training_split: float = 0.5, patience: Optional[int] = None,
                 threshold: Optional[float] = None, refit: bool = True,
                 max_models: int = 1_000_000, random_state: Optional[int] = None) -> None:
        self.reference = reference
        self.max_terms = max_terms
        self.criterion = criterion
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.refit = refit
        self.max_models = max_models
        self.random_state = random_state


class CombiGMDHClassifier(_CombinatorialSearch, GMDHClassifierBase):
    """Combinatorial (single-layer) GMDH binary classifier.

    Every logistic model made of the intercept plus a subset of ``k`` terms
    of the reference polynomial is fitted on the training split, for
    ``k = 0, 1, 2, ...``, and ranked by its log loss on the selection split.
    The best model over all complexity levels is selected, so the result is
    a single logistic regression on an explicit polynomial.

    Parameters
    ----------
    reference : {"linear", "interaction", "quadratic"}, default="quadratic"
        Reference polynomial whose terms are searched.
    max_terms : int, optional
        Maximum number of terms (besides the intercept) in a model. By
        default all complexity levels are searched that fit within
        ``max_models``.
    ridge : float, default=1e-6
        L2 penalty of each logistic model (the inverse of scikit-learn's
        ``C``).
    training_split : float, default=0.5
        Fraction of samples used to fit models; the remainder forms the
        selection set used to rank them.
    patience : int, optional
        Stop after this many consecutive complexity levels without
        improvement. By default all levels are searched.
    threshold : float, optional
        Stop as soon as the best selection log loss falls to or below this
        value.
    refit : bool, default=True
        Refit the coefficients of the selected model on all data.
    max_models : int, default=20_000
        Maximum number of candidate models evaluated.
    calibrate : bool, default=True
        Recalibrate the output probability with Platt scaling fitted on
        out-of-fold predictions. This refits the model ``calibration_cv``
        additional times.
    calibration_cv : int, default=3
        Number of stratified folds used to fit the calibration.
    random_state : int, optional
        Seed controlling the train/selection split.

    Attributes
    ----------
    terms_ : list of tuple of int
        Terms of the selected model.
    coef_ : ndarray of shape (len(terms_) + 1,)
        Logistic coefficients of the intercept and of ``terms_`` on the
        standardized scale.
    levels_ : list of dict
        Best model (``n_terms``, ``score``, ``terms``) at each complexity
        level.
    best_score_ : float
        Selection log loss of the selected model.
    classes_ : ndarray of shape (2,)
        Class labels; ``classes_[1]`` is the positive class.
    calibration_slope_, calibration_intercept_ : float
        Platt scaling parameters ``a`` and ``b`` of ``sigmoid(a * z + b)``,
        where ``z`` is the model's logit; ``1`` and ``0`` when
        ``calibrate=False``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during :meth:`fit`, when available.
    x_mean_, x_scale_ : ndarray of shape (n_features_in_,)
        Per-feature standardization statistics.
    """

    def __init__(self, reference: str = "quadratic", max_terms: Optional[int] = None,
                 ridge: float = 1e-6, training_split: float = 0.5,
                 patience: Optional[int] = None, threshold: Optional[float] = None,
                 refit: bool = True, max_models: int = 20_000,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.reference = reference
        self.max_terms = max_terms
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.refit = refit
        self.max_models = max_models
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
