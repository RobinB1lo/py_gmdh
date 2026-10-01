import numpy as np
import pytest
import sympy as sp
from sklearn.base import clone

import py_gmdh


def _make(cls, **params):
    """Instantiate ``cls`` with the subset of ``params`` it accepts."""
    accepted = cls().get_params()
    return cls(**{k: v for k, v in params.items() if k in accepted})

CLASSIFIERS = [getattr(py_gmdh, name) for name in py_gmdh.__all__ if name.endswith("Classifier")]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 4)) * [1.0, 10.0, 0.5, 5.0] + [0.0, 50.0, 3.0, -1.0]
    logit = np.sin(X[:, 0]) + (X[:, 1] - 50) / 10 * (X[:, 3] + 1) / 5 - (X[:, 2] - 3)
    y = np.where(logit + rng.logistic(size=240) > 0, "yes", "no")
    X_new = rng.normal(size=(40, 4)) * [1.0, 10.0, 0.5, 5.0] + [0.0, 50.0, 3.0, -1.0]
    return X, y, X_new


def _evaluate(expr, X):
    f = sp.lambdify(sp.symbols(f"x0:{X.shape[1]}"), expr, "numpy")
    return np.broadcast_to(np.asarray(f(*X.T), dtype=float), (len(X),))


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=lambda c: c.__name__)
def test_equation_reproduces_probabilities(cls, data):
    X, y, X_new = data
    model = _make(cls, max_terms=3, n_keep=3, max_layers=2, ridge=1e-2, random_state=0).fit(X, y)
    expr = model.equation(precision=None, as_sympy=True)
    np.testing.assert_allclose(_evaluate(expr, X_new), model.predict_proba(X_new)[:, 1],
                               rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=lambda c: c.__name__)
def test_classifier_interface(cls, data):
    X, y, X_new = data
    model = _make(cls, max_terms=3, n_keep=3, max_layers=2, ridge=1e-2, random_state=0).fit(X, y)

    proba = model.predict_proba(X_new)
    assert proba.shape == (len(X_new), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(model.predict(X_new)) <= {"no", "yes"}
    assert list(model.classes_) == ["no", "yes"]
    assert model.equation().startswith("P(y = yes) = ")
    assert isinstance(model.summary(), str)

    again = clone(model).fit(X, y).predict_proba(X_new)
    np.testing.assert_array_equal(proba, again)


def test_rejects_multiclass():
    X = np.random.default_rng(0).normal(size=(60, 3))
    with pytest.raises(ValueError, match="Only binary classification"):
        py_gmdh.GMDHClassifier().fit(X, np.arange(60) % 3)
