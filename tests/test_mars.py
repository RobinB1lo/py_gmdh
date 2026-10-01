import numpy as np
import pytest
import sympy as sp
from sklearn.base import clone

from py_gmdh.mars import MARSGMDH, evaluate_basis, fit_mars


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(240, 4)) * [1.0, 10.0, 0.1, 5.0] + [0.0, 50.0, 3.0, -1.0]
    y = np.sin(X[:, 0]) * X[:, 1] + 10 * X[:, 2] ** 2 + 0.5 * X[:, 3]
    X_new = rng.uniform(-2.5, 2.5, size=(40, 4)) * [1.0, 10.0, 0.1, 5.0] + [0.0, 50.0, 3.0, -1.0]
    return X, y, X_new


def test_fit_mars_recovers_hinge_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(400, 2))
    y = 3 * np.maximum(0, X[:, 0] - 0.5) - 2 * np.maximum(0, 0.2 - X[:, 1])
    basis = fit_mars(X, y, max_degree=1)
    B = evaluate_basis(basis, X)
    residual = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
    assert np.sqrt(np.mean(residual ** 2)) < 0.05


def test_equation_reproduces_predictions(data):
    X, y, X_new = data
    model = MARSGMDH(n_keep=4, max_layers=3, patience=2, random_state=0).fit(X, y)
    expr = model.equation(precision=None, as_sympy=True)
    f = sp.lambdify(sp.symbols("x0:4"), expr, "numpy")
    np.testing.assert_allclose(f(*X_new.T), model.predict(X_new), rtol=1e-9)
    assert isinstance(model.summary(), str)


def test_reproducible_and_clonable(data):
    X, y, X_new = data
    model = MARSGMDH(n_keep=3, max_layers=2, random_state=1)
    first = model.fit(X, y).predict(X_new)
    np.testing.assert_array_equal(first, clone(model).fit(X, y).predict(X_new))
