import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from py_gmdh._logistic import ConvergenceWarning, logistic_fit, sigmoid


def _gradient(X, t, w, ridge):
    return X.T @ (sigmoid(X @ w) - t) + ridge * w


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(300), rng.normal(size=(300, 4))])
    t = (rng.random(300) < sigmoid(X @ [0.3, 1.0, -2.0, 0.5, 0.0])).astype(float)
    return X, t


@pytest.mark.parametrize("ridge", [1e-6, 1e-2, 1.0])
def test_matches_scikit_learn(data, ridge):
    X, t = data
    w = logistic_fit(X, t, ridge)
    reference = LogisticRegression(C=1.0 / ridge, fit_intercept=False, tol=1e-10,
                                   max_iter=10_000).fit(X, t).coef_[0]
    np.testing.assert_allclose(w, reference, rtol=1e-5, atol=1e-7)


def test_solution_is_stationary(data):
    X, t = data
    w = logistic_fit(X, t, 1e-6)
    assert np.max(np.abs(_gradient(X, t, w, 1e-6))) <= 1e-8 * len(t)


def test_separable_data_converge_to_finite_solution():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(100), rng.normal(size=(100, 2))])
    t = (X[:, 1] > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        w = logistic_fit(X, t, 1e-6)
    assert np.all(np.isfinite(w))
    assert np.all((X @ w > 0) == (t == 1))


def test_badly_scaled_columns_converge():
    rng = np.random.default_rng(2)
    a = rng.normal(size=120)
    X = np.column_stack([np.ones(120), a, np.abs(a) ** -2.0])
    t = (rng.random(120) < sigmoid(a)).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        w = logistic_fit(X, t, 1e-6)
    reference = LogisticRegression(C=1e6, fit_intercept=False, max_iter=10_000).fit(X, t)
    objective = lambda v: np.sum(np.logaddexp(0, X @ v) - t * (X @ v)) + 5e-7 * v @ v
    assert objective(w) <= objective(reference.coef_[0]) + 1e-8


def test_soft_targets():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=200), np.ones(200)])
    t = sigmoid(1.5 * X[:, 0] - 0.5)
    np.testing.assert_allclose(logistic_fit(X, t, 0.0), [1.5, -0.5], atol=1e-6)


def test_sigmoid_is_stable():
    z = np.array([-1000.0, -30.0, 0.0, 30.0, 1000.0])
    p = sigmoid(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, [0.0, 1 / (1 + np.exp(30)), 0.5, 1 / (1 + np.exp(-30)), 1.0])


def test_sympy_is_imported_only_for_equations():
    code = (
        "import sys, numpy as np, py_gmdh\n"
        "assert 'sympy' not in sys.modules\n"
        "X = np.random.default_rng(0).normal(size=(80, 3)); y = X[:, 0] - X[:, 1]\n"
        "m = py_gmdh.GMDH(random_state=0).fit(X, y); m.predict(X)\n"
        "c = py_gmdh.GMDHClassifier(random_state=0).fit(X, y > 0); c.predict_proba(X)\n"
        "assert 'sympy' not in sys.modules\n"
        "m.equation()\n"
        "assert 'sympy' in sys.modules\n"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    paths = [src, os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
