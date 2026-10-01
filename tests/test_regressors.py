import numpy as np
import pytest
import sympy as sp
from sklearn.base import clone

import py_gmdh


def _make(cls, **params):
    """Instantiate ``cls`` with the subset of ``params`` it accepts."""
    accepted = cls().get_params()
    return cls(**{k: v for k, v in params.items() if k in accepted})

REGRESSORS = [getattr(py_gmdh, name) for name in py_gmdh.__all__
              if name.endswith("GMDH")]


def _raw(result):
    """Equation in the original features from ``equation(as_sympy=True)``."""
    if isinstance(result, tuple):
        expr, definitions = result
        return expr.subs(definitions)
    return result


def _evaluate(expr, X):
    f = sp.lambdify(sp.symbols(f"x0:{X.shape[1]}"), expr, "numpy")
    return np.broadcast_to(np.asarray(f(*X.T), dtype=float), (len(X),))


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(240, 4)) * [1.0, 10.0, 0.1, 5.0] + [0.0, 50.0, 3.0, -1.0]
    y = np.sin(X[:, 0]) * X[:, 1] + 10 * X[:, 2] ** 2 + 0.5 * X[:, 3]
    X_new = rng.uniform(-2, 2, size=(40, 4)) * [1.0, 10.0, 0.1, 5.0] + [0.0, 50.0, 3.0, -1.0]
    return X, y, X_new


@pytest.mark.parametrize("variables", ["standardized", "raw"])
@pytest.mark.parametrize("cls", REGRESSORS, ids=lambda c: c.__name__)
def test_equation_reproduces_predictions(cls, variables, data):
    X, y, X_new = data
    model = _make(cls, max_terms=3, n_keep=4, max_layers=2, random_state=0).fit(X, y)
    expr = _raw(model.equation(expand=False, as_sympy=True, variables=variables))
    np.testing.assert_allclose(_evaluate(expr, X_new), model.predict(X_new), rtol=1e-9)


def test_expanded_equation_matches_predictions(data):
    X, y, X_new = data
    model = py_gmdh.GMDH(n_keep=4, max_layers=2, random_state=0).fit(X, y)
    expr, definitions = model.equation(expand=True, as_sympy=True)
    assert expr.is_polynomial()
    np.testing.assert_allclose(_evaluate(expr.subs(definitions), X_new), model.predict(X_new),
                               rtol=1e-9)


def _parse_equation(text, n_features):
    """Parse the printed equation, substituting the listed z definitions."""
    rhs, _, definitions = text.partition("\nwhere\n")
    names = {f"x{k}": sp.Symbol(f"x{k}") for k in range(n_features)}
    z = {}
    for line in definitions.splitlines():
        name, value = line.split(" = ")
        z[sp.Symbol(name.strip())] = sp.sympify(value, locals=names)
    return sp.sympify(rhs.split(" = ", 1)[1]).subs(z)


def test_rounding_is_for_display_only(data):
    X, y, X_new = data
    X, X_new = X + 1e4, X_new + 1e4  # uncentered features
    model = py_gmdh.GMDH(n_keep=4, max_layers=2, random_state=0).fit(X, y)
    expr = _parse_equation(model.equation(), 4)  # 4 digits, standardized, expanded
    error = np.abs(_evaluate(expr, X_new) - model.predict(X_new)).max() / y.std()
    assert error < 1e-2
    nested = _parse_equation(model.equation(expand=False, variables="raw"), 4)
    error = np.abs(_evaluate(nested, X_new) - model.predict(X_new)).max() / y.std()
    assert error < 1e-2


def test_raw_expansion_warns_when_ill_conditioned(data):
    X, y, _ = data
    model = py_gmdh.GMDH(n_keep=4, max_layers=2, random_state=0).fit(X + 1e4, y)
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        model.equation(variables="raw", expand=True)


def test_equation_too_long_raises(data):
    X, y, _ = data
    model = py_gmdh.GMDH(n_keep=4, max_layers=2, random_state=0).fit(X, y)
    with pytest.raises(py_gmdh.EquationTooLongError, match="too long"):
        model.equation(expand=False, max_length=10)
    assert model.equation(expand=False, max_length=None).startswith("y = ")


@pytest.mark.parametrize("precision", [0, -1, 2.5, True])
def test_invalid_precision(precision, data):
    X, y, _ = data
    model = py_gmdh.GMDH(n_keep=3, max_layers=1, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="precision"):
        model.equation(precision=precision)


@pytest.mark.parametrize("cls", REGRESSORS, ids=lambda c: c.__name__)
def test_equation_and_summary_are_strings(cls, data):
    X, y, _ = data
    model = _make(cls, max_terms=3, n_keep=3, max_layers=2, random_state=0).fit(X, y)
    assert model.equation(feature_names=["a", "b", "c", "d"]).startswith("y = ")
    assert isinstance(model.summary(), str)


@pytest.mark.parametrize("cls", REGRESSORS, ids=lambda c: c.__name__)
def test_reproducible_and_clonable(cls, data):
    X, y, X_new = data
    model = _make(cls, max_terms=3, n_keep=3, max_layers=2, random_state=1)
    first = model.fit(X, y).predict(X_new)
    second = clone(model).fit(X, y).predict(X_new)
    np.testing.assert_array_equal(first, second)


def test_feature_names_from_dataframe():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["temp", "wind", "rh"])
    y = X["temp"] * X["rh"] + X["wind"]
    model = py_gmdh.GMDH(random_state=0).fit(X, y)
    expr, definitions = model.equation(as_sympy=True)
    assert {s.name for s in expr.free_symbols} <= {"z_temp", "z_wind", "z_rh"}
    assert {s.name for s in expr.subs(definitions).free_symbols} <= {"temp", "wind", "rh"}
    raw = model.equation(as_sympy=True, variables="raw")
    assert {s.name for s in raw.free_symbols} <= {"temp", "wind", "rh"}


def test_rejects_single_feature():
    with pytest.raises(ValueError, match="at least 2 features"):
        py_gmdh.GMDH().fit(np.ones((10, 1)), np.ones(10))


def test_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        py_gmdh.GMDH().predict(np.ones((3, 2)))
