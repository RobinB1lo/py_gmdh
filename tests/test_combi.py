import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from py_gmdh.combi import CombiGMDH, CombiGMDHClassifier, reference_terms


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = 2 * X[:, 0] * X[:, 1] - X[:, 2] ** 2 + rng.normal(0, 0.1, 300)
    return X, y


def test_reference_terms():
    assert reference_terms(3, "linear") == [(0,), (1,), (2,)]
    assert len(reference_terms(3, "interaction")) == 6
    assert len(reference_terms(3, "quadratic")) == 9


@pytest.mark.parametrize("criterion", ["regularity", "press"])
def test_recovers_polynomial_structure(data, criterion):
    X, y = data
    model = CombiGMDH(criterion=criterion, random_state=0).fit(X, y)
    assert {(0, 1), (2, 2)} <= set(model.terms_)
    assert model.score(X, y) > 0.99
    assert [level["n_terms"] for level in model.levels_] == list(range(10))


def test_max_models_limits_search(data):
    X, y = data
    with pytest.warns(UserWarning, match="limited to models with at most 2 terms"):
        model = CombiGMDH(max_models=46, random_state=0).fit(X, y)
    assert len(model.levels_) == 3


def test_classifier_calibration_toggle(data):
    X, y = data
    labels = (y > np.median(y)).astype(int)
    raw = CombiGMDHClassifier(max_terms=3, calibrate=False, random_state=0).fit(X, labels)
    assert (raw.calibration_slope_, raw.calibration_intercept_) == (1.0, 0.0)
    calibrated = CombiGMDHClassifier(max_terms=3, random_state=0).fit(X, labels)
    assert calibrated.terms_ == raw.terms_
    assert roc_auc_score(labels, calibrated.predict_proba(X)[:, 1]) == pytest.approx(
        roc_auc_score(labels, raw.predict_proba(X)[:, 1]), abs=1e-3)
