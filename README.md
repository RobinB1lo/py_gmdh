# py_gmdh

Group Method of Data Handling (GMDH) estimators for Python with a
scikit-learn compatible API.

GMDH builds a self-organizing network layer by layer. Each layer fits a small
two-input model (a *neuron*) for every pair of its inputs, ranks the
candidates on held-out data, and passes the best ones on to the next layer.
Growth stops once the external criterion stops improving, so the depth and
structure of the model are chosen from the data. The fitted network can be
written out as a closed-form equation.

## Installation

```bash
pip install py_gmdh
```

## Quick start

```python
import numpy as np
from py_gmdh.gmdh import GMDH

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(500, 3))
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2

model = GMDH(n_keep=6, max_layers=2, random_state=0).fit(X, y)
model.predict(X[:5])
model.score(X, y)

print(model.equation(feature_names=["a", "b", "c"]))
print(model.summary())
```

`equation()` returns the whole network as a single formula in the original,
unscaled features. `summary()` lists every neuron layer by layer. Every
estimator works with `Pipeline`, `GridSearchCV`, `clone` and the rest of
scikit-learn.

### Classification

Every variant has a binary classifier counterpart whose neurons are logistic
models ranked by selection log loss. By default the output probabilities are
recalibrated with Platt scaling fitted on out-of-fold predictions
(`calibrate=True`, `calibration_cv=3`); pass `calibrate=False` to skip it.

```python
from py_gmdh.gmdh import GMDHClassifier

labels = (X[:, 0] * X[:, 1] + X[:, 2] > 0.3).astype(int)
clf = GMDHClassifier(n_keep=6, max_layers=2, random_state=0).fit(X, labels)
clf.predict_proba(X[:5])          # shape (5, 2)
print(clf.equation(feature_names=["a", "b", "c"]))   # "P(y = 1) = ..."
```

## Estimators

| Regressor | Classifier | Module | Neuron / selection |
| --- | --- | --- | --- |
| `GMDH` | `GMDHClassifier` | `py_gmdh.gmdh` | Quadratic Ivakhnenko polynomial; selection RMSE / log loss |
| `CombiGMDH` | `CombiGMDHClassifier` | `py_gmdh.combi` | Combinatorial (single-layer) GMDH: exhaustive search over term subsets of a reference polynomial |
| `HierarchicalGMDH` | `HierarchicalGMDHClassifier` | `py_gmdh.hierarchical` | Polynomial; RMSE (or log loss) with ties broken by secondary metrics |
| `AICGMDH` | `AICGMDHClassifier` | `py_gmdh.aic` | Polynomial; Akaike information criterion |
| `CVGMDH` | `CVGMDHClassifier` | `py_gmdh.cv` | Polynomial; k-fold cross-validated loss |
| `UFPGMDH` | `UFPGMDHClassifier` | `py_gmdh.ufp` | Unconstrained fractional polynomial with learned powers |
| `CFPGMDH` | `CFPGMDHClassifier` | `py_gmdh.cfp` | Constrained fractional polynomial over a fixed set of powers |
| `FourierGMDH` | `FourierGMDHClassifier` | `py_gmdh.fourier` | Sine/cosine basis |
| `RadialGMDH` | `RadialGMDHClassifier` | `py_gmdh.radial` | Gaussian radial basis functions |
| `SigmoidGMDH` | `SigmoidGMDHClassifier` | `py_gmdh.sigmoid` | Logistic sigmoid basis |
| `MARSGMDH` | `MARSGMDHClassifier` | `py_gmdh.mars` | MARS hinge functions selected per neuron and pruned by GCV |
| `LookbackGMDH` | `LookbackGMDHClassifier` | `py_gmdh.lookback` | Polynomial; neurons may connect to any earlier layer |

All estimators can also be imported from the top-level package, for example
`from py_gmdh import FourierGMDH`.

### Combinatorial GMDH

`CombiGMDH` is Ivakhnenko's single-layer combinatorial algorithm (COMBI).
The reference polynomial (`reference="linear"`, `"interaction"` or
`"quadratic"`) is expanded into its terms, every model built from a subset of
them is fitted, with the number of terms increasing from 0, and the model with
the best external criterion is kept: the selection-set RMSE
(`criterion="regularity"`) or the leave-one-out PRESS (`criterion="press"`).
The number of candidate models grows as `2 ** n_terms`, so the search is
capped by `max_terms` and `max_models`; with many features, use a smaller
reference or limit `max_terms`.

```python
from py_gmdh.combi import CombiGMDH

model = CombiGMDH(reference="quadratic", random_state=0).fit(X, y)
print(model.equation(feature_names=["a", "b", "c"]))
print(model.summary())    # best model at every complexity level
```

## Common parameters

`CombiGMDH` has no layers, so it takes `reference`, `max_terms`,
`criterion`, `refit` and `max_models` instead of `n_keep` and `max_layers`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_keep` | `10` | Neurons kept per layer |
| `max_layers` | `10` | Maximum network depth |
| `ridge` | `1e-6` | L2 regularization of each neuron (inverse of `C` for classifiers) |
| `training_split` | `0.5` | Fraction of samples used to fit neurons; the rest ranks them |
| `patience` | `0` | Non-improving layers tolerated before stopping |
| `threshold` | `None` | Stop once the selection criterion reaches this value |
| `random_state` | `None` | Seed for the train/selection split |

## Equations

```python
model.equation()                  # "y = ..." with 4 significant digits
model.equation(precision=None)    # full precision, reproduces predict()
model.equation(expand=True)       # multiply out nested polynomials
model.equation(as_sympy=True)     # sympy expression for further manipulation
```

Polynomial regressors with up to three layers are expanded by default.
Classifier equations are nested sigmoids and are left unexpanded unless
`expand=True`, which multiplies out the first-layer arguments. The fully
substituted equation grows quickly with depth, so for deep networks
`summary()` is usually the more readable view.

## License

MIT
