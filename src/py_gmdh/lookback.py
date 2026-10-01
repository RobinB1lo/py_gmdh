"""GMDH whose neurons may connect to nodes from any earlier layer."""

from __future__ import annotations

from itertools import combinations, count
from typing import Dict, List, Optional

import numpy as np

from ._base import GMDHClassifierBase, GMDHRegressorBase, Neuron
from ._lazy import sp

_SCOPES = ("all", "inputs")


class InputNode:
    """Leaf of the network graph holding one standardized input feature."""

    __slots__ = ("id", "index", "ancestors", "depth")

    def __init__(self, node_id: int, index: int) -> None:
        self.id = node_id
        self.index = index
        self.ancestors: frozenset = frozenset()
        self.depth = 0


class NeuronNode:
    """Neuron combining two earlier nodes of the network graph.

    ``neuron.i`` and ``neuron.j`` hold the ids of the two parent nodes.
    """

    __slots__ = ("id", "neuron", "ancestors", "depth")

    def __init__(self, node_id: int, neuron: Neuron, parent_a, parent_b) -> None:
        self.id = node_id
        self.neuron = neuron
        self.ancestors = parent_a.ancestors | parent_b.ancestors | {parent_a.id, parent_b.id}
        self.depth = 1 + max(parent_a.depth, parent_b.depth)


class _LookbackNetwork:
    """Network search in which frontier neurons may pair with earlier nodes.

    Each frontier node is paired with the other frontier nodes and with
    every eligible earlier node that is not one of its own ancestors. The
    fitted model is a directed acyclic graph of :class:`NeuronNode` objects.
    """

    _fitted_attr = "output_node_"

    def _check_params(self) -> None:
        super()._check_params()
        if self.lookback_scope not in _SCOPES:
            raise ValueError(f"lookback_scope must be one of {_SCOPES}, "
                             f"got {self.lookback_scope!r}.")
        if self.max_lookback_candidates is not None and self.max_lookback_candidates < 1:
            raise ValueError("max_lookback_candidates must be a positive integer or None, "
                             f"got {self.max_lookback_candidates!r}.")

    def _lookback_pool(self, nodes, frontier, rng) -> list:
        frontier_ids = {node.id for node in frontier}
        pool = [node for node in nodes if node.id not in frontier_ids]
        if self.lookback_scope == "inputs":
            pool = [node for node in pool if isinstance(node, InputNode)]
        limit = self.max_lookback_candidates
        if limit is not None and len(pool) > limit:
            pool = [pool[k] for k in sorted(rng.choice(len(pool), limit, replace=False))]
        return pool

    def _fit_network(self, Z_tr, Z_se, y_tr, y_se, rng) -> None:
        ids = count()
        inputs = [InputNode(next(ids), k) for k in range(Z_tr.shape[1])]
        values_tr = {node.id: Z_tr[:, node.index] for node in inputs}
        values_se = {node.id: Z_se[:, node.index] for node in inputs}
        nodes: list = list(inputs)
        frontier: list = list(inputs)

        best_score = np.inf
        best_node = None
        stalled = 0

        for _ in range(self.max_layers):
            pairs = list(combinations(frontier, 2))
            pool = self._lookback_pool(nodes, frontier, rng)
            pairs += [(f, p) for f in frontier for p in pool if p.id not in f.ancestors]
            if not pairs:
                break

            candidates = []
            for a, b in pairs:
                neuron = self._create_neuron(a.id, b.id).fit(
                    values_tr[a.id], values_tr[b.id], y_tr, self.ridge)
                error = neuron.loss(y_se, neuron.predict(values_se[a.id], values_se[b.id]))
                candidates.append((error, a, b, neuron))
            candidates.sort(key=lambda c: c[0])

            frontier = []
            for _, a, b, neuron in candidates[:self.n_keep]:
                node = NeuronNode(next(ids), neuron, a, b)
                values_tr[node.id] = neuron.predict(values_tr[a.id], values_tr[b.id])
                values_se[node.id] = neuron.predict(values_se[a.id], values_se[b.id])
                nodes.append(node)
                frontier.append(node)

            layer_score = candidates[0][0]
            if layer_score < best_score - self._improvement_tol:
                best_score, best_node, stalled = layer_score, frontier[0], 0
            else:
                stalled += 1

            if self.threshold is not None and best_score <= self.threshold:
                break
            if stalled > self.patience:
                break

        if best_node is None:
            raise RuntimeError("No neuron produced a finite selection error.")
        self.output_node_ = best_node
        self.nodes_ = [node for node in nodes
                       if node.id in best_node.ancestors or node.id == best_node.id]
        self.best_score_ = best_score

    def _forward(self, Z: np.ndarray, linear: bool = False) -> np.ndarray:
        values: Dict[int, np.ndarray] = {}
        for node in self.nodes_:
            if isinstance(node, InputNode):
                values[node.id] = Z[:, node.index]
            else:
                n = node.neuron
                a, b = values[n.i], values[n.j]
                last = node is self.output_node_
                values[node.id] = n.linear_predict(a, b) if linear and last else n.predict(a, b)
        return values[self.output_node_.id]

    def _output_expression(self, inputs: List[sp.Expr], linear: bool = False) -> sp.Expr:
        exprs: Dict[int, sp.Expr] = {}
        for node in self.nodes_:
            if isinstance(node, InputNode):
                exprs[node.id] = inputs[node.index]
            else:
                n = node.neuron
                a, b = exprs[n.i], exprs[n.j]
                last = node is self.output_node_
                exprs[node.id] = (n._linear_expression(a, b) if linear and last
                                  else n.expression(a, b))
        return exprs[self.output_node_.id]

    def _depth(self) -> int:
        return self.output_node_.depth

    def _summary_body(self, names: List[str], precision: Optional[int]) -> List[str]:
        neurons = [node for node in self.nodes_ if isinstance(node, NeuronNode)]
        labels: Dict[int, str] = {node.id: names[node.index]
                                  for node in self.nodes_ if isinstance(node, InputNode)}
        lines = [f"Graph ({len(neurons)} neurons, depth {self._depth()}, "
                 f"look-back scope: {self.lookback_scope})"]
        for k, node in enumerate(neurons):
            labels[node.id] = f"n{k}"
            n = node.neuron
            lines.append(f"  n{k} = {n.describe(labels[n.i], labels[n.j], precision)}")
        lines.append(f"Output: {labels[self.output_node_.id]}")
        return lines


class LookbackGMDH(_LookbackNetwork, GMDHRegressorBase):
    """GMDH regressor with look-back connections.

    In classical GMDH a neuron can only combine outputs of the immediately
    preceding layer. Here, each neuron of the current frontier may also be
    paired with any earlier node - a raw input or a neuron from an older
    layer - provided that node is not one of its own ancestors. The fitted
    model is therefore a directed acyclic graph rather than a strict stack of
    layers. Neurons use the quadratic Ivakhnenko polynomial and are ranked by
    selection RMSE.

    Parameters
    ----------
    n_keep : int, default=10
        Number of neurons kept in each layer; they form the next frontier.
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
    lookback_scope : {"all", "inputs"}, default="all"
        Nodes the frontier may look back to: every earlier node, or only the
        raw inputs.
    max_lookback_candidates : int, optional
        If set, the look-back pool is randomly subsampled to at most this
        many nodes per layer to bound the number of candidates.
    random_state : int, optional
        Seed controlling the train/selection split and pool subsampling.

    Attributes
    ----------
    output_node_ : NeuronNode
        Node whose output is the model prediction.
    nodes_ : list of InputNode or NeuronNode
        Nodes the output depends on, in topological order.
    best_score_ : float
        Selection RMSE of the output node on the standardized target.
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
                 threshold: Optional[float] = None, lookback_scope: str = "all",
                 max_lookback_candidates: Optional[int] = None,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.lookback_scope = lookback_scope
        self.max_lookback_candidates = max_lookback_candidates
        self.random_state = random_state


class LookbackGMDHClassifier(_LookbackNetwork, GMDHClassifierBase):
    """GMDH binary classifier with look-back connections.

    Each neuron of the current frontier may be paired with another frontier
    neuron or with any earlier node - a raw input or a neuron from an older
    layer - provided that node is not one of its own ancestors. Neurons are
    logistic models of the quadratic Ivakhnenko polynomial and are ranked
    by selection log loss.

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
    lookback_scope : {"all", "inputs"}, default="all"
        Nodes the frontier may look back to: every earlier node, or only the
        raw inputs.
    max_lookback_candidates : int, optional
        If set, the look-back pool is randomly subsampled to at most this
        many nodes per layer to bound the number of candidates.
    calibrate : bool, default=True
        Recalibrate the output probability with Platt scaling fitted on
        out-of-fold predictions. This refits the network ``calibration_cv``
        additional times.
    calibration_cv : int, default=3
        Number of stratified folds used to fit the calibration.
    random_state : int, optional
        Seed controlling the train/selection split and pool subsampling.

    Attributes
    ----------
    output_node_ : NeuronNode
        Node whose output is the predicted probability of ``classes_[1]``.
    nodes_ : list of InputNode or NeuronNode
        Nodes the output depends on, in topological order.
    best_score_ : float
        Selection log loss of the output node.
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
                 lookback_scope: str = "all",
                 max_lookback_candidates: Optional[int] = None,
                 calibrate: bool = True, calibration_cv: int = 3,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.threshold = threshold
        self.lookback_scope = lookback_scope
        self.max_lookback_candidates = max_lookback_candidates
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state
