"""Suite trees: leaves, weights, aggregation and stderr propagation (spec 2.6)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

EQUAL_WEIGHT_AGGREGATIONS = {"average", "average_of_averages", "none", "display_only"}
NO_SCORE_AGGREGATIONS = {"none", "display_only"}


@dataclass(frozen=True)
class SuiteNode:
    name: str
    type: Literal["task", "suite"]
    aggregation: str | None
    children: tuple[SuiteNode, ...]


def build_tree(root: str, defs: Mapping[str, tuple[str, list[dict]]]) -> SuiteNode:
    """Build the tree under ``root``.

    ``defs`` maps suite name -> (aggregation, children[{type, name}]). Unknown child suites
    become empty suites. Cycles raise ValueError.
    """

    def build(name: str, stack: tuple[str, ...]) -> SuiteNode:
        if name in stack:
            raise ValueError(f"suite cycle: {' -> '.join((*stack, name))}")
        if name not in defs:
            return SuiteNode(name=name, type="suite", aggregation=None, children=())
        aggregation, children = defs[name]
        nodes = []
        for child in children:
            if child.get("type") == "suite":
                nodes.append(build(child["name"], (*stack, name)))
            else:
                nodes.append(
                    SuiteNode(name=child["name"], type="task", aggregation=None, children=())
                )
        return SuiteNode(name=name, type="suite", aggregation=aggregation, children=tuple(nodes))

    return build(root, ())


def leaves(node: SuiteNode) -> list[str]:
    """Task names in pre-order, deduplicated."""
    out: list[str] = []
    seen: set[str] = set()

    def walk(n: SuiteNode) -> None:
        if n.type == "task":
            if n.name not in seen:
                seen.add(n.name)
                out.append(n.name)
            return
        for child in n.children:
            walk(child)

    walk(node)
    return out


def leaf_weights(
    node: SuiteNode, n_instances: Mapping[str, int], present: set[str]
) -> dict[str, float]:
    """Weights over present leaves summing to 1 (empty when no leaf is present).

    average / average_of_averages / none / display_only: equal weight among children that have
    at least one present leaf, recursively. weighted_average: proportional to n_instances over
    the node's present leaves.
    """
    if node.type == "task":
        return {node.name: 1.0} if node.name in present else {}
    if node.aggregation == "weighted_average":
        names = [name for name in leaves(node) if name in present]
        total = sum(max(n_instances.get(name, 0), 0) for name in names)
        if not names:
            return {}
        if total <= 0:
            return {name: 1.0 / len(names) for name in names}
        return {name: max(n_instances.get(name, 0), 0) / total for name in names}
    child_weights = [leaf_weights(child, n_instances, present) for child in node.children]
    child_weights = [w for w in child_weights if w]
    if not child_weights:
        return {}
    out: dict[str, float] = {}
    share = 1.0 / len(child_weights)
    for weights in child_weights:
        for name, w in weights.items():
            out[name] = out.get(name, 0.0) + share * w
    return out


def aggregate(
    node: SuiteNode, scores: Mapping[str, float | None], n_instances: Mapping[str, int]
) -> float | None:
    if node.type == "suite" and node.aggregation in NO_SCORE_AGGREGATIONS:
        return None
    present = {name for name, score in scores.items() if score is not None}
    weights = leaf_weights(node, n_instances, present)
    if not weights:
        return None
    total = 0.0
    for name, w in weights.items():
        value = scores[name]
        assert value is not None
        total += w * value
    return total


def propagate_stderr(
    weights: Mapping[str, float], stderrs: Mapping[str, float | None]
) -> float | None:
    if not weights:
        return None
    total = 0.0
    for name, w in weights.items():
        se = stderrs.get(name)
        if se is None:
            return None
        total += w * w * se * se
    return math.sqrt(total)


def walk_suites(node: SuiteNode, depth: int = 0, parent: str | None = None):
    """Yield (node, depth, parent) for suite nodes in pre-order."""
    if node.type != "suite":
        return
    yield node, depth, parent
    for child in node.children:
        yield from walk_suites(child, depth + 1, node.name)
