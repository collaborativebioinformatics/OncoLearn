"""
Stack-based executor for the pipeline DSL.
"""
from typing import List, Union

import pandas as pd

from .nodes import Join, Load, Sequence
from .readers.base import BaseReader


def _flatten(node: Union[Load, Join, Sequence]) -> List[Union[Load, Join]]:
    """Recursively flatten a pipeline tree into a flat list of ops."""
    if isinstance(node, (Load, Join)):
        return [node]
    if isinstance(node, Sequence):
        result: List[Union[Load, Join]] = []
        for step in node.steps:
            result.extend(_flatten(step))
        return result
    raise ValueError(f"Unknown pipeline node type: {type(node)}")


def run(pipeline_node: Union[Load, Sequence], reader: BaseReader) -> pd.DataFrame:
    """Execute a pipeline node and return the resulting DataFrame.

    Flattens the node tree into a sequence of :class:`Load` and :class:`Join`
    operations, then executes them using a stack machine:

    - :class:`Load` → pushes one DataFrame read via *reader*.
    - :class:`Join` → pops two DataFrames (right first), merges, pushes result.

    Args:
        pipeline_node: Root node to execute (``Load`` or ``Sequence``).
        reader: :class:`BaseReader` used to load data by name.

    Returns:
        The single remaining DataFrame on the stack after all operations.

    Raises:
        RuntimeError: If the stack does not contain exactly one DataFrame when
                      execution completes.
    """
    steps = _flatten(pipeline_node)
    stack: List[pd.DataFrame] = []

    for step in steps:
        if isinstance(step, Load):
            stack.append(reader.read(step.name))
        elif isinstance(step, Join):
            if len(stack) < 2:
                raise RuntimeError(
                    f"Join requires at least 2 DataFrames on the stack, "
                    f"but only {len(stack)} are present."
                )
            right = stack.pop()
            left = stack.pop()
            merged = pd.merge(left, right, on=step.on, how=step.how, suffixes=("", "_dup"))
            # Drop duplicate columns introduced by the join (keep originals)
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            if dup_cols:
                merged = merged.drop(columns=dup_cols)
            stack.append(merged)
        else:
            raise ValueError(f"Unknown pipeline step type: {type(step)}")

    if len(stack) != 1:
        raise RuntimeError(
            f"Pipeline execution left {len(stack)} DataFrames on the stack "
            f"(expected exactly 1).  Check that every pair of Load nodes is "
            f"followed by a Join node."
        )
    return stack[0]
