"""Unit tests for the pipeline executor using a mock reader."""
import numpy as np
import pandas as pd
import pytest

from oncolearn.data.pipeline.executor import run, _flatten
from oncolearn.data.pipeline.nodes import DataSource, Join, Load, Log2Normalization, Sequence
from oncolearn.data.pipeline.readers.base import BaseReader


# ---------------------------------------------------------------------------
# Mock reader
# ---------------------------------------------------------------------------

class MockReader(BaseReader):
    def __init__(self, frames: dict):
        self._frames = frames

    def read(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            raise KeyError(f"MockReader: unknown dataset '{name}'")
        return self._frames[name].copy()


_DS = DataSource(config="mock", base_dir="mock")


# ---------------------------------------------------------------------------
# _flatten tests
# ---------------------------------------------------------------------------

def test_flatten_single_load():
    node = Load("a", source=_DS)
    assert _flatten(node) == [node]


def test_flatten_join():
    j = Join()
    assert _flatten(j) == [j]


def test_flatten_sequence():
    load_a = Load("a", source=_DS)
    load_b = Load("b", source=_DS)
    join = Join()
    seq = Sequence(steps=[load_a, load_b, join])
    result = _flatten(seq)
    assert result == [load_a, load_b, join]


def test_flatten_nested_sequence():
    load_a = Load("a", source=_DS)
    load_b = Load("b", source=_DS)
    join = Join()
    inner = Sequence(steps=[load_b, join])
    outer = Sequence(steps=[load_a, inner])
    result = _flatten(outer)
    assert result == [load_a, load_b, join]


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------

def test_run_single_load():
    df = pd.DataFrame({"patient_id": ["TCGA-01", "TCGA-02"], "age": [50, 60]})
    reader = MockReader({"clinical": df})
    result = run(Load("clinical", source=_DS), reader)
    assert list(result.columns) == ["patient_id", "age"]
    assert len(result) == 2


def test_run_sequence_with_join():
    df_a = pd.DataFrame({"patient_id": ["P1", "P2", "P3"], "feat_a": [1.0, 2.0, 3.0]})
    df_b = pd.DataFrame({"patient_id": ["P1", "P2"], "feat_b": [10.0, 20.0]})
    reader = MockReader({"a": df_a, "b": df_b})
    seq = Sequence(steps=[Load("a", source=_DS), Load("b", source=_DS), Join()])
    result = run(seq, reader)
    # Inner join → only P1 and P2
    assert len(result) == 2
    assert "feat_a" in result.columns
    assert "feat_b" in result.columns


def test_run_join_drops_duplicate_columns():
    # Both frames have a "label" column — the duplicate should be dropped
    df_a = pd.DataFrame({"patient_id": ["P1"], "val": [1.0], "label": [0]})
    df_b = pd.DataFrame({"patient_id": ["P1"], "score": [5.0], "label": [0]})
    reader = MockReader({"a": df_a, "b": df_b})
    seq = Sequence(steps=[Load("a", source=_DS), Load("b", source=_DS), Join(on="patient_id")])
    result = run(seq, reader)
    # Should not have _dup suffix columns
    dup_cols = [c for c in result.columns if c.endswith("_dup")]
    assert dup_cols == []


def test_run_left_join_keeps_all_left():
    df_a = pd.DataFrame({"patient_id": ["P1", "P2", "P3"], "feat_a": [1.0, 2.0, 3.0]})
    df_b = pd.DataFrame({"patient_id": ["P1", "P2"], "feat_b": [10.0, 20.0]})
    reader = MockReader({"a": df_a, "b": df_b})
    seq = Sequence(steps=[Load("a", source=_DS), Load("b", source=_DS), Join(how="left")])
    result = run(seq, reader)
    assert len(result) == 3  # all left rows preserved


def test_run_raises_on_empty_stack_join():
    df_a = pd.DataFrame({"patient_id": ["P1"], "feat_a": [1.0]})
    reader = MockReader({"a": df_a})
    # Only one Load before Join — stack underflow
    seq = Sequence(steps=[Load("a", source=_DS), Join()])
    with pytest.raises(RuntimeError, match="at least 2"):
        run(seq, reader)


def test_run_raises_on_multiple_frames_remaining():
    df_a = pd.DataFrame({"patient_id": ["P1"], "feat_a": [1.0]})
    df_b = pd.DataFrame({"patient_id": ["P1"], "feat_b": [2.0]})
    reader = MockReader({"a": df_a, "b": df_b})
    # Two Loads, no Join → two frames on stack
    seq = Sequence(steps=[Load("a", source=_DS), Load("b", source=_DS)])
    with pytest.raises(RuntimeError, match="2 DataFrames"):
        run(seq, reader)


# ---------------------------------------------------------------------------
# Log2Normalization tests
# ---------------------------------------------------------------------------

def test_log2_normalization_transforms_numeric_cols():
    df = pd.DataFrame({"patient_id": ["P1", "P2"], "expr_a": [0.0, 3.0], "expr_b": [7.0, 15.0]})
    reader = MockReader({"data": df})
    seq = Sequence(steps=[Load("data", source=_DS), Log2Normalization()])
    result = run(seq, reader)
    expected_a = np.log2(np.array([0.0, 3.0]) + 1)
    expected_b = np.log2(np.array([7.0, 15.0]) + 1)
    np.testing.assert_allclose(result["expr_a"].values, expected_a)
    np.testing.assert_allclose(result["expr_b"].values, expected_b)


def test_log2_normalization_preserves_patient_id():
    df = pd.DataFrame({"patient_id": ["P1", "P2"], "expr": [4.0, 8.0]})
    reader = MockReader({"data": df})
    seq = Sequence(steps=[Load("data", source=_DS), Log2Normalization()])
    result = run(seq, reader)
    assert list(result["patient_id"]) == ["P1", "P2"]


def test_log2_normalization_preserves_string_columns():
    df = pd.DataFrame({"patient_id": ["P1"], "label": ["LumA"], "expr": [3.0]})
    reader = MockReader({"data": df})
    seq = Sequence(steps=[Load("data", source=_DS), Log2Normalization()])
    result = run(seq, reader)
    assert result["label"].iloc[0] == "LumA"
    np.testing.assert_allclose(result["expr"].values, np.log2(np.array([3.0]) + 1))


def test_log2_normalization_single_frame_on_stack():
    df = pd.DataFrame({"patient_id": ["P1"], "expr": [1.0]})
    reader = MockReader({"data": df})
    seq = Sequence(steps=[Load("data", source=_DS), Log2Normalization()])
    result = run(seq, reader)
    # run() returns exactly one DataFrame — no RuntimeError
    assert result.shape == (1, 2)


def test_flatten_log2_normalization():
    node = Log2Normalization()
    assert _flatten(node) == [node]
