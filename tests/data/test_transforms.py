"""Unit tests for pipeline label transform functions."""
import pytest

from oncolearn.data.pipeline.transforms import make_subtype_transform, map_ajcc_stage


# ---------------------------------------------------------------------------
# make_subtype_transform
# ---------------------------------------------------------------------------

def test_subtype_transform_known_label():
    transform = make_subtype_transform({"A": 0, "B": 1})
    assert transform("A") == 0
    assert transform("B") == 1


def test_subtype_transform_unknown_string():
    transform = make_subtype_transform({"A": 0, "B": 1})
    assert transform("C") is None


def test_subtype_transform_empty_string():
    transform = make_subtype_transform({"A": 0})
    assert transform("") is None


def test_subtype_transform_non_string_none():
    transform = make_subtype_transform({"A": 0})
    assert transform(None) is None


def test_subtype_transform_non_string_float():
    transform = make_subtype_transform({"A": 0})
    assert transform(1.5) is None


def test_subtype_transform_whitespace_padded():
    transform = make_subtype_transform({"BRCA_LumA": 2})
    assert transform("  BRCA_LumA  ") == 2


def test_subtype_transform_pam50_mapping():
    _PAM50 = {"BRCA_Basal": 0, "BRCA_Her2": 1, "BRCA_LumA": 2, "BRCA_LumB": 3, "BRCA_Normal": 4}
    transform = make_subtype_transform(_PAM50)
    assert transform("BRCA_Basal") == 0
    assert transform("BRCA_Normal") == 4
    assert transform("BRCA_Unknown") is None


# ---------------------------------------------------------------------------
# map_ajcc_stage
# ---------------------------------------------------------------------------

def test_map_ajcc_stage_stage_I():
    assert map_ajcc_stage("Stage I") == 0
    assert map_ajcc_stage("Stage IA") == 0
    assert map_ajcc_stage("Stage IB") == 0
    assert map_ajcc_stage("STAGE IA") == 0


def test_map_ajcc_stage_stage_II():
    assert map_ajcc_stage("Stage II") == 1
    assert map_ajcc_stage("Stage IIA") == 1
    assert map_ajcc_stage("Stage IIB") == 1
    assert map_ajcc_stage("STAGE IIA") == 1


def test_map_ajcc_stage_stage_III():
    assert map_ajcc_stage("Stage III") == 2
    assert map_ajcc_stage("Stage IIIA") == 2


def test_map_ajcc_stage_stage_IV():
    assert map_ajcc_stage("Stage IV") == 3
    assert map_ajcc_stage("STAGE IV") == 3


def test_map_ajcc_stage_unknown_returns_none():
    assert map_ajcc_stage("Stage X") is None
    assert map_ajcc_stage("STAGE X") is None
    assert map_ajcc_stage("") is None
    assert map_ajcc_stage("unknown") is None


def test_map_ajcc_stage_non_string_returns_none():
    assert map_ajcc_stage(None) is None
    assert map_ajcc_stage(float("nan")) is None
    assert map_ajcc_stage(2) is None
