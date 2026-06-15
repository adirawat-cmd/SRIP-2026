"""Tests for hgad_cms.graph.constants."""

import pytest

from hgad_cms.graph.constants import (
    G2_EXPECTED_EDGE_COUNTS_V1_1,
    SCHEMA_REGISTRY,
    SCHEMA_V1_1,
    SCHEMA_V1_1_NO_PP,
    get_schema,
)


def test_schema_registry_contains_all_ablations():
    expected = {"v1.1", "v1.1-no_pp", "v1.1-pp_t1", "v1.1-pp_t5", "v1b", "v2", "v4"}
    assert expected == set(SCHEMA_REGISTRY.keys())


def test_v1_1_pp_threshold():
    assert SCHEMA_V1_1.pp_min_shared_beneficiaries == 2
    assert SCHEMA_V1_1.include_provider_provider is True


def test_v1_1_no_pp_disables_collaborates():
    assert SCHEMA_V1_1_NO_PP.include_provider_provider is False
    assert "collaborates" not in SCHEMA_V1_1_NO_PP.relation_names()


def test_get_schema_unknown_raises():
    with pytest.raises(KeyError):
        get_schema("missing")


def test_g2_expected_edge_counts_v1_1():
    assert G2_EXPECTED_EDGE_COUNTS_V1_1["collaborates"] == 75_604
