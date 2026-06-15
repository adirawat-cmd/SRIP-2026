"""Tests for hgad_cms.constants."""

from hgad_cms.constants import (
    EXPECTED_CLAIM_COUNT,
    EXPECTED_PROVIDER_COUNT,
    FRAUD_LABEL_MAP,
    GENDER_FEMALE_VALUES,
    GENDER_MALE_VALUES,
)


def test_fraud_label_map_values():
    assert FRAUD_LABEL_MAP["Yes"] == 1
    assert FRAUD_LABEL_MAP["No"] == 0


def test_expected_counts_match_g1_spec():
    assert EXPECTED_PROVIDER_COUNT == 5410
    assert EXPECTED_CLAIM_COUNT == 558_211


def test_gender_value_sets_disjoint():
    assert GENDER_FEMALE_VALUES.isdisjoint(GENDER_MALE_VALUES)
