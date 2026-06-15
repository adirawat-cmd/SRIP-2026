"""Tests for hgad_cms.evaluation.significance."""

import pytest

from hgad_cms.evaluation.significance import compare_all_pairs, paired_test


def test_paired_test_wilcoxon():
    result = paired_test([0.9, 0.8, 0.85], [0.7, 0.75, 0.72], metric="auprc", model_a="a", model_b="b")
    assert result.p_value >= 0.0
    assert result.n_pairs == 3


def test_compare_all_pairs():
    scores = {"a": [0.9, 0.8], "b": [0.7, 0.75], "c": [0.6, 0.65]}
    results = compare_all_pairs(scores, metric="auprc")
    assert len(results) == 3


def test_paired_test_requires_equal_length():
    with pytest.raises(ValueError):
        paired_test([0.9, 0.8], [0.7], metric="auprc", model_a="a", model_b="b")
