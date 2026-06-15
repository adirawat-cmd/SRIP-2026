"""Tests for hgad_cms.data.loader."""

from pathlib import Path

import pandas as pd
import pytest

from hgad_cms.data.loader import detect_file_names, load_raw_cms
from hgad_cms.exceptions import DataLoadError


def test_detect_file_names_resolves_all_patterns(cms_raw_dir: Path):
    paths = detect_file_names(cms_raw_dir)
    assert paths.providers.exists()
    assert paths.beneficiaries.exists()
    assert paths.inpatient.exists()
    assert paths.outpatient.exists()


def test_load_raw_cms_success(cms_raw_dir: Path):
    raw = load_raw_cms(cms_raw_dir)
    assert len(raw.providers) == 3
    assert len(raw.inpatient) == 2
    assert len(raw.outpatient) == 1


def test_detect_file_names_missing_dir(tmp_path: Path):
    with pytest.raises(DataLoadError, match="does not exist"):
        detect_file_names(tmp_path / "missing")


def test_load_raw_cms_missing_provider_column(tmp_path: Path, sample_inpatient, sample_outpatient, sample_beneficiaries):
    pd.DataFrame({"bad": [1]}).to_csv(tmp_path / "Train-x.csv", index=False)
    sample_beneficiaries.to_csv(tmp_path / "Train_Beneficiarydata-x.csv", index=False)
    sample_inpatient.to_csv(tmp_path / "Train_Inpatientdata-x.csv", index=False)
    sample_outpatient.to_csv(tmp_path / "Train_Outpatientdata-x.csv", index=False)
    with pytest.raises(DataLoadError, match="missing columns"):
        load_raw_cms(tmp_path)
