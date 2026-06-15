"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hgad_cms.constants import (
    COL_BENE_ID,
    COL_CLAIM_ID,
    COL_POTENTIAL_FRAUD,
    COL_PROVIDER,
    FRAUD_LABEL_NO,
    FRAUD_LABEL_YES,
)


@pytest.fixture
def sample_providers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_PROVIDER: ["PRV1", "PRV2", "PRV3"],
            COL_POTENTIAL_FRAUD: [FRAUD_LABEL_NO, FRAUD_LABEL_YES, FRAUD_LABEL_NO],
        }
    )


@pytest.fixture
def sample_inpatient() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_BENE_ID: ["B1", "B2"],
            COL_CLAIM_ID: ["C1", "C2"],
            "ClaimStartDt": ["2009-01-01", "2009-02-01"],
            "ClaimEndDt": ["2009-01-05", "2009-02-01"],
            COL_PROVIDER: ["PRV1", "PRV2"],
            "InscClaimAmtReimbursed": [1000.0, 2000.0],
            "AttendingPhysician": ["PHY1", "PHY2"],
            "OperatingPhysician": ["NA", "PHY2"],
            "OtherPhysician": ["NA", "NA"],
            "AdmissionDt": ["2009-01-01", "2009-02-01"],
            "ClmAdmitDiagnosisCode": ["A1", "A2"],
            "DeductibleAmtPaid": [100, 200],
            "DischargeDt": ["2009-01-05", "2009-02-01"],
            "DiagnosisGroupCode": ["001", "002"],
            "ClmDiagnosisCode_1": ["D1", "D2"],
            "ClmProcedureCode_1": ["P1", "P2"],
        }
    )


@pytest.fixture
def sample_outpatient() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_BENE_ID: ["B3"],
            COL_CLAIM_ID: ["C3"],
            "ClaimStartDt": ["2009-03-01"],
            "ClaimEndDt": ["2009-03-01"],
            COL_PROVIDER: ["PRV1"],
            "InscClaimAmtReimbursed": [50.0],
            "AttendingPhysician": ["PHY3"],
            "OperatingPhysician": ["NA"],
            "OtherPhysician": ["NA"],
            "ClmDiagnosisCode_1": ["D3"],
            "DeductibleAmtPaid": [0],
            "ClmAdmitDiagnosisCode": ["A3"],
        }
    )


@pytest.fixture
def sample_beneficiaries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_BENE_ID: ["B1", "B2", "B3"],
            "DOB": ["1940-01-01", "1950-06-15", "1960-12-31"],
            "DOD": ["NA", "NA", "NA"],
            "Gender": ["F", "M", "f"],
            "Race": [1, 1, 2],
            "RenalDiseaseIndicator": ["0", "0", "0"],
            "State": [39, 39, 52],
            "County": [100, 200, 300],
            "NoOfMonths_PartACov": [12, 12, 12],
            "NoOfMonths_PartBCov": [12, 12, 12],
            "ChronicCond_Alzheimer": [1, 2, 2],
            "ChronicCond_Heartfailure": [2, 2, 1],
            "ChronicCond_KidneyDisease": [2, 2, 2],
            "ChronicCond_Cancer": [2, 2, 2],
            "ChronicCond_ObstrPulmonary": [2, 2, 2],
            "ChronicCond_Depression": [2, 2, 2],
            "ChronicCond_Diabetes": [1, 2, 2],
            "ChronicCond_IschemicHeart": [1, 2, 2],
            "ChronicCond_Osteoporasis": [2, 2, 2],
            "ChronicCond_rheumatoidarthritis": [2, 2, 2],
            "ChronicCond_stroke": [2, 2, 2],
            "IPAnnualReimbursementAmt": [0, 1000, 0],
            "IPAnnualDeductibleAmt": [0, 100, 0],
            "OPAnnualReimbursementAmt": [100, 200, 50],
            "OPAnnualDeductibleAmt": [10, 20, 5],
        }
    )


@pytest.fixture
def cms_raw_dir(tmp_path: Path, sample_providers, sample_inpatient, sample_outpatient, sample_beneficiaries) -> Path:
    """Temporary raw directory with minimal valid CSV set."""
    sample_providers.to_csv(tmp_path / "Train-test.csv", index=False)
    sample_beneficiaries.to_csv(tmp_path / "Train_Beneficiarydata-test.csv", index=False)
    sample_inpatient.to_csv(tmp_path / "Train_Inpatientdata-test.csv", index=False)
    sample_outpatient.to_csv(tmp_path / "Train_Outpatientdata-test.csv", index=False)
    return tmp_path
