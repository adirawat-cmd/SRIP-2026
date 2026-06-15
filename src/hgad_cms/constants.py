"""
Canonical column names, file patterns, and validation thresholds for the CMS dataset.
"""

from __future__ import annotations

from typing import Final

__version__: Final[str] = "0.1.0"

# ---------------------------------------------------------------------------
# Raw file glob patterns (Kaggle CMS Healthcare Provider Fraud Detection)
# ---------------------------------------------------------------------------
PROVIDER_FILE_PATTERN: Final[str] = "Train-*.csv"
BENEFICIARY_FILE_PATTERN: Final[str] = "Train_Beneficiarydata-*.csv"
INPATIENT_FILE_PATTERN: Final[str] = "Train_Inpatientdata-*.csv"
OUTPATIENT_FILE_PATTERN: Final[str] = "Train_Outpatientdata-*.csv"

RAW_FILE_PATTERNS: Final[dict[str, str]] = {
    "providers": PROVIDER_FILE_PATTERN,
    "beneficiaries": BENEFICIARY_FILE_PATTERN,
    "inpatient": INPATIENT_FILE_PATTERN,
    "outpatient": OUTPATIENT_FILE_PATTERN,
}

# ---------------------------------------------------------------------------
# Provider / label columns
# ---------------------------------------------------------------------------
COL_PROVIDER: Final[str] = "Provider"
COL_POTENTIAL_FRAUD: Final[str] = "PotentialFraud"
FRAUD_LABEL_YES: Final[str] = "Yes"
FRAUD_LABEL_NO: Final[str] = "No"
FRAUD_LABEL_MAP: Final[dict[str, int]] = {FRAUD_LABEL_NO: 0, FRAUD_LABEL_YES: 1}

# ---------------------------------------------------------------------------
# Beneficiary columns
# ---------------------------------------------------------------------------
COL_BENE_ID: Final[str] = "BeneID"
COL_DOB: Final[str] = "DOB"
COL_DOD: Final[str] = "DOD"
COL_GENDER: Final[str] = "Gender"
COL_RACE: Final[str] = "Race"
COL_RENAL: Final[str] = "RenalDiseaseIndicator"
COL_STATE: Final[str] = "State"
COL_COUNTY: Final[str] = "County"
COL_PART_A_MONTHS: Final[str] = "NoOfMonths_PartACov"
COL_PART_B_MONTHS: Final[str] = "NoOfMonths_PartBCov"

CHRONIC_CONDITION_COLUMNS: Final[tuple[str, ...]] = (
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
)

BENEFICIARY_ANNUAL_COLUMNS: Final[tuple[str, ...]] = (
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt",
)

BENEFICIARY_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    COL_BENE_ID,
    COL_DOB,
    COL_DOD,
    COL_GENDER,
    COL_RACE,
    COL_RENAL,
    COL_STATE,
    COL_COUNTY,
    COL_PART_A_MONTHS,
    COL_PART_B_MONTHS,
    *CHRONIC_CONDITION_COLUMNS,
    *BENEFICIARY_ANNUAL_COLUMNS,
)

# ---------------------------------------------------------------------------
# Claim columns (shared + type-specific)
# ---------------------------------------------------------------------------
COL_CLAIM_ID: Final[str] = "ClaimID"
COL_CLAIM_START: Final[str] = "ClaimStartDt"
COL_CLAIM_END: Final[str] = "ClaimEndDt"
COL_REIMBURSED: Final[str] = "InscClaimAmtReimbursed"
COL_ATTENDING: Final[str] = "AttendingPhysician"
COL_OPERATING: Final[str] = "OperatingPhysician"
COL_OTHER_PHYSICIAN: Final[str] = "OtherPhysician"
COL_DEDUCTIBLE: Final[str] = "DeductibleAmtPaid"
COL_ADMIT_DX: Final[str] = "ClmAdmitDiagnosisCode"
COL_ADMISSION: Final[str] = "AdmissionDt"
COL_DISCHARGE: Final[str] = "DischargeDt"
COL_DX_GROUP: Final[str] = "DiagnosisGroupCode"
COL_IS_INPATIENT: Final[str] = "is_inpatient"

DIAGNOSIS_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"ClmDiagnosisCode_{i}" for i in range(1, 11)
)
PROCEDURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"ClmProcedureCode_{i}" for i in range(1, 7)
)

CLAIM_CORE_COLUMNS: Final[tuple[str, ...]] = (
    COL_BENE_ID,
    COL_CLAIM_ID,
    COL_CLAIM_START,
    COL_CLAIM_END,
    COL_PROVIDER,
    COL_REIMBURSED,
    COL_ATTENDING,
    COL_OPERATING,
    COL_OTHER_PHYSICIAN,
    *DIAGNOSIS_COLUMNS,
    *PROCEDURE_COLUMNS,
)

INPATIENT_EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    COL_ADMISSION,
    COL_ADMIT_DX,
    COL_DEDUCTIBLE,
    COL_DISCHARGE,
    COL_DX_GROUP,
)

OUTPATIENT_EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    COL_DEDUCTIBLE,
    COL_ADMIT_DX,
)

# Union column order for merged claims frame
CLAIM_ALL_COLUMNS: Final[tuple[str, ...]] = (
    *CLAIM_CORE_COLUMNS,
    COL_ADMISSION,
    COL_ADMIT_DX,
    COL_DEDUCTIBLE,
    COL_DISCHARGE,
    COL_DX_GROUP,
    COL_IS_INPATIENT,
)

# ---------------------------------------------------------------------------
# Derived / cleaned column names
# ---------------------------------------------------------------------------
COL_GENDER_NORM: Final[str] = "gender_norm"
COL_CLAIM_DURATION_DAYS: Final[str] = "claim_duration_days"
COL_FRAUD_LABEL: Final[str] = "fraud_label"

# ---------------------------------------------------------------------------
# Gender normalization
# ---------------------------------------------------------------------------
GENDER_FEMALE_VALUES: Final[frozenset[str]] = frozenset({"F", "f", "FF"})
GENDER_MALE_VALUES: Final[frozenset[str]] = frozenset({"M", "m", "MF"})
GENDER_UNK_VALUE: Final[str] = "UNK"
GENDER_FEMALE_CODE: Final[int] = 0
GENDER_MALE_CODE: Final[int] = 1
GENDER_UNK_CODE: Final[int] = 2

# ---------------------------------------------------------------------------
# Missing value tokens in raw CSV
# ---------------------------------------------------------------------------
NA_STRINGS: Final[tuple[str, ...]] = ("NA", "NaN", "nan", "", "None", "null")

# ---------------------------------------------------------------------------
# G1 validation thresholds (train corpus)
# ---------------------------------------------------------------------------
EXPECTED_PROVIDER_COUNT: Final[int] = 5410
EXPECTED_CLAIM_COUNT: Final[int] = 558_211
EXPECTED_BENEFICIARY_COUNT: Final[int] = 138_556
EXPECTED_FRAUD_RATE: Final[float] = 0.0935
FRAUD_RATE_TOLERANCE: Final[float] = 0.01
MIN_BENEFICIARY_CLAIM_JOIN_RATE: Final[float] = 0.99
MIN_PROVIDER_CLAIM_JOIN_RATE: Final[float] = 1.0
REIMBURSEMENT_CLIP_QUANTILE: Final[float] = 0.999

# ---------------------------------------------------------------------------
# Processed output filenames
# ---------------------------------------------------------------------------
OUTPUT_CLAIMS: Final[str] = "claims.parquet"
OUTPUT_PROVIDERS: Final[str] = "providers.parquet"
OUTPUT_BENEFICIARIES: Final[str] = "beneficiaries.parquet"
OUTPUT_MANIFEST: Final[str] = "manifest.json"
