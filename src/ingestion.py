from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
import re

import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "source_type",
    "source_id",
    "property_id",
    "building_id",
    "unit_id",
    "resident_id",
    "timestamp",
    "text",
]

SOURCE_TYPE_OPTIONS = [
    "maintenance_request",
    "complaint",
    "review",
    "event_feedback",
    "amenity_request",
    "message",
]

INPUT_COLUMN_ALIASES = {
    "source_id": ["source_id", "id", "request_id", "feedback_id"],
    "property_id": ["property_id", "property", "community_id"],
    "building_id": ["building_id", "building", "tower_id"],
    "unit_id": ["unit_id", "unit", "apartment", "apt"],
    "resident_id": ["resident_id", "resident", "tenant_id", "user_id"],
    "timestamp": ["timestamp", "created_at", "date", "submitted_at"],
    "text": ["text", "message", "feedback", "description", "comment", "content"],
}

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


@dataclass
class IngestionResult:
    normalized_df: pd.DataFrame
    warnings: list[str]


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    return df.rename(columns=renamed)


def _find_alias_column(columns: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def mask_pii(text: str) -> tuple[str, bool]:
    masked = text
    replaced = False
    for pattern, replacement in (
        (EMAIL_PATTERN, "[REDACTED_EMAIL]"),
        (PHONE_PATTERN, "[REDACTED_PHONE]"),
        (SSN_PATTERN, "[REDACTED_SSN]"),
    ):
        updated = pattern.sub(replacement, masked)
        if updated != masked:
            replaced = True
        masked = updated
    return masked, replaced


def normalize_feedback_dataframe(
    df: pd.DataFrame,
    default_source_type: str,
    default_property_id: str,
) -> IngestionResult:
    warnings: list[str] = []
    working_df = _normalize_column_names(df.copy())
    columns = list(working_df.columns)

    output_df = pd.DataFrame(index=working_df.index)
    output_df["source_type"] = default_source_type
    output_df["property_id"] = default_property_id
    output_df["building_id"] = None

    for target_col, aliases in INPUT_COLUMN_ALIASES.items():
        found_col = _find_alias_column(columns, aliases)
        if found_col is not None:
            output_df[target_col] = working_df[found_col]
        else:
            output_df[target_col] = None
            warnings.append(f"Missing input column for `{target_col}`; using null values.")

    # Ensure text is string and strip whitespace.
    output_df["text"] = output_df["text"].fillna("").astype(str).str.strip()

    # Basic PII masking for common personal data patterns.
    pii_masked_count = 0
    masked_texts: list[str] = []
    for value in output_df["text"]:
        masked, changed = mask_pii(value)
        pii_masked_count += int(changed)
        masked_texts.append(masked)
    output_df["text"] = masked_texts
    if pii_masked_count:
        warnings.append(f"PII masking applied to {pii_masked_count} rows.")

    # Drop fully empty text rows because they cannot be used downstream.
    before_count = len(output_df)
    output_df = output_df[output_df["text"] != ""].copy()
    after_count = len(output_df)
    if after_count < before_count:
        warnings.append(f"Dropped {before_count - after_count} rows with empty text.")

    # Parse timestamp when present.
    output_df["timestamp"] = pd.to_datetime(output_df["timestamp"], errors="coerce")
    invalid_ts = output_df["timestamp"].isna().sum()
    if invalid_ts:
        warnings.append(f"{invalid_ts} rows have invalid/missing timestamp.")

    # Generate source_id when missing.
    missing_source_id = output_df["source_id"].isna() | (output_df["source_id"].astype(str).str.strip() == "")
    if missing_source_id.any():
        output_df.loc[missing_source_id, "source_id"] = [
            f"auto_{i}" for i in output_df.index[missing_source_id]
        ]
        warnings.append("Generated source_id values for rows with missing IDs.")

    # Ensure consistent column order.
    output_df = output_df[REQUIRED_OUTPUT_COLUMNS]
    return IngestionResult(normalized_df=output_df, warnings=warnings)


def read_txt_feedback(file_obj: BinaryIO) -> pd.DataFrame:
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return pd.DataFrame({"text": lines})


def save_normalized_data(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    output_path = output_dir / f"{safe_stem}_normalized.csv"
    df.to_csv(output_path, index=False)
    return output_path
