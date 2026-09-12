from pathlib import Path
from typing import Optional

import pandas as pd


TEXT_COLUMN_CANDIDATES = [
    "text",
    "tweet_text",
    "message",
    "body",
    "content",
]

BRAND_COLUMN_CANDIDATES = [
    "brand",
    "inbound",
    "author_id",
    "company",
    "account",
]


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    lower_to_original = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    return None


def load_dataset(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".json", ".jsonl"]:
        dataframe = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(
            "Only CSV and JSON/JSONL files are supported."
        )

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    text_column = find_column(
        dataframe,
        TEXT_COLUMN_CANDIDATES
    )

    if text_column is None:
        raise ValueError(
            "Could not find a text column. "
            f"Expected one of: {TEXT_COLUMN_CANDIDATES}"
        )

    dataframe = dataframe.copy()
    dataframe["message"] = (
        dataframe[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[dataframe["message"] != ""]

    brand_column = find_column(
        dataframe,
        BRAND_COLUMN_CANDIDATES
    )

    if brand_column:
        dataframe["detected_brand"] = (
            dataframe[brand_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        dataframe["detected_brand"] = ""

    return dataframe.reset_index(drop=True)


def filter_brand(
    dataframe: pd.DataFrame,
    brand: str,
) -> pd.DataFrame:
    if not brand:
        return dataframe.copy()

    if "detected_brand" not in dataframe.columns:
        return dataframe.copy()

    mask = dataframe["detected_brand"].str.lower() == brand.lower()
    filtered = dataframe[mask].copy()

    if filtered.empty:
        print(
            f"Warning: brand '{brand}' was not found. "
            "Using the complete dataset."
        )
        return dataframe.copy()

    return filtered.reset_index(drop=True)


def load_golden_set(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(file_path)

    required = [
        "message",
        "intent",
        "should_escalate",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Golden set is missing columns: {missing}"
        )

    return dataframe
