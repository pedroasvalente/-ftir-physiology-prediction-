from pathlib import Path

import pandas as pd

from ftir_pred.data.config import METADATA_COLS, WATER_REGION


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_ftir_columns(df: pd.DataFrame) -> pd.Index:
    cols = df.columns[~df.columns.isin(METADATA_COLS)]
    wavenumbers = pd.to_numeric(cols, errors="coerce")
    in_water = (wavenumbers >= WATER_REGION[0]) & (wavenumbers <= WATER_REGION[1])
    return cols[~in_water]


def filter_samples(
    df: pd.DataFrame,
    sample_type: str,
    target: str,
    timepoints: list[int] | None = None,
    min_test_samples: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """
    Return (X_ftir, y, groups) for the given sample_type and target.
    Returns None if there are too few valid samples.

    groups is group + person_code (globally unique person id) — used for person-aware splits.
    """
    data = df[df["sample_type"] == sample_type].copy()

    if timepoints is not None:
        data = data[data["timepoint"].isin(timepoints)]

    if target not in data.columns:
        return None

    ftir_cols = get_ftir_columns(df)
    X = data[ftir_cols]
    y = data[target]
    groups = data["group"] + "_" + data["person_code"]

    valid = y.notna() & X.notna().all(axis=1) & (X != 0).any(axis=1)
    X, y, groups = X[valid], y[valid], groups[valid]

    if len(y) < min_test_samples * 2:
        return None

    return X.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)
