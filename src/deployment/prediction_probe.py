from typing import Any

import pandas as pd


PROBE_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Date",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]


def build_prediction_probe(
    *,
    validated_data_path: str,
) -> dict[str, Any]:
    """
    Build one deterministic prediction request from validated release data.

    The newest usable open-store row is selected so the probe uses data that
    belongs to the same dataset snapshot as the serving release.
    """
    source_df = pd.read_parquet(
        validated_data_path
    )

    missing_columns = [
        column
        for column in PROBE_COLUMNS
        if column not in source_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction probe source is missing "
            f"columns: {missing_columns}"
        )

    probe_df = source_df[
        PROBE_COLUMNS
    ].copy()

    probe_df["Date"] = pd.to_datetime(
        probe_df["Date"],
        errors="coerce",
    )

    numeric_columns = [
        "Store",
        "DayOfWeek",
        "Customers",
        "Open",
        "Promo",
        "SchoolHoliday",
    ]

    for column in numeric_columns:
        probe_df[column] = pd.to_numeric(
            probe_df[column],
            errors="coerce",
        )

    probe_df = probe_df.dropna(
        subset=PROBE_COLUMNS
    )

    probe_df = probe_df.loc[
        (probe_df["Open"] == 1)
        & (probe_df["Customers"] > 0)
    ]

    if probe_df.empty:
        raise ValueError(
            "Prediction probe source contains "
            "no usable open-store row."
        )

    # Deterministic selection: newest date, then lowest store ID.
    probe_row = (
        probe_df.sort_values(
            ["Date", "Store"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    prediction_input = {
        "Store": int(
            probe_row["Store"]
        ),
        "DayOfWeek": int(
            probe_row["DayOfWeek"]
        ),
        "Date": (
            probe_row["Date"]
            .date()
            .isoformat()
        ),
        "Customers": int(
            probe_row["Customers"]
        ),
        "Open": int(
            probe_row["Open"]
        ),
        "Promo": int(
            probe_row["Promo"]
        ),
        "StateHoliday": str(
            probe_row["StateHoliday"]
        ),
        "SchoolHoliday": int(
            probe_row["SchoolHoliday"]
        ),
    }

    return {
        "inputs": [
            prediction_input,
        ],
        "context": {
            "purpose": (
                "post_deployment_verification"
            ),
        },
    }