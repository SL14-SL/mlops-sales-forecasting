import argparse
import json
import os
from typing import Any

import pandas as pd
import requests

from src.configs.loader import load_config, get_path


CFG = load_config()
API_URL = CFG.get("api", {}).get("url", "http://127.0.0.1:8000/predict")

API_KEY = os.getenv("API_KEY")
HEADERS = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json",
}

REQUEST_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Date",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]


def load_latest_batch() -> pd.DataFrame:
    raw_dir = get_path("raw_data")
    batch_dir = os.path.join(raw_dir, "new_batches")

    if not os.path.isdir(batch_dir):
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    candidates = [
        os.path.join(batch_dir, name)
        for name in os.listdir(batch_dir)
        if name.endswith(".csv") and "ground_truth_" in name
    ]

    if not candidates:
        raise FileNotFoundError(f"No ground_truth_*.csv files found in: {batch_dir}")

    candidates.sort(key=os.path.getmtime, reverse=True)
    latest_batch = candidates[0]

    print(f"Using latest batch file: {latest_batch}")

    df = pd.read_csv(latest_batch)

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    return df


def prepare_rows(
    df: pd.DataFrame,
    n_rows: int,
    random_state: int,
) -> list[dict[str, Any]]:
    request_df = df.drop(columns=["Sales"], errors="ignore").copy()
    request_df = request_df[REQUEST_COLUMNS].copy()

    sample_size = min(n_rows, len(request_df))
    sampled = request_df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    rows = json.loads(sampled.to_json(orient="records", date_format="iso"))
    return rows


def post_predict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {"inputs": rows}
    response = requests.post(
        API_URL,
        json=payload,
        headers=HEADERS,
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed with status {response.status_code}: {response.text}"
        )

    return response.json()


def run_single_predictions(rows: list[dict[str, Any]]) -> list[float]:
    predictions: list[float] = []

    for idx, row in enumerate(rows, start=1):
        body = post_predict([row])

        preds = body.get("predictions", [])
        if len(preds) != 1:
            raise RuntimeError(
                f"Single request for row {idx} returned {len(preds)} predictions instead of 1."
            )

        predictions.append(float(preds[0]))

    return predictions


def run_batch_prediction(rows: list[dict[str, Any]]) -> list[float]:
    body = post_predict(rows)

    preds = body.get("predictions", [])
    if len(preds) != len(rows):
        raise RuntimeError(
            f"Batch request returned {len(preds)} predictions for {len(rows)} rows."
        )

    return [float(x) for x in preds]


def build_comparison_table(
    rows: list[dict[str, Any]],
    single_preds: list[float],
    batch_preds: list[float],
) -> pd.DataFrame:
    records = []

    for i, (row, single_pred, batch_pred) in enumerate(
        zip(rows, single_preds, batch_preds),
        start=1,
    ):
        diff = batch_pred - single_pred
        abs_diff = abs(diff)

        records.append(
            {
                "row_no": i,
                "Store": row.get("Store"),
                "Date": row.get("Date"),
                "Customers": row.get("Customers"),
                "single_pred": single_pred,
                "batch_pred": batch_pred,
                "diff": diff,
                "abs_diff": abs_diff,
            }
        )

    result_df = pd.DataFrame(records)
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare single-row predictions vs batch predictions."
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=5,
        help="Number of rows to compare.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling rows from the latest batch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/compare_single_vs_batch.csv",
        help="Path to save the comparison CSV.",
    )

    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("API_KEY environment variable is not set.")

    df = load_latest_batch()
    rows = prepare_rows(df, n_rows=args.n_rows, random_state=args.random_state)

    print(f"Sending {len(rows)} single requests...")
    single_preds = run_single_predictions(rows)

    print(f"Sending 1 batch request with {len(rows)} rows...")
    batch_preds = run_batch_prediction(rows)

    comparison_df = build_comparison_table(rows, single_preds, batch_preds)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    comparison_df.to_csv(args.output, index=False)

    max_abs_diff = float(comparison_df["abs_diff"].max()) if not comparison_df.empty else 0.0
    mean_abs_diff = float(comparison_df["abs_diff"].mean()) if not comparison_df.empty else 0.0

    print()
    print(comparison_df.to_string(index=False))
    print()
    print(f"Saved comparison to: {args.output}")
    print(f"Max abs diff:  {max_abs_diff:.6f}")
    print(f"Mean abs diff: {mean_abs_diff:.6f}")

    if max_abs_diff > 1e-6:
        print("WARNING: Batch and single predictions differ.")
    else:
        print("OK: Batch and single predictions are identical.")


if __name__ == "__main__":
    main()