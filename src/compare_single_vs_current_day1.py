import json
import os
from pathlib import Path

import pandas as pd
import requests

from src.configs.loader import load_config
from src.data.features.build_features import preprocess_data
from src.data.features.core import get_lag_feature_names
from src.api.app import (
    validate_prediction_input,
    align_features_for_model,
    inverse_transform_target,
    apply_prediction_postprocessing,
)

from src.inference.adapters import (
    request_to_dataframe,
    resolve_forecasting_store_id,
    resolve_open_flags,
)

# 🔥 wichtig: Training config, nicht dev.yaml
CFG = load_config("training.yaml")

TARGET_COLUMN = CFG["data"]["target_column"]
ENTITY_COLUMN = CFG["data"]["id_columns"][0]
DATE_COLUMN = CFG["data"]["time_column"]

API_URL = "http://127.0.0.1:8000/predict"
API_KEY = os.getenv("API_KEY")


def load_artifacts():
    import json
    import xgboost as xgb
    import pandas as pd

    model = xgb.XGBRegressor()
    model.load_model("models/model.ubj")

    with open("models/latest_state.json", "r", encoding="utf-8") as f:
        store_state = json.load(f)

    store_metadata = pd.read_parquet("data/validation/store.parquet")
    store_metadata["Store"] = store_metadata["Store"].astype(int)

    print(
        f"Loaded local artifacts | "
        f"model_path=models/model.ubj | "
        f"state_entities={len(store_state)} | "
        f"metadata_rows={len(store_metadata)}"
    )

    return model, store_state, store_metadata


def load_first_sim_day_rows():
    sim = pd.read_csv("data/raw/simulation_ground_truth.csv", low_memory=False)
    sim["Date"] = pd.to_datetime(sim["Date"], errors="coerce")

    first_date = sim["Date"].min()
    df = sim[sim["Date"] == first_date].copy()

    # Target für Inference entfernen
    df = df.drop(columns=["Sales"], errors="ignore")

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    if "Store" in df.columns:
        df["Store"] = pd.to_numeric(df["Store"], errors="coerce").astype(int)

    if "DayOfWeek" in df.columns:
        df["DayOfWeek"] = pd.to_numeric(df["DayOfWeek"], errors="coerce").astype(int)

    if "Customers" in df.columns:
        df["Customers"] = pd.to_numeric(df["Customers"], errors="coerce").astype(int)

    if "Open" in df.columns:
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce").astype(int)

    if "Promo" in df.columns:
        df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").astype(int)

    if "SchoolHoliday" in df.columns:
        df["SchoolHoliday"] = pd.to_numeric(df["SchoolHoliday"], errors="coerce").astype(int)

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df = df[
        [
            "Store",
            "DayOfWeek",
            "Date",
            "Customers",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
        ]
    ].copy()

    print(f"Using first sim date: {first_date.date()} | rows={len(df)}")

    return json.loads(df.to_json(orient="records"))

def call_api(rows):
    response = requests.post(
        API_URL,
        json={"inputs": rows},
        headers={
            "X-API-KEY": API_KEY,
            "Content-Type": "application/json",
        },
        timeout=300,
    )

    if response.status_code != 200:
        print("API status:", response.status_code)
        print("API response text:", response.text)
        response.raise_for_status()

    return response.json()["predictions"]

# ===== OLD SINGLE LOGIC =====

def old_single_predict(rows, model, store_state, store_metadata):
    preds = []

    lag_cfg = CFG.get("features", {}).get("lag_features", {})
    lags = lag_cfg.get("lags", [1, 7])
    rolling_windows = lag_cfg.get("rolling_windows", [7])

    feature_names = get_lag_feature_names(
        TARGET_COLUMN,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    for row in rows:
        df = request_to_dataframe([row])
        df = validate_prediction_input(df)

        store_id = resolve_forecasting_store_id(df)
        open_flags = resolve_open_flags(df)

        df = df.merge(store_metadata, on=ENTITY_COLUMN, how="left")

        df = preprocess_data(df, mode="inference")

        history = store_state.get(str(store_id), [])

        if not history:
            history = [0.0] * 7

        if len(history) < 7:
            history = [0.0] * (7 - len(history)) + history

        for lag in lags:
            key = f"lag_{lag}"
            df[feature_names[key]] = (
                float(history[-lag]) if len(history) >= lag else 0.0
            )

        for window in rolling_windows:
            key = f"rolling_mean_{window}"
            values = history[-window:] if len(history) >= window else history
            df[feature_names[key]] = float(sum(values) / len(values)) if values else 0.0

        if DATE_COLUMN in df.columns:
            df = df.drop(columns=[DATE_COLUMN])

        df = align_features_for_model(
            processed_df=df,
            model=model,
            model_type="xgboost",
        )

        raw_pred = model.predict(df)[0]

        pred = float(
            inverse_transform_target(raw_pred, CFG["training"]["target_transformation"])
        )

        pred = apply_prediction_postprocessing([pred], open_flags)[0]

        preds.append(round(pred, 2))

    return preds


def main():
    model, store_state, store_metadata = load_artifacts()

    rows = load_first_sim_day_rows()

    print("➡️ Calling API...")
    api_preds = call_api(rows)

    print("➡️ Running old single logic...")
    old_preds = old_single_predict(rows, model, store_state, store_metadata)

    df = pd.DataFrame({
        "Store": [r["Store"] for r in rows],
        "api_pred": api_preds,
        "old_pred": old_preds,
    })

    df["diff"] = df["api_pred"] - df["old_pred"]
    df["abs_diff"] = df["diff"].abs()

    print(df.head(20))
    print()
    print("Max abs diff:", df["abs_diff"].max())
    print("Mean abs diff:", df["abs_diff"].mean())

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/debug_old_vs_api.csv", index=False)


if __name__ == "__main__":
    main()