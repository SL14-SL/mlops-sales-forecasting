import glob
import json
import os

import gcsfs
import pandas as pd
import requests

from src.configs.loader import get_path, load_config


# Load configuration
CFG = load_config()
GCP_CFG = load_config("gcp.yaml")

# Extract API URL from config
# This will be localhost in dev.yaml and Cloud Run URL in prod.yaml
API_URL = CFG.get("api", {}).get("url", "http://127.0.0.1:8000/predict")

# Load API key for security
api_key = os.getenv("API_KEY")
headers = {
    "X-API-KEY": api_key
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


def _load_latest_batch() -> pd.DataFrame:
    """
    Load the latest available batch from local storage or GCS.
    Fall back to the validation split if no new batch exists.
    """
    raw_dir = get_path("raw_data")
    is_gcs = raw_dir.startswith("gs://")

    batch_files = []

    if is_gcs:
        fs = gcsfs.GCSFileSystem()
        bucket_name = GCP_CFG["gcp"]["gcs"]["bucket_name"]

        found_files = fs.ls(
            f"gs://{bucket_name}/data/raw/new_batches/",
            detail=True,
        )

        batch_files = [
            f"gs://{f['name']}"
            for f in found_files
            if "ground_truth_" in f["name"]
        ]

        batch_files.sort(
            key=lambda x: next(
                f["updated"] for f in found_files if f"gs://{f['name']}" == x
            ),
            reverse=True,
        )
    else:
        batch_pattern = os.path.join(raw_dir, "new_batches", "*.csv")
        batch_files = glob.glob(batch_pattern)
        batch_files.sort(key=os.path.getctime, reverse=True)

    if batch_files:
        latest_batch = batch_files[0]
        print(f"🔥 Using latest batch data from: {latest_batch}")
        df = pd.read_csv(latest_batch)

        if "StateHoliday" in df.columns:
            df["StateHoliday"] = df["StateHoliday"].astype(str)

        return df

    splits_dir = get_path("splits")
    val_file = os.path.join(splits_dir, "val.parquet")
    print(f"⚠️ No new batches found. Falling back to: {val_file}")

    df = pd.read_parquet(val_file)

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    return df


def _prepare_request_dataframe(
    df: pd.DataFrame,
    *,
    use_full_batch: bool = True,
    n_requests: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Prepare the request dataframe either from the full daily batch
    or from a sampled subset.
    """
    request_df = df.drop(columns=["Sales"], errors="ignore").copy()
    request_df = request_df[REQUEST_COLUMNS].copy()

    if use_full_batch:
        return request_df

    sample_size = min(n_requests, len(request_df))
    return request_df.sample(sample_size, random_state=random_state)


def run_stress_test(
    n_requests: int = 100,
    *,
    use_full_batch: bool = True,
    random_state: int = 42,
) -> None:
    """
    Send requests to the prediction API using either the full latest daily batch
    or a sampled subset.
    """
    df = _load_latest_batch()

    request_df = _prepare_request_dataframe(
        df,
        use_full_batch=use_full_batch,
        n_requests=n_requests,
        random_state=random_state,
    )

    payloads = json.loads(
        request_df.to_json(orient="records", date_format="iso")
    )

    mode = "full batch" if use_full_batch else f"sample ({len(payloads)} rows)"
    print(f"🚀 Sending {len(payloads)} rows to {API_URL} using {mode} mode...")

    try:
        request_body = {"inputs": payloads}
        response = requests.post(
            API_URL,
            json=request_body,
            headers=headers,
            timeout=300,
        )

        if response.status_code == 200:
            body = response.json()
            print(
                f"✅ Batch request successful. "
                f"rows={body.get('metadata', {}).get('rows')} | "
                f"unique_stores={body.get('metadata', {}).get('unique_stores')}"
            )
        else:
            print(
                f"❌ Batch request failed with status {response.status_code}: "
                f"{response.text}"
            )

    except Exception as e:
        print(f"❌ Batch request raised exception: {e}")

        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run stress test against prediction API")

    parser.add_argument(
        "--use-full-batch",
        action="store_true",
        help="Use full batch instead of sampling"
    )

    parser.add_argument(
        "--n-requests",
        type=int,
        default=100,
        help="Number of sampled requests if not using full batch"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling"
    )

    args = parser.parse_args()

    run_stress_test(
        n_requests=args.n_requests,
        use_full_batch=args.use_full_batch,
        random_state=args.random_state,
    )