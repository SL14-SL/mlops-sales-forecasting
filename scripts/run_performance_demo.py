import json
import subprocess
import sys

import pandas as pd

from src.configs.loader import (
    ensure_dir,
    file_exists,
    get_path,
    join_uri,
    list_files,
    modified_time,
    path_name,
    path_suffix,
    read_text,
    remove_file,
    write_text,
)
from src.constants import PROJECT_ROOT
from src.monitoring.alerts import send_alert
from src.utils.logger import get_logger


# Ensure project root is importable when script is run directly
sys.path.append(str(PROJECT_ROOT))

logger = get_logger(__name__)

PREDICTIONS_PATH = get_path("predictions")
RAW_DATA_PATH = get_path("raw_data")
MONITORING_PATH = get_path("monitoring")
RESULTS_PATH = PROJECT_ROOT / "results"

INFERENCE_LOG_FILE = join_uri(PREDICTIONS_PATH, "inference_log.parquet")
BATCH_DIR = join_uri(RAW_DATA_PATH, "new_batches")
CUMULATIVE_GT_FILE = join_uri(MONITORING_PATH, "cumulative_ground_truth.csv")
METRICS_OUTPUT = join_uri(MONITORING_PATH, "performance_rolling.parquet")
RESULTS_OUTPUT = RESULTS_PATH / "performance_demo_history.csv"
LAST_RETRAIN_FILE = join_uri(MONITORING_PATH, "last_retrain.txt")


def run_command(command: list[str], description: str) -> tuple[str, str]:
    """
    Run a command from project root and capture stdout/stderr.

    Returns:
        ("SUCCESS" | "ERROR" | "END_OF_POOL", full_output)
    """
    logger.info(f"--- 🚀 {description} ---")

    if command[0] != "uv":
        command = ["uv", "run", "--active"] + command

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )

    full_output = []
    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        full_output.append(line)

    process.wait()
    output_str = "".join(full_output)

    if "Remaining days in pool: 0" in output_str:
        return "END_OF_POOL", output_str

    if process.returncode != 0:
        return "ERROR", output_str

    return "SUCCESS", output_str


def find_latest_ground_truth_batch() -> str:
    """
    Find the newest ground_truth_*.csv file in the new_batches directory.

    Works for both local paths and GCS paths.
    """
    candidates = list_files(join_uri(BATCH_DIR, "ground_truth_*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No ground_truth_*.csv files found in: {BATCH_DIR}")

    return max(candidates, key=modified_time)


def build_cumulative_ground_truth(latest_batch_file: str | None = None) -> pd.DataFrame:
    """
    Incrementally update the cumulative ground truth table.

    Works for both local paths and GCS paths.
    """
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if "Store" in df.columns:
            df["Store"] = pd.to_numeric(df["Store"], errors="coerce").astype("Int64")

        df = df.dropna(subset=["Store", "Date"])
        return df

    if not file_exists(CUMULATIVE_GT_FILE):
        batch_files = list_files(join_uri(BATCH_DIR, "ground_truth_*.csv"))

        if not batch_files:
            raise FileNotFoundError(f"No ground_truth_*.csv files found in: {BATCH_DIR}")

        frames = []
        for batch_file in batch_files:
            df = pd.read_csv(batch_file)
            frames.append(_normalize(df))

        cumulative_df = pd.concat(frames, ignore_index=True)
        cumulative_df = cumulative_df.sort_values(["Date", "Store"]).drop_duplicates(
            subset=["Store", "Date"],
            keep="last",
        )

        cumulative_df.to_csv(CUMULATIVE_GT_FILE, index=False)

        logger.info(
            f"🧾 Bootstrapped cumulative ground truth with {len(cumulative_df)} rows "
            f"from {len(batch_files)} batch files."
        )
        return cumulative_df

    if latest_batch_file is None:
        latest_candidates = list_files(join_uri(BATCH_DIR, "ground_truth_*.csv"))

        if not latest_candidates:
            raise FileNotFoundError(f"No ground_truth_*.csv files found in: {BATCH_DIR}")

        latest_batch_file = max(latest_candidates, key=modified_time)

    cumulative_df = pd.read_csv(CUMULATIVE_GT_FILE)
    cumulative_df = _normalize(cumulative_df)

    latest_df = pd.read_csv(latest_batch_file)
    latest_df = _normalize(latest_df)

    cumulative_df = pd.concat([cumulative_df, latest_df], ignore_index=True)
    cumulative_df = cumulative_df.sort_values(["Date", "Store"]).drop_duplicates(
        subset=["Store", "Date"],
        keep="last",
    )

    cumulative_df.to_csv(CUMULATIVE_GT_FILE, index=False)

    logger.info(
        f"🧾 Incrementally updated cumulative ground truth with {len(cumulative_df)} rows "
        f"using latest batch: {path_name(latest_batch_file)}"
    )

    return cumulative_df


def load_metrics(path: str) -> pd.DataFrame:
    """
    Load metrics from local or GCS path.
    """
    if not file_exists(path):
        return pd.DataFrame()

    suffix = path_suffix(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported metrics file format: {suffix}")


def save_metrics_table(metrics: pd.DataFrame, output_path: str) -> None:
    """
    Save metrics as parquet or CSV depending on the file suffix.

    Works for both local paths and GCS paths.
    """
    suffix = path_suffix(output_path)

    if suffix == ".parquet":
        metrics.to_parquet(output_path, index=False)
        return

    if suffix == ".csv":
        metrics.to_csv(output_path, index=False)
        return

    raise ValueError(f"Unsupported metrics file format: {suffix}")


def prepare_latest_joined_evaluation_frame() -> pd.DataFrame:
    """
    Load predictions and cumulative ground truth, harmonize keys,
    and return the joined evaluation dataframe.
    """
    if not file_exists(INFERENCE_LOG_FILE):
        raise FileNotFoundError(f"Predictions file not found: {INFERENCE_LOG_FILE}")

    if not file_exists(CUMULATIVE_GT_FILE):
        raise FileNotFoundError(f"Ground truth file not found: {CUMULATIVE_GT_FILE}")

    preds = pd.read_parquet(INFERENCE_LOG_FILE)
    gt = pd.read_csv(CUMULATIVE_GT_FILE)

    if "Date" not in preds.columns or "Date" not in gt.columns:
        raise KeyError("Both predictions and ground truth must contain 'Date'.")

    if "Store" not in preds.columns or "Store" not in gt.columns:
        raise KeyError("Both predictions and ground truth must contain 'Store'.")

    preds["Date"] = pd.to_datetime(preds["Date"], errors="coerce")
    gt["Date"] = pd.to_datetime(gt["Date"], errors="coerce")

    preds["Store"] = pd.to_numeric(preds["Store"], errors="coerce").astype("Int64")
    gt["Store"] = pd.to_numeric(gt["Store"], errors="coerce").astype("Int64")

    preds = preds.dropna(subset=["Store", "Date"])
    gt = gt.dropna(subset=["Store", "Date"])

    joined = preds.merge(gt, on=["Store", "Date"], how="inner")

    if joined.empty:
        raise ValueError("No matching rows found between predictions and ground truth.")

    if "prediction" not in joined.columns:
        raise KeyError("Missing prediction column 'prediction' after merge.")

    if "Sales" not in joined.columns:
        raise KeyError("Missing ground truth column 'Sales' after merge.")

    return joined


def compute_latest_rolling_metrics_row(
    df: pd.DataFrame,
    *,
    time_col: str = "Date",
    window: str = "7D",
    y_true_col: str = "Sales",
    y_pred_col: str = "prediction",
    min_samples: int = 1,
) -> pd.DataFrame:
    """
    Compute only the most recent rolling metrics row instead of recomputing
    the full rolling history.
    """
    working_df = df.copy()
    working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
    working_df = working_df.dropna(subset=[time_col, y_true_col, y_pred_col])
    working_df = working_df.sort_values(time_col)

    if working_df.empty:
        raise ValueError("No valid rows available for latest rolling metrics computation.")

    end_time = working_df[time_col].max()
    start_time = end_time - pd.Timedelta(window)

    window_df = working_df[
        (working_df[time_col] > start_time)
        & (working_df[time_col] <= end_time)
    ].copy()

    if len(window_df) < min_samples:
        return pd.DataFrame(
            columns=["rmse", "mae", "bias", "n_samples", "window_start", "window_end"]
        )

    errors = window_df[y_true_col].astype(float) - window_df[y_pred_col].astype(float)

    metrics_row = pd.DataFrame(
        [
            {
                "rmse": float((errors.pow(2).mean()) ** 0.5),
                "mae": float(errors.abs().mean()),
                "bias": float(errors.mean()),
                "n_samples": int(len(window_df)),
                "window_start": start_time,
                "window_end": end_time,
            }
        ]
    )

    return metrics_row


def append_latest_metrics_row(
    output_path: str,
    latest_row: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append the latest rolling metrics row to the existing metrics table.

    Works for both local paths and GCS paths.
    """
    if latest_row.empty:
        if file_exists(output_path):
            return load_metrics(output_path)
        return latest_row

    if file_exists(output_path):
        existing = load_metrics(output_path)
    else:
        existing = pd.DataFrame(columns=latest_row.columns)

    combined = pd.concat([existing, latest_row], ignore_index=True)

    if "window_end" in combined.columns:
        combined["window_end"] = pd.to_datetime(combined["window_end"], errors="coerce")
        combined = combined.sort_values("window_end").drop_duplicates(
            subset=["window_end"],
            keep="last",
        )

    save_metrics_table(combined, output_path)
    return combined


def read_last_retrain_day() -> int | None:
    """
    Read the last retrain simulation day from local or GCS storage.
    """
    if not file_exists(LAST_RETRAIN_FILE):
        return None

    try:
        return int(read_text(LAST_RETRAIN_FILE).strip())
    except Exception:
        return None


def write_last_retrain_day(day: int) -> None:
    """
    Persist the last retrain simulation day to local or GCS storage.
    """
    write_text(LAST_RETRAIN_FILE, str(day))


def in_retrain_cooldown_by_day(
    current_day: int,
    cooldown_days: int = 14,
) -> tuple[bool, int | None]:
    """
    Check cooldown using simulation days instead of wall-clock time.
    """
    last_retrain_day = read_last_retrain_day()
    if last_retrain_day is None:
        return False, None

    days_since_retrain = current_day - last_retrain_day
    return days_since_retrain < cooldown_days, days_since_retrain


def should_retrain(
    df: pd.DataFrame,
    current_day: int,
    *,
    min_points: int = 14,
    min_samples_latest: int = 1500,
    cooldown_days: int = 14,
    consecutive_bad_points: int = 3,
    rmse_baseline: float = 1300.0,
    mae_baseline: float = 900.0,
    bias_limit: float = 900.0,
    rmse_rel_increase: float = 0.30,
    mae_rel_increase: float = 0.25,
    rmse_step_threshold: float = 250.0,
    bias_step_threshold: float = 200.0,
) -> tuple[bool, str]:
    """
    Decide whether retraining should be triggered based on persistent degradation.
    Uses simulation-day cooldown instead of wall-clock cooldown.
    """
    if df.empty:
        return False, "No monitoring data available."

    if len(df) < min_points:
        return False, f"Not enough metric history yet ({len(df)}/{min_points})."

    cooldown_active, sim_days_since_retrain = in_retrain_cooldown_by_day(
        current_day=current_day,
        cooldown_days=cooldown_days,
    )
    if cooldown_active:
        return (
            False,
            f"Simulation cooldown active "
            f"({sim_days_since_retrain}/{cooldown_days} sim days since last retrain).",
        )

    recent = df.tail(min_points).copy()
    latest = recent.iloc[-1]

    latest_samples = float(latest.get("n_samples", 0) or 0)
    if latest_samples < min_samples_latest:
        return (
            False,
            f"Not enough samples in latest window "
            f"({latest_samples:.0f}/{min_samples_latest}).",
        )

    rmse_limit = rmse_baseline * (1.0 + rmse_rel_increase)
    mae_limit = mae_baseline * (1.0 + mae_rel_increase)

    recent["is_bad"] = (
        ((recent["rmse"] > rmse_limit) & (recent["mae"] > mae_limit))
        | (recent["bias"].abs() > bias_limit)
    )

    last_bad_points = recent["is_bad"].tail(consecutive_bad_points)
    persistent_bad = (
        len(last_bad_points) == consecutive_bad_points and last_bad_points.all()
    )

    if len(recent) < 6:
        return False, "Not enough recent points for robust trend comparison."

    rmse_last3 = recent["rmse"].tail(3).mean()
    rmse_prev3 = recent["rmse"].tail(6).head(3).mean()
    rmse_step_up = (rmse_last3 - rmse_prev3) > rmse_step_threshold

    bias_last3 = recent["bias"].abs().tail(3).mean()
    bias_prev3 = recent["bias"].abs().tail(6).head(3).mean()
    bias_step_up = (bias_last3 - bias_prev3) > bias_step_threshold

    logger.info(
        "Retrain check | "
        f"persistent_bad={persistent_bad} | "
        f"rmse_last3={rmse_last3:.1f} vs prev3={rmse_prev3:.1f} ({rmse_step_up}) | "
        f"bias_last3={bias_last3:.1f} vs prev3={bias_prev3:.1f} ({bias_step_up}) | "
        f"latest_samples={latest_samples:.0f}"
    )

    if persistent_bad and (rmse_step_up or bias_step_up):
        return (
            True,
            "Persistent degradation detected: "
            f"rmse_last3={rmse_last3:.1f} vs prev3={rmse_prev3:.1f}, "
            f"abs_bias_last3={bias_last3:.1f} vs prev3={bias_prev3:.1f}.",
        )

    return False, "No persistent degradation detected."


def extract_training_result(output: str) -> dict:
    """
    Extract the TRAINING_RESULT_JSON line emitted by training_flow.py.
    """
    marker = "TRAINING_RESULT_JSON="
    for line in output.splitlines():
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return {}
    return {}


def main():
    history = []
    day_counter = 1
    max_days = 999

    ensure_dir(MONITORING_PATH)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    if file_exists(INFERENCE_LOG_FILE):
        logger.info(f"🧹 Removing old inference log: {INFERENCE_LOG_FILE}")
        remove_file(INFERENCE_LOG_FILE)

    if file_exists(CUMULATIVE_GT_FILE):
        logger.info(f"🧹 Removing old cumulative ground truth file: {CUMULATIVE_GT_FILE}")
        remove_file(CUMULATIVE_GT_FILE)

    if file_exists(LAST_RETRAIN_FILE):
        logger.info(f"🧹 Removing old retrain marker: {LAST_RETRAIN_FILE}")
        remove_file(LAST_RETRAIN_FILE)

    logger.info("=" * 60)
    logger.info("📈 STARTING PERFORMANCE MONITORING DEMO")
    logger.info("=" * 60)

    while day_counter <= max_days:
        logger.info(f"📅 [DAY {day_counter}]")

        status, _ = run_command(
            ["python", "scripts/simulate_ground_truth.py"],
            f"Simulating Day {day_counter}",
        )

        if status == "END_OF_POOL":
            logger.info("🏁 End of simulation pool reached.")
            break
        if status == "ERROR":
            raise RuntimeError("simulate_ground_truth.py failed")

        latest_batch = find_latest_ground_truth_batch()
        logger.info(f"📦 Latest batch selected: {path_name(latest_batch)}")

        status, _ = run_command(
            ["python", "scripts/stress_test.py", "--n-requests", "150"],
            f"Generating Predictions for Day {day_counter}",
        )
        if status == "ERROR":
            raise RuntimeError("stress_test.py failed")

        cumulative_gt_df = build_cumulative_ground_truth(latest_batch)

        unique_days = (
            cumulative_gt_df["Date"].dt.date.nunique()
            if "Date" in cumulative_gt_df.columns and not cumulative_gt_df.empty
            else 0
        )
        logger.info(f"🗓 Ground truth currently covers {unique_days} unique days.")

        logger.info(f"--- 🚀 Computing Latest Rolling Metrics for Day {day_counter} ---")

        joined_eval_df = prepare_latest_joined_evaluation_frame()

        latest_metrics_row = compute_latest_rolling_metrics_row(
            joined_eval_df,
            time_col="Date",
            window="7D",
            y_true_col="Sales",
            y_pred_col="prediction",
            min_samples=1,
        )

        metrics_df = append_latest_metrics_row(METRICS_OUTPUT, latest_metrics_row)

        event = None
        champion_promoted = False
        latest_metrics = {}

        if latest_metrics_row.empty:
            logger.warning("No metrics available yet for the latest rolling window.")
        else:
            latest_metrics = latest_metrics_row.iloc[0].to_dict()

            logger.info("✅ Latest rolling metrics row computed and appended.")

            logger.info(
                f"📊 Latest Metrics | "
                f"RMSE={latest_metrics.get('rmse')} | "
                f"MAE={latest_metrics.get('mae')} | "
                f"Bias={latest_metrics.get('bias')} | "
                f"Samples={latest_metrics.get('n_samples')}"
            )

            retrain_needed, retrain_reason = should_retrain(
                metrics_df,
                current_day=day_counter,
                min_points=10,
                min_samples_latest=600,
                cooldown_days=21,
                consecutive_bad_points=2,
                rmse_baseline=2200.0,
                mae_baseline=1850.0,
                bias_limit=1650.0,
                rmse_rel_increase=0.112,
                mae_rel_increase=0.112,
                rmse_step_threshold=180.0,
                bias_step_threshold=1100.0,
            )

            if retrain_needed:
                event = "alert"

                send_alert(
                    title="Model performance degraded",
                    message=(
                        f"{retrain_reason} | "
                        f"RMSE={latest_metrics.get('rmse')}, "
                        f"MAE={latest_metrics.get('mae')}, "
                        f"Bias={latest_metrics.get('bias')}, "
                        f"Samples={latest_metrics.get('n_samples')}"
                    ),
                    severity="critical",
                )

                logger.warning(
                    f"🚨 Performance degraded → triggering retraining | {retrain_reason}"
                )

                retrain_status, retrain_output = run_command(
                    ["python", "-m", "flows.training_flow"],
                    "Retraining triggered by performance",
                )

                if retrain_status == "ERROR":
                    send_alert(
                        title="Retraining failed",
                        message="Performance degradation detected, but retraining failed.",
                        severity="critical",
                    )
                    raise RuntimeError(
                        "training_flow failed during performance-triggered retraining"
                    )

                training_result = extract_training_result(retrain_output)
                champion_promoted = bool(training_result.get("champion_promoted", False))

                write_last_retrain_day(day_counter)
                event = "retrain"
            else:
                logger.info(f"✅ No retrain triggered | {retrain_reason}")

        history.append(
            {
                "day": day_counter,
                "latest_batch_file": path_name(latest_batch),
                "cumulative_days": unique_days,
                "rmse": latest_metrics.get("rmse"),
                "mae": latest_metrics.get("mae"),
                "bias": latest_metrics.get("bias"),
                "n_samples": latest_metrics.get("n_samples"),
                "window_start": latest_metrics.get("window_start"),
                "window_end": latest_metrics.get("window_end"),
                "event": event,
                "champion_promoted": champion_promoted,
            }
        )

        pd.DataFrame(history).to_csv(RESULTS_OUTPUT, index=False)

        day_counter += 1

    logger.info("=" * 60)
    logger.info(f"✅ Demo finished. Results saved to: {RESULTS_OUTPUT}")
    logger.info(f"✅ Cumulative ground truth saved to: {CUMULATIVE_GT_FILE}")
    logger.info(f"✅ Rolling metrics saved to: {METRICS_OUTPUT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()