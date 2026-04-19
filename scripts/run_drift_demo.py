import os
import subprocess
import pandas as pd
import numpy as np
import sys
import mlflow
from datetime import datetime
from sklearn.metrics import mean_squared_error
from src.configs.loader import load_config
from src.utils.logger import get_logger
from src.configs.loader import get_path


# Add project root to sys.path to resolve src and flows modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


ENV_CFG = load_config()
MODEL_NAME = ENV_CFG["model"]["registry_name"]

# Standard project logger
logger = get_logger(__name__)

def run_command(command, description):
    logger.info(f"--- 🚀 {description} ---")
    # Ensure we use the uv virtual environment runner
    if command[0] != "uv":
        command = ["uv", "run"] + command
    
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        cwd=project_root # Always execute from project root
    )
    
    full_output = []
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

def main():
    history = []
    day_counter = 1
    max_days = 90
    
    # Dynamic path for results based on project root
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "evolution_results.csv")

    # 1. Pipeline Reset & Initial Training
    # Scripts are called relative to the project root (cwd in run_command)
    run_command(["python", "scripts/reset_pipeline.py"], "Resetting Pipeline")
    run_command(["python", "-m", "flows.training_flow", "--force"], "Initial Training (V1)")

    logger.info("="*60)
    logger.info("📈 STARTING SMART EVOLUTION: ADAPTIVE VS. STATIC BASELINE")
    logger.info("="*60)

    while day_counter <= max_days:
        logger.info(f"📅 [DAY {day_counter}]")

        # 2. Simulate Ground Truth
        status, _ = run_command(["python", "scripts/simulate_ground_truth.py"], f"Simulating Day {day_counter}")
        if status == "END_OF_POOL":
            logger.info("🏁 End of data pool reached.")
            break

        # 3. API Interaction
        run_command(["python", "scripts/stress_test.py"], "Injecting to API")

        # 4. Run Adaptive Training Flow
        status, out = run_command(["python", "-m", "flows.training_flow"], "Executing Flow")
        
        # 5. Extract Adaptive Metrics
        extracted_rmse = None
        champ_rmse = None
        strategy = "NORMAL"
        drift_status = "False"

        for line in out.split('\n'):
            if "Challenger RMSE:" in line or "real-scale RMSE:" in line:
                extracted_rmse = line.split(":")[-1].strip()
            if "Champion RMSE:" in line: 
                champ_rmse = line.split(":")[-1].strip()
            if "[EMERGENCY]" in line:
                strategy = "EMERGENCY"
            if "Drift status: True" in line:
                drift_status = "True"

        try:
            final_adaptive_rmse = float(extracted_rmse) if extracted_rmse else (float(champ_rmse) if champ_rmse else None)
        except (ValueError, TypeError):
            final_adaptive_rmse = None

        # --- 6. CALCULATE STATIC BASELINE ---
        static_rmse = None
        try:
            # Load baseline model from registry
            static_model_uri = f"models:/{MODEL_NAME}/1"
            static_model = mlflow.xgboost.load_model(static_model_uri)
            
            # Using the dynamic loader for the validation split path
            val_split_dir = get_path('splits')
            val_path = os.path.join(val_split_dir, "val.parquet")
            
            val_df = pd.read_parquet(val_path)
            
            X_val = val_df.drop(columns=["Sales", "Date"], errors='ignore')
            y_val = val_df["Sales"]

            static_preds_log = static_model.predict(X_val)
            static_preds = np.expm1(static_preds_log)
            static_rmse = np.sqrt(mean_squared_error(y_val, static_preds))
            
            logger.info(f"🛡️ Static Baseline RMSE: {static_rmse:.2f} €")
        except Exception as e:
            logger.warning(f"Could not calculate baseline: {e}")

        # 7. Logging and UI
        strat_color = "🚨" if strategy == "EMERGENCY" else "✅"
        drift_icon = "🌩️" if drift_status == "True" else "☀️"
        
        logger.info(f"RESULT DAY {day_counter}: Strategy: {strat_color} | Drift: {drift_icon}")
        
        # 8. Save to History
        history.append({
            "day": day_counter,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "rmse_euro": final_adaptive_rmse,
            "static_rmse_euro": static_rmse,
            "strategy": strategy,
            "drift_detected": drift_status
        })
        
        pd.DataFrame(history).to_csv(results_file, index=False)

        # Early exit if stable
        if "No drift detected. System stable." in out and day_counter > 30:
            logger.info("✨ STABILITY DETECTED. System adapted.")
            break
            
        day_counter += 1

    logger.info("="*60)
    logger.info(f"✅ Demo finished. Results saved to '{results_file}'.")
    logger.info("="*60)

if __name__ == "__main__":
    main()