import os
import shutil
from src.configs.loader import get_path

def cleanup():
    """
    Cleans up all local data and MLflow artifacts for a fresh start.
    Now including the simulation pool file.
    """
    # 1. Clear Data Directories
    folders_to_clear = ["splits", "features", "validated_data"]
    for folder in folders_to_clear:
        path = get_path(folder)
        if os.path.exists(path):
            print(f"🧹 Clearing {path}...")
            shutil.rmtree(path)
            os.makedirs(path)

    # 2. Clear New Batches and the Simulation Pool
    raw_path = get_path("raw_data")
    
    # Delete the simulation pool file (the "future")
    sim_pool_file = os.path.join(raw_path, "simulation_ground_truth.csv")
    if os.path.exists(sim_pool_file):
        print(f"🧹 Removing simulation pool: {sim_pool_file}")
        os.remove(sim_pool_file)

    # Delete existing batches
    new_batches = os.path.join(raw_path, "new_batches")
    if os.path.exists(new_batches):
        print(f"🧹 Clearing new batches in {new_batches}...")
        shutil.rmtree(new_batches)
        os.makedirs(new_batches)

    # 3. Remove MLflow Database and Artifacts
    if os.path.exists("mlflow.db"):
        print("🧹 Removing mlflow.db...")
        os.remove("mlflow.db")
    
    if os.path.exists("mlruns"):
        print("🧹 Removing mlruns directory...")
        shutil.rmtree("mlruns")

    print("✨ System reset complete. Run 'training_flow --force' next.")

if __name__ == "__main__":
    cleanup()