from __future__ import annotations

import subprocess

from src.configs.loader import load_config


def main() -> None:
    cfg = load_config()
    orch = cfg.get("orchestration", {})

    deployment_name = orch.get("deployment_name", "auto-retrain")
    work_pool = orch.get("work_pool", "local-pool")
    cron = orch.get("retrain_cron", "0 3 * * *")

    subprocess.run(
        [
            "prefect",
            "deploy",
            "flows/auto_retrain_flow.py:auto_retrain_flow",
            "--name",
            deployment_name,
            "--pool",
            work_pool,
            "--cron",
            cron,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()