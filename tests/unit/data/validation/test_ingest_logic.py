import os
import shutil
from unittest.mock import patch

import pandas as pd

from src.data.raw.ingest import ingest


@patch("src.data.raw.ingest.pd.read_csv")
def test_ingest_quarantine_logic(mock_read_csv):
    with patch.dict(os.environ, {"APP_ENV": "dev"}):
        base_test_dir = "data/test_tmp"
        raw_dir = os.path.join(base_test_dir, "raw")
        batch_dir = os.path.join(raw_dir, "new_batches")
        quarantine_dir = os.path.join(raw_dir, "quarantine")

        if os.path.exists(base_test_dir):
            shutil.rmtree(base_test_dir)
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(quarantine_dir, exist_ok=True)

        fake_df = pd.DataFrame(
            {
                "Store": [1],
                "DayOfWeek": [1],
                "Date": ["2026-01-01"],
                "Sales": [10.0],
                "Customers": [5],
                "Open": [1],
                "Promo": [0],
                "StateHoliday": ["0"],
                "SchoolHoliday": [0],
            }
        )
        fake_store = pd.DataFrame(
            {
                "Store": [1],
                "StoreType": ["a"],
                "Assortment": ["a"],
                "CompetitionDistance": [500.0],
            }
        )
        mock_read_csv.side_effect = [fake_df, fake_store]

        invalid_batch = fake_df.copy()
        invalid_batch["Sales"] = -99.0
        bad_batch_path = os.path.join(batch_dir, "bad_batch.csv")
        invalid_batch.to_csv(bad_batch_path, index=False)

        with patch("src.data.raw.ingest.get_path") as mock_get:
            mock_get.side_effect = (
                lambda x: raw_dir if x == "raw_data" else os.path.join(base_test_dir, "output")
            )
            ingest()

        assert os.path.exists(os.path.join(quarantine_dir, "bad_batch.csv"))
        assert not os.path.exists(bad_batch_path)

        shutil.rmtree(base_test_dir)