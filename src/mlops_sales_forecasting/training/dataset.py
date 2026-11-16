import pandas as pd


def normalize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature dtypes for model training and MLflow signature inference."""
    df = df.copy()

    object_columns = df.select_dtypes(include=["object"]).columns
    for col in object_columns:
        df[col] = df[col].astype("category")

    return df


def load_training_data(
    train_file: str,
    val_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data from local or object storage."""
    df_train = pd.read_parquet(train_file)
    df_val = pd.read_parquet(val_file)

    return df_train, df_val
