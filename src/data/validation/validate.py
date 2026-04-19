import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from src.data.validation.forecasting_schema import SalesSchema, StoreSchema
from src.data.validation.inference_schema import inference_schema

def _coerce_datetime_if_present(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Ensure datetime-like columns are parsed before Pandera validation.
    """
    df = df.copy()

    if column_name in df.columns and not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        df[column_name] = pd.to_datetime(df[column_name])

    return df

def validate_inference(df): 
    validated = inference_schema.validate(df.copy())

    if "Sales" in validated.columns: 
        raise ValueError("Inference input must not contain target column 'Sales'")
    
    if validated.empty:
        raise ValueError("Inference input is empty.")

    if "Store" in validated.columns and validated["Store"].isna().any():
        raise ValueError("Inference input contains missing Store values.")
    
    return validated



@pa.check_types
def validate_train(df: pd.DataFrame) -> DataFrame[SalesSchema]:
    """
    Validate forecasting train / inference rows against SalesSchema.
    """
    df = _coerce_datetime_if_present(df, "Date")
    return df


@pa.check_types
def validate_store(df: pd.DataFrame) -> DataFrame[StoreSchema]:
    """
    Validate static store metadata against StoreSchema.
    """
    return df