import pandera.pandas as pa
from pandera.typing import Series
from typing import Optional


class SalesSchema(pa.DataFrameModel):
    """
    Schema for forecasting sales data.
    This remains domain-specific on purpose and can later be replaced
    by another policy schema for a different project.
    """
    Store: Series[int] = pa.Field(gt=0)
    Date: Series[pa.DateTime]
    Sales: Optional[Series[float]] = pa.Field(ge=0, nullable=True)
    Customers: Series[int] = pa.Field(ge=0, nullable=True)
    Open: Series[int] = pa.Field(isin=[0, 1])
    Promo: Series[int] = pa.Field(isin=[0, 1])
    StateHoliday: Series[str] = pa.Field(isin=["0", "a", "b", "c"])
    SchoolHoliday: Series[int] = pa.Field(isin=[0, 1])
    DayOfWeek: Optional[Series[int]] = pa.Field(isin=[1, 2, 3, 4, 5, 6, 7], nullable=True)

    class Config:
        # strict=False is a bit more blueprint-friendly than strict=True,
        # because it allows additional columns in broader dataflows.
        # If you want the old exact behavior back, switch this to True.
        strict = False
        coerce = True


class StoreSchema(pa.DataFrameModel):
    """
    Schema for static store metadata.
    This stays separate because metadata validation is a different concern
    from train/inference row validation.
    """
    Store: Series[int] = pa.Field(gt=0)
    StoreType: Series[str] = pa.Field(isin=["a", "b", "c", "d"])
    Assortment: Series[str] = pa.Field(isin=["a", "b", "c"])
    CompetitionDistance: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = False
        coerce = True