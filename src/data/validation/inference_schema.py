import pandera.pandas as pa

inference_schema = pa.DataFrameSchema(
    {
        "Store": pa.Column(int, nullable=False, checks=pa.Check.ge(1)),
        "Date": pa.Column(pa.DateTime, nullable=False),
        "Open": pa.Column(int, nullable=True, checks=pa.Check.isin([0, 1])),
        "Promo": pa.Column(int, nullable=True, checks=pa.Check.isin([0, 1])),
        "StateHoliday": pa.Column(str, nullable=True),
        "SchoolHoliday": pa.Column(int, nullable=True, checks=pa.Check.isin([0, 1])),
    },
    strict=False,
    coerce=True,
)