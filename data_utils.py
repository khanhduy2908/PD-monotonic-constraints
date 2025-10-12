import os
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_master_data() -> pd.DataFrame:
    """Load dataset from /data. Prefer XLSX, fallback to CSV."""
    xlsx = os.path.join("data", "bctc_final.xlsx")
    csv  = os.path.join("data", "bctc_final.csv")
    if os.path.exists(xlsx):
        if os.path.getsize(xlsx) == 0:
            raise pd.errors.EmptyDataError("Excel file exists but is empty.")
        return pd.read_excel(xlsx)
    if os.path.exists(csv):
        if os.path.getsize(csv) == 0:
            raise pd.errors.EmptyDataError("CSV file exists but is empty.")
        return pd.read_csv(csv)
    raise FileNotFoundError("No dataset found in /data. Please add bctc_final.xlsx or bctc_final.csv.")

def filter_by_ticker_period(df: pd.DataFrame, ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    df2 = df.copy()
    df2["Ticker"] = df2["Ticker"].astype(str).str.upper()
    mask = (df2["Ticker"] == str(ticker).upper()) & (df2["Year"].astype(int).between(int(start_year), int(end_year)))
    return df2.loc[mask].copy()
