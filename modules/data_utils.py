import os
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_master_data() -> pd.DataFrame:
    """
    Load default dataset from data/ (Excel preferred; fallback to CSV).
    Raises explicit errors for missing/empty files.
    """
    xlsx_path = os.path.join("bctc_final.xlsx")
    csv_path  = os.path.join("bctc_final.csv")

    if os.path.exists(xlsx_path):
        if os.path.getsize(xlsx_path) == 0:
            raise pd.errors.EmptyDataError("Excel file exists but is empty.")
        return pd.read_excel(xlsx_path)

    if os.path.exists(csv_path):
        if os.path.getsize(csv_path) == 0:
            raise pd.errors.EmptyDataError("CSV file exists but is empty.")
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "Could not find dataset in data/. Please add bctc_final.xlsx or bctc_final.csv to the repository."
    )

def filter_by_ticker_period(df: pd.DataFrame, ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    df2 = df.copy()
    df2["Ticker"] = df2["Ticker"].astype(str).str.upper()
    out = df2[(df2["Ticker"] == str(ticker).upper()) & (df2["Year"].astype(int).between(int(start_year), int(end_year)))]
    return out
