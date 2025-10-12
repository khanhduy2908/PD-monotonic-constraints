import pandas as pd
import streamlit as st
import os

@st.cache_data(show_spinner=False)
def load_master_data(path_xlsx="data/bctc_final.xlsx", path_csv="data/bctc_final.csv") -> pd.DataFrame:
    # Ưu tiên xlsx, fallback sang csv
    if os.path.exists(path_xlsx):
        return pd.read_excel(path_xlsx)
    elif os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    else:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {path_xlsx} hoặc {path_csv}")


def list_tickers(df: pd.DataFrame):
    return sorted(df["Ticker"].dropna().astype(str).unique())

def get_record_by_ticker_year(df: pd.DataFrame, ticker: str, year: int) -> pd.DataFrame:
    sub = df[(df["Ticker"].astype(str).str.upper() == str(ticker).upper()) & (df["Year"] == int(year))]
    return sub

def has_labels(df: pd.DataFrame) -> bool:
    return "Default" in df.columns

def safe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
