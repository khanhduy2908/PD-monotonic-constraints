import pandas as pd
import os
import streamlit as st

@st.cache_data(show_spinner=False)
def load_master_data() -> pd.DataFrame:
    xlsx_path = "data/bctc_final.xlsx"
    csv_path = "data/bctc_final.csv"

    if os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"❌ Không tìm thấy file dữ liệu tại {xlsx_path} hoặc {csv_path}. "
            f"Vui lòng kiểm tra lại repo."
        )

def list_tickers(df: pd.DataFrame):
    return sorted(df["Ticker"].dropna().astype(str).unique())

def get_record_by_ticker_year(df: pd.DataFrame, ticker: str, year: int) -> pd.DataFrame:
    sub = df[(df["Ticker"].astype(str).str.upper() == str(ticker).upper()) & (df["Year"] == int(year))]
    return sub

def has_labels(df: pd.DataFrame) -> bool:
    return "Default" in df.columns

def safe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
