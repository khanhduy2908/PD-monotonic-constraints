import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def load_master_data(path: str = "data/bctc_final.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    return df

def list_tickers(df: pd.DataFrame):
    return sorted(df["Ticker"].dropna().astype(str).unique())

def get_record_by_ticker_year(df: pd.DataFrame, ticker: str, year: int) -> pd.DataFrame:
    sub = df[(df["Ticker"].astype(str).str.upper() == str(ticker).upper()) & (df["Year"] == int(year))]
    return sub

def has_labels(df: pd.DataFrame) -> bool:
    return "Default" in df.columns

def safe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
