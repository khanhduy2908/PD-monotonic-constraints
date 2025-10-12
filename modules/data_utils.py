import os
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_master_data() -> pd.DataFrame:
    """
    Load the default master dataset from the data/ folder.
    Priority: Excel > CSV.
    Raises clear errors for missing/empty files.
    """
    xlsx_path = os.path.join("bctc_final.xlsx")
    csv_path = os.path.join("bctc_final.csv")

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

def require_columns(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
