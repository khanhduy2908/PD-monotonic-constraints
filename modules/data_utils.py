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
            f"Could not find dataset at {xlsx_path} or {csv_path}. "
            "Please make sure the file exists in the repository."
        )
