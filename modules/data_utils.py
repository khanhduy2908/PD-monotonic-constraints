import pandas as pd
import os
import streamlit as st

@st.cache_data
def load_master_data():
    csv_path = os.path.join("data", "bctc_final.csv")
    xlsx_path = os.path.join("data", "bctc_final.xlsx")

    if os.path.exists(csv_path):
        if os.path.getsize(csv_path) == 0:
            raise FileNotFoundError("CSV file found but it's empty. Please upload a valid dataset.")
        return pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(
            "No default dataset found in data/. Please upload bctc_final.xlsx or bctc_final.csv to the repository."
        )
