import streamlit as st
from modules.ui_components import app_header
from modules.data_utils import load_master_data

# Lazy import for blocks
PAGES = {
    "Data Ingestion": "blocks.1_Data_Ingestion",
    "Features & Interactions": "blocks.2_Features_&_Interactions",
    "Scoring": "blocks.3_Scoring",
    "Evaluation Dashboard": "blocks.4_Evaluation_Dashboard",
    "Admin & Registry": "blocks.5_Admin_Registry",
}

st.set_page_config(page_title="Risk App — Default Prediction", layout="wide")

# Shared state
if "state" not in st.session_state:
    st.session_state["state"] = {}
state = st.session_state["state"]

with st.sidebar:
    st.image("https://static.streamlit.io/examples/dice.jpg", width=96)
    st.markdown("## Navigation")
    choice = st.radio("Go to", list(PAGES.keys()), index=0)

# Ensure default data loaded once
if "df_master" not in state:
    try:
        state["df_master"] = load_master_data()
    except Exception:
        state["df_master"] = None

# Dynamic loader
module_path = PAGES[choice]
mod = __import__(module_path, fromlist=["render"])
mod.render(state)
