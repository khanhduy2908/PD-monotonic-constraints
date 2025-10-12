import streamlit as st
from modules.ui_components import app_header, section
from modules.data_utils import load_master_data
from modules.viz_utils import plot_default_distribution_year, plot_default_rate_by_sector

def render(state):
    app_header("Data Ingestion", "Default data only — no manual upload")

    # Chỉ load file trong repo
    if "df_master" not in state:
        try:
            state["df_master"] = load_master_data()
        except FileNotFoundError as e:
            st.error(str(e))
            return

    df = state["df_master"]
    st.success(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    st.dataframe(df.head(50), use_container_width=True)

    section("Quick EDA")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_default_distribution_year(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_default_rate_by_sector(df), use_container_width=True)
