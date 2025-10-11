import streamlit as st
import pandas as pd
from modules.ui_components import app_header, section, info_card
from modules.data_utils import load_master_data, list_tickers, safe_to_csv_download
from modules.viz_utils import plot_default_distribution_year, plot_default_rate_by_sector

def render(state):
    app_header("📦 Data Ingestion", "Upload & validate your master dataset")
    with st.container(border=True):
        st.write("Ứng dụng sẽ nạp dữ liệu mặc định từ **data/bctc_final.xlsx**. "
                 "Bạn có thể tải file mới để tạm thời thay thế (không ghi đè lên ổ đĩa).")
        f = st.file_uploader("Upload Excel (.xlsx) hoặc CSV", type=["xlsx","csv"])
        if "df_master" not in state:
            state["df_master"] = load_master_data()
        if f is not None:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            state["df_master"] = df
            st.success(f"Đã nạp tạm thời: {df.shape[0]:,} dòng, {df.shape[1]:,} cột.")
        st.dataframe(state["df_master"].head(50), use_container_width=True)

    section("📊 Quick EDA")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_default_distribution_year(state["df_master"]), use_container_width=True)
    with c2:
        st.plotly_chart(plot_default_rate_by_sector(state["df_master"]), use_container_width=True)

    section("📥 Download snapshot")
    csv_bytes = safe_to_csv_download(state["df_master"])
    st.download_button("Tải snapshot CSV", data=csv_bytes, file_name="master_snapshot.csv", use_container_width=True)
