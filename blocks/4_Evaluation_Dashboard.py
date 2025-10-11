import streamlit as st
import pandas as pd
from modules.ui_components import app_header, section
from modules.model_utils import load_model_and_assets, score_dataframe
from modules.viz_utils import plot_default_distribution_year, plot_default_rate_by_sector, plot_probability_histogram

def render(state):
    app_header("📈 Evaluation Dashboard", "Đánh giá nhanh toàn tập dữ liệu")
    df_master = state.get("df_master")
    if df_master is None or df_master.empty:
        st.warning("Chưa có dữ liệu. Vào mục **Data Ingestion** để nạp dữ liệu trước.")
        return

    model, scaler, features, threshold, constraints = load_model_and_assets()

    st.write("Hệ thống sẽ **chấm điểm toàn bộ dataset hiện tại** để xem phân phối xác suất & các chỉ số tổng quan.")
    if st.button("Chấm điểm toàn bộ dataset", use_container_width=True):
        scored = score_dataframe(model, df_master, scaler, features, threshold)
        state["scored_df"] = scored
        st.success(f"Đã chấm điểm {len(scored):,} bản ghi.")

    scored = state.get("scored_df")
    if scored is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_probability_histogram(scored), use_container_width=True)
        with c2:
            if "Default" in scored.columns:
                st.plotly_chart(plot_default_distribution_year(scored), use_container_width=True)
            else:
                st.info("Dataset không có nhãn 'Default' để vẽ biểu đồ theo năm.")

        st.dataframe(scored.head(50), use_container_width=True)
        st.download_button("Tải predictions.csv",
                           data=scored.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv",
                           use_container_width=True)
    else:
        st.info("Nhấn nút **Chấm điểm toàn bộ dataset** để tạo dashboard.")
