from modules.data_utils import load_master_data

def render(state):
    app_header("Data Ingestion", "Upload & validate your master dataset")
    with st.container(border=True):
        st.write("Ứng dụng sẽ nạp dữ liệu mặc định từ **data/bctc_final.xlsx**. "
                 "Bạn có thể tải file mới để tạm thời thay thế (không ghi đè lên ổ đĩa).")
        f = st.file_uploader("Upload Excel (.xlsx) hoặc CSV", type=["xlsx","csv"])

        if "df_master" not in state or state["df_master"] is None:
            try:
                state["df_master"] = load_master_data()
            except Exception:
                state["df_master"] = None

        if f is not None:
            import pandas as pd
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            state["df_master"] = df
            st.success(f"Đã nạp tạm thời: {df.shape[0]:,} dòng, {df.shape[1]:,} cột.")

        if state["df_master"] is not None:
            st.dataframe(state["df_master"].head(50), use_container_width=True)
        else:
            st.warning("⚠️ Chưa có dữ liệu. Vui lòng tải file lên để bắt đầu.")
