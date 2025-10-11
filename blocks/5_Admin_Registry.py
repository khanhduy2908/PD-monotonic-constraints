import streamlit as st
import json, joblib
from modules.ui_components import app_header, section, info_card
from modules.model_utils import load_model_and_assets

def render(state):
    app_header("🛠️ Admin & Registry", "Quản trị mô hình và artifacts")
    model, scaler, features, threshold, constraints = load_model_and_assets()

    section("📦 Artifacts")
    st.write("- `artifacts/lgbm_model.pkl`")
    st.write("- `artifacts/scaler.pkl`")
    st.write("- `artifacts/features.pkl`")
    st.write("- `artifacts/threshold.json`")
    st.write("- `artifacts/constraints.json`")

    section("ℹ️ Model Params")
    try:
        st.json(model.get_params())
    except Exception:
        st.info("Không thể hiển thị tham số mô hình.")

    section("🔐 Constraints (preview)")
    st.json(constraints if constraints else {"info":"No constraints provided."})

    section("⬇️ Downloads")
    st.download_button("Tải features.pkl", data=joblib.dumps(features), file_name="features.pkl")
    st.download_button("Tải scaler.pkl", data=joblib.dumps(scaler), file_name="scaler.pkl")
    st.download_button("Tải threshold.json", data=json.dumps({"threshold":threshold}).encode("utf-8"),
                       file_name="threshold.json")
