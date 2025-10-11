import streamlit as st
from modules.ui_components import app_header, section
from modules.model_utils import load_model_and_assets

def render(state):
    app_header("🧮 Features & Interactions", "Read-only view for deployed model")
    model, scaler, features, threshold, constraints = load_model_and_assets()

    section("📋 Feature list used by the model")
    st.write(f"**Total features:** {len(features)}")
    st.dataframe({"Feature": features})

    section("🧭 Monotonic constraints (if any)")
    if constraints:
        rows = [{"Feature": k, "Constraint": v} for k,v in constraints.items()]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No constraints file found or provided.")
