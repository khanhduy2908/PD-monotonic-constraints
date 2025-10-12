import streamlit as st
import pandas as pd
from modules.data_utils import load_master_data
from modules.model_utils import load_model_and_assets, predict_proba_single
from modules.viz_utils import plot_roc_auc_plotly, plot_precision_recall_plotly

st.set_page_config(
    page_title="Corporate Default Risk Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== HEADER =====
st.title("Corporate Default Risk Scoring System")
st.markdown("""
This internal tool provides real-time corporate default risk scoring using a pre-trained LightGBM model. 
It is designed for financial institutions to assess firm-level probability of default.
""")

# ===== DATA SECTION =====
st.subheader("Master Dataset")
try:
    df_master = load_master_data()
    st.dataframe(df_master.head(50), use_container_width=True)
    st.caption(f"Dataset loaded successfully: {df_master.shape[0]} rows × {df_master.shape[1]} columns")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.subheader("Exploratory Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_default_distribution_year(df_master), use_container_width=True)
with col2:
    st.plotly_chart(plot_default_rate_by_sector(df_master), use_container_width=True)

# ===== MODEL SECTION =====
st.subheader("Scoring Interface")

model, scaler, feature_list, threshold = load_model_and_assets()

col1, col2 = st.columns([1, 2])

with col1:
    firm_code = st.text_input("Enter Firm Code (e.g. ABC123)")
    score_btn = st.button("Run Scoring")

with col2:
    if score_btn:
        if firm_code not in df_master['firm_code'].values:
            st.error("Firm code not found in dataset.")
        else:
            x_input = df_master[df_master['firm_code'] == firm_code][feature_list]
            proba, label = predict_proba_single(model, scaler, x_input, threshold)
            st.metric("Default Probability", f"{proba:.2%}")
            st.metric("Predicted Class", "Default" if label == 1 else "Non-Default")

# ===== EVALUATION SECTION =====
st.subheader("Model Evaluation Dashboard")

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(plot_roc_auc_plotly(model, df_master[feature_list], df_master['default']), use_container_width=True)
with col_b:
    st.plotly_chart(plot_precision_recall_plotly(model, df_master[feature_list], df_master['default']), use_container_width=True)

st.caption("Evaluation charts are based on the full dataset and pre-trained model.")
