import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ---------- Data Exploration ----------

def plot_default_distribution_year(df: pd.DataFrame):
    if "Default" not in df.columns or "Year" not in df.columns:
        return go.Figure()
    summary = df.groupby(["Year", "Default"]).size().reset_index(name="Count")
    fig = px.bar(
        summary, x="Year", y="Count", color="Default",
        barmode="stack", title="Number of Companies by Default Status per Year",
        template="plotly_white"
    )
    fig.update_layout(legend_title_text="Default")
    return fig

def plot_default_rate_by_sector(df: pd.DataFrame):
    if "Default" not in df.columns or "Sector" not in df.columns:
        return go.Figure()
    summary = df.groupby(["Sector", "Default"]).size().unstack(fill_value=0)
    summary.columns = ["No Default", "Default"]
    summary["Total"] = summary.sum(axis=1)
    summary["Default Rate (%)"] = (summary["Default"] / summary["Total"] * 100).round(2)
    summary = summary.reset_index().sort_values("Default Rate (%)", ascending=False)
    fig = px.bar(
        summary, x="Sector", y="Default Rate (%)",
        color="Default Rate (%)", template="plotly_white",
        title="Default Rate by Sector"
    )
    fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
    return fig

def plot_probability_histogram(scored_df: pd.DataFrame):
    if "Default_Proba" not in scored_df.columns:
        return go.Figure()
    fig = px.histogram(
        scored_df, x="Default_Proba", nbins=40,
        template="plotly_white", title="Predicted Default Probability Distribution"
    )
    fig.update_layout(xaxis_title="Default Probability", yaxis_title="Count")
    return fig

# ---------- Model Evaluation ----------

def plot_roc_auc_plotly(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC={roc_auc:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"
    )
    return fig

def plot_precision_recall_plotly(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall"))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white"
    )
    return fig
