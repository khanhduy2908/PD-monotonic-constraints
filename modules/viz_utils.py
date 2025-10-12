import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# -------- EDA (subset only) --------

def plot_default_distribution_year(df: pd.DataFrame):
    if "Default" not in df.columns or "Year" not in df.columns or len(df) == 0:
        return go.Figure()
    summary = df.groupby(["Year", "Default"]).size().reset_index(name="Count")
    fig = px.bar(summary, x="Year", y="Count", color="Default",
                 barmode="stack", title="Default vs Non-Default by Year (Selected Subset)",
                 template="plotly_white")
    fig.update_layout(legend_title_text="Default")
    return fig

def plot_default_rate_by_sector(df: pd.DataFrame):
    if "Default" not in df.columns or "Sector" not in df.columns or len(df) == 0:
        return go.Figure()
    summary = df.groupby(["Sector", "Default"]).size().unstack(fill_value=0)
    summary.columns = ["No Default", "Default"]
    summary["Total"] = summary.sum(axis=1)
    summary["Default Rate (%)"] = (summary["Default"] / summary["Total"] * 100).round(2)
    summary = summary.reset_index().sort_values("Default Rate (%)", ascending=False)
    fig = px.bar(summary, x="Sector", y="Default Rate (%)",
                 color="Default Rate (%)", template="plotly_white",
                 title="Default Rate by Sector (Selected Subset)")
    fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
    return fig

def plot_probability_histogram(scored_df: pd.DataFrame):
    if "Default_Proba" not in scored_df.columns or len(scored_df) == 0:
        return go.Figure()
    fig = px.histogram(scored_df, x="Default_Proba", nbins=30,
                       template="plotly_white", title="Predicted Default Probability Distribution")
    fig.update_layout(xaxis_title="Default Probability", yaxis_title="Count")
    return fig

# -------- Evaluation (full labeled dataset) --------

def plot_roc_auc_plotly(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      template="plotly_white")
    return fig

def plot_precision_recall_plotly(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall'))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
                      template="plotly_white")
    return fig

# -------- Forecast visuals (selected ticker & period) --------

def plot_pd_line_forecast(forecast_df: pd.DataFrame):
    # expects columns: Year, Default_Proba
    if len(forecast_df) == 0:
        return go.Figure()
    fig = px.line(forecast_df, x="Year", y="Default_Proba", markers=True,
                  template="plotly_white", title="PD Forecast by Year")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig

def plot_pd_risk_bucket_bar(forecast_df: pd.DataFrame, threshold: float):
    # Convert proba to buckets relative to threshold
    if len(forecast_df) == 0:
        return go.Figure()
    df = forecast_df.copy()
    df["Risk_Bucket"] = df["Default_Proba"].apply(lambda p: "High Risk" if p >= threshold else "Low Risk")
    fig = px.bar(df, x="Year", y="Default_Proba", color="Risk_Bucket",
                 template="plotly_white", title="Risk Classification by Year")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig
