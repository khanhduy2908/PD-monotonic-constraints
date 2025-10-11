import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_default_distribution_year(df: pd.DataFrame):
    if "Default" not in df.columns:
        return go.Figure()
    summary = df.groupby(["Year","Default"]).size().reset_index(name="Count")
    fig = px.bar(summary, x="Year", y="Count", color="Default",
                 barmode="stack", title="Number of Companies by Default Status per Year",
                 template="plotly_white")
    fig.update_layout(legend_title_text="Default")
    return fig

def plot_default_rate_by_sector(df: pd.DataFrame):
    if "Default" not in df.columns or "Sector" not in df.columns:
        return go.Figure()
    summary = df.groupby(["Sector","Default"]).size().unstack(fill_value=0)
    summary.columns = ["No Default","Default"]
    summary["Total"] = summary.sum(axis=1)
    summary["Default Rate (%)"] = (summary["Default"]/summary["Total"]*100).round(2)
    summary = summary.reset_index().sort_values("Default Rate (%)", ascending=False)
    fig = px.bar(summary, x="Sector", y="Default Rate (%)", color="Default Rate (%)",
                 template="plotly_white", title="Default Rate by Sector")
    fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
    return fig

def plot_probability_histogram(scored_df: pd.DataFrame):
    if "Default_Proba" not in scored_df.columns:
        return go.Figure()
    fig = px.histogram(scored_df, x="Default_Proba", nbins=40,
                       template="plotly_white", title="Predicted Default Probability Distribution")
    return fig

def shap_bar_figure(importances: pd.Series, topk: int = 12):
    s = importances.sort_values(key=lambda x: x.abs(), ascending=False)[:topk]
    fig = px.bar(x=s.values, y=s.index, orientation="h", template="plotly_white",
                 title="Top Feature Contributions (approx)", labels={"x":"|importance|","y":"Feature"})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig
