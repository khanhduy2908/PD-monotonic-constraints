import streamlit as st

def app_header(title: str, subtitle: str = None):
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"**{subtitle}**")

def status_badge(text: str, color: str = "blue"):
    st.markdown(
        f'<span style="background:{color};color:white;padding:4px 8px;border-radius:8px;font-size:12px">{text}</span>',
        unsafe_allow_html=True,
    )

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for i,(k,v) in enumerate(metrics.items()):
        cols[i].metric(k, v)

def section(title: str):
    st.markdown(f"### {title}")

def info_card(title: str, body: str):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(body)
