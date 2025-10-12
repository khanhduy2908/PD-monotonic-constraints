import streamlit as st

def page_title(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.markdown(subtitle)

def section_header(title: str):
    st.markdown(f"### {title}")

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, value)

def info_box(msg: str, kind: str = "info"):
    if kind == "info":
        st.info(msg)
    elif kind == "warning":
        st.warning(msg)
    elif kind == "error":
        st.error(msg)
    else:
        st.write(msg)
