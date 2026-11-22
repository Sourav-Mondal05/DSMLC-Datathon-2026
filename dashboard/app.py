"""Invisible Oil Loss Detector — Streamlit multipage application."""
import streamlit as st

st.set_page_config(
    page_title="Invisible Oil Loss Detector",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/1_Overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/2_Predictive_Maintenance.py", title="Predictive Maintenance", icon=":material/monitor_heart:"),
    st.Page("pages/3_Production_Analysis.py", title="Production Analysis", icon=":material/show_chart:"),
    st.Page("pages/4_Operational_Optimization.py", title="Operational Optimization", icon=":material/tune:"),
    st.Page("pages/5_Maintenance_Priorities.py", title="Maintenance Priorities", icon=":material/assignment_turned_in:"),
    st.Page("pages/6_Well_Details.py", title="Well Details", icon=":material/oil_barrel:"),
]
# Hide built-in sidebar navigation. Every page renders the same horizontal nav.
st.navigation(pages, position="hidden").run()
