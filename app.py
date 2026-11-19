"""Stable Streamlit entry point with visible startup diagnostics.

If the application fails during import/startup, render the exception in the app
instead of leaving users with a blank page. Do not print secrets or environment values.
"""
import traceback

import streamlit as st

try:
    import app_v4  # noqa: F401 — importing executes the Streamlit application.
except Exception as exc:
    st.error("Nkana Water Treatment Intelligence could not finish starting.")
    st.subheader("Startup diagnostic")
    st.write("The app has captured a startup exception. Share this message with the maintainer; do not share passwords or secret values.")
    st.code(f"{type(exc).__name__}: {exc}")
    with st.expander("Technical traceback"):
        st.code(traceback.format_exc())
    st.info("If this is a deployment/import issue, check the Streamlit Cloud app settings: repository, branch main, main file app.py, and installed requirements.")
