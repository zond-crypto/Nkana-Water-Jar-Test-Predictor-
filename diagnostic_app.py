"""Isolated Streamlit frontend diagnostic.

Run this as a SECOND Streamlit Community Cloud app using this file as the
entrypoint. It deliberately imports no Nkana application modules, reads no
secrets, and makes no Supabase/database calls.
"""
from datetime import datetime, timezone
import streamlit as st

st.set_page_config(
    page_title="Nkana Streamlit Diagnostic",
    page_icon="🧪",
    layout="centered",
)

st.title("Nkana Streamlit Diagnostic")
st.success("SUCCESS: Streamlit rendered this page.")
st.write("If you can read this, the Streamlit frontend is rendering.")
st.write("This diagnostic does not connect to Supabase or load the Nkana app.")
st.divider()

st.subheader("Runtime checks")
st.write("Python execution: **PASS**")
st.write(f"UTC server time: {datetime.now(timezone.utc).isoformat()}")
st.write(f"Streamlit version: **{st.__version__}**")

value = st.text_input("Frontend interaction test", placeholder="Type any short word")
if value:
    st.info(f"Interaction received: {value}")

if st.button("Test rerun"):
    st.success("Button interaction and rerun: PASS")

st.caption("Diagnostic only — no credentials, laboratory records, or database writes.")
