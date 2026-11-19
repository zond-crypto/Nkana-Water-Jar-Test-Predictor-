import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

st.set_page_config(page_title="Nkana Water | Jar Test Predictor", page_icon="💧", layout="wide")
st.title("Nkana Water | Jar Test Predictor")
st.caption("Historical jar-test analysis • Alum dose-response • Turbidity decision support")

DB_PATH = "jar_test_predictor.sqlite3"
REQUIRED = ["Raw Turbidity (NTU)", "Raw pH", "Alum Dose (mg/L)", "Treated Turbidity (NTU)"]

def connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, source TEXT,
        raw_turbidity REAL, raw_ph REAL, alum_dose REAL, treated_turbidity REAL,
        treated_ph REAL, imported_at TEXT, record_origin TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, raw_turbidity REAL,
        raw_ph REAL, alum_dose REAL, predicted_turbidity REAL, actual_turbidity REAL)""")
    con.commit()
    return con

def read_observations():
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM observations ORDER BY id", con)

def import_excel(file):
    book = pd.ExcelFile(file)
    if "Jar Test Data" in book.sheet_names:
        df = pd.read_excel(file, sheet_name="Jar Test Data", header=3)
    else:
        df = pd.read_excel(file)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Required columns missing: " + ", ".join(missing))
    for col in REQUIRED + ["Treated pH"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in df.iterrows():
        ph = r.get("Raw pH")
        tph = r.get("Treated pH")
        rows.append((str(r.get("Experiment ID", "IMPORTED")),
            str(r.get("Source Sheet", "Uploaded workbook")), float(r[REQUIRED[0]]),
            None if pd.isna(ph) else float(ph), float(r[REQUIRED[2]]),
            float(r[REQUIRED[3]]), None if pd.isna(tph) else float(tph),
            now, "Historical workbook import"))
    with connect() as con:
        con.executemany("""INSERT INTO observations
        (experiment_id,source,raw_turbidity,raw_ph,alum_dose,treated_turbidity,treated_ph,imported_at,record_origin)
        VALUES (?,?,?,?,?,?,?,?,?)""", rows)
    return len(rows)

with st.sidebar:
    st.header("Historical data")
    st.write("Upload the Nkana Water Intelligent Jar Test Workbook (.xlsx) to import its numeric dose-response observations.")
    upload = st.file_uploader("Historical workbook", type=["xlsx"])
    if upload and st.button("Append workbook observations", type="primary"):
        try:
            n = import_excel(upload)
            st.success(f"Added {n} observations. Avoid importing the same workbook twice.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    st.caption("Hosted local SQLite files may be reset by the hosting platform. Configure a persistent database before operational use.")

data = read_observations()
m1, m2, m3 = st.columns(3)
m1.metric("Dose observations", f"{len(data):,}")
m2.metric("Experiments", int(data.experiment_id.nunique()) if len(data) else 0)
m3.metric("Numeric treated results", int(data.treated_turbidity.notna().sum()) if len(data) else 0)

predict_tab, record_tab, audit_tab = st.tabs(["Predict", "Record real jar test", "History & validation"])
with predict_tab:
    st.subheader("Treated turbidity estimate")
    c1, c2, c3 = st.columns(3)
    raw_t = c1.number_input("Raw turbidity (NTU)", min_value=0.0, value=50.0, step=1.0)
    raw_ph = c2.number_input("Raw-water pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    dose = c3.number_input("Alum dose (mg/L)", min_value=0.0, value=30.0, step=0.5)
    if st.button("Predict treated turbidity", type="primary"):
        usable = data.dropna(subset=["raw_turbidity", "raw_ph", "alum_dose", "treated_turbidity"])
        if len(usable) < 5:
            st.warning("At least five complete historical observations are needed.")
        else:
            cols = ["raw_turbidity", "raw_ph", "alum_dose"]
            X, y = usable[cols], usable.treated_turbidity
            model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
            model.fit(X, y)
            pred = float(model.predict(pd.DataFrame([[raw_t, raw_ph, dose]], columns=cols))[0])
            st.metric("Estimated treated turbidity", f"{max(0, pred):.2f} NTU")
            outside = any(raw < X[col].min() or raw > X[col].max()
                          for raw, col in zip([raw_t, raw_ph, dose], cols))
            if outside:
                st.warning("One or more inputs are outside historical ranges. This is extrapolation; do not rely on it without a confirming jar test.")
            st.info(f"Model fitted on {len(usable)} complete real observations. Predictions are estimates, not dosing instructions.")
            with connect() as con:
                con.execute("INSERT INTO predictions(created_at,raw_turbidity,raw_ph,alum_dose,predicted_turbidity) VALUES(?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), raw_t, raw_ph, dose, pred))

with record_tab:
    st.subheader("Append a measured result")
    with st.form("record"):
        exp = st.text_input("Experiment ID")
        source = st.text_input("Source / location", value="Manual entry")
        a, b, c, d = st.columns(4)
        rt = a.number_input("Raw turbidity (NTU)", min_value=0.0, value=50.0)
        ph = b.number_input("Raw pH", min_value=0.0, max_value=14.0, value=7.0)
        ad = c.number_input("Alum dose (mg/L)", min_value=0.0, value=30.0)
        tt = d.number_input("Treated turbidity (NTU)", min_value=0.0, value=5.0)
        tph = st.number_input("Treated pH (optional; zero means not recorded)", min_value=0.0, max_value=14.0, value=0.0)
        submit = st.form_submit_button("Save measured observation")
    if submit:
        with connect() as con:
            con.execute("""INSERT INTO observations
            (experiment_id,source,raw_turbidity,raw_ph,alum_dose,treated_turbidity,treated_ph,imported_at,record_origin)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (exp or "UNSPECIFIED", source, rt, ph, ad, tt, tph or None,
             datetime.now(timezone.utc).isoformat(), "Manual measured result"))
        st.success("Measured result appended.")
        st.rerun()

with audit_tab:
    st.subheader("Historical record")
    if data.empty:
        st.info("No observations yet. Import the historical workbook.")
    else:
        st.dataframe(data, use_container_width=True)
        st.download_button("Download observations (CSV)", data.to_csv(index=False).encode("utf-8"),
                           file_name="nkana_jar_test_observations.csv", mime="text/csv")
        usable = data.dropna(subset=["raw_turbidity", "raw_ph", "alum_dose", "treated_turbidity"])
        if len(usable) >= 10:
            X = usable[["raw_turbidity", "raw_ph", "alum_dose"]]
            y = usable.treated_turbidity
            cv = KFold(n_splits=min(5, len(usable)), shuffle=True, random_state=42)
            pred = cross_val_predict(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=42), X, y, cv=cv)
            p, q = st.columns(2)
            p.metric("Cross-validated MAE", f"{mean_absolute_error(y, pred):.2f} NTU")
            q.metric("Cross-validated RMSE", f"{mean_squared_error(y, pred)**0.5:.2f} NTU")
        else:
            st.caption("At least ten complete records are needed to show cross-validation metrics.")

st.divider()
st.caption("Decision support only. Verify every candidate dose with a fresh jar test and current plant conditions, alum strength, pH, alkalinity and approved procedures.")
