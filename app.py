from datetime import datetime, timezone
from uuid import uuid4
import hmac

import pandas as pd
import streamlit as st
from supabase import Client, create_client
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

st.set_page_config(page_title="Nkana Water | Jar Test Predictor", page_icon="💧", layout="wide")
st.title("Nkana Water | Jar Test Predictor")

st.caption("Historical jar-test analysis • Alum dose-response • Turbidity decision support")

# Basic shared-password gate: protect the public app while the deployment is being hardened.
app_password = st.secrets.get("APP_PASSWORD", "")
if not app_password:
    st.error("App access is locked. Add APP_PASSWORD in Streamlit Secrets.")
    st.stop()
if not st.session_state.get("authenticated", False):
    with st.form("login"):
        st.subheader("Laboratory access")
        entered = st.text_input("Application password", type="password")
        login = st.form_submit_button("Sign in")
    if login:
        if hmac.compare_digest(entered, app_password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
with st.sidebar:
    if st.button("Sign out"):
        st.session_state["authenticated"] = False
        st.rerun()

REQUIRED = ["Raw Turbidity (NTU)", "Raw pH", "Alum Dose (mg/L)", "Treated Turbidity (NTU)"]

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError("Supabase credentials are not configured in Streamlit Secrets.")
    return create_client(url, key)

def db_ready():
    try:
        get_supabase()
        return True
    except Exception as exc:
        st.error(f"Database connection is not configured or failed: {exc}")
        return False

def load_observations():
    client = get_supabase()
    experiments = client.table("experiments").select(
        "id,experiment_code,source_location,water_source,raw_turbidity_ntu,raw_ph,experiment_date"
    ).execute().data or []
    results = client.table("jar_test_results").select(
        "id,experiment_id,dose_mg_l,treated_turbidity_ntu,treated_ph,result_origin,created_at"
    ).execute().data or []
    if not experiments or not results:
        return pd.DataFrame(columns=[
            "id", "experiment_id", "source", "raw_turbidity", "raw_ph",
            "alum_dose", "treated_turbidity", "treated_ph", "record_origin"
        ])
    exp_df = pd.DataFrame(experiments)
    result_df = pd.DataFrame(results)
    merged = result_df.merge(exp_df, left_on="experiment_id", right_on="id", how="left", suffixes=("", "_experiment"))
    return pd.DataFrame({
        "id": merged["id"],
        "experiment_id": merged["experiment_code"],
        "source": merged["source_location"].fillna(merged["water_source"]),
        "raw_turbidity": merged["raw_turbidity_ntu"],
        "raw_ph": merged["raw_ph"],
        "alum_dose": merged["dose_mg_l"],
        "treated_turbidity": merged["treated_turbidity_ntu"],
        "treated_ph": merged["treated_ph"],
        "record_origin": merged["result_origin"]
    })

def save_observation(experiment_code, source, raw_turbidity, raw_ph, dose, treated_turbidity,
                     treated_ph=None, origin="measured"):
    client = get_supabase()
    code = experiment_code.strip() or f"JT-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:5].upper()}"
    existing = client.table("experiments").select("id").eq("experiment_code", code).limit(1).execute().data or []
    if existing:
        exp_id = existing[0]["id"]
    else:
        created = client.table("experiments").insert({
            "experiment_code": code,
            "source_location": source,
            "water_source": source,
            "raw_turbidity_ntu": float(raw_turbidity),
            "raw_ph": float(raw_ph),
            "coagulant_name": "Aluminium sulphate"
        }).execute().data
        exp_id = created[0]["id"]
    client.table("jar_test_results").upsert({
        "experiment_id": exp_id,
        "dose_mg_l": float(dose),
        "treated_turbidity_ntu": float(treated_turbidity),
        "treated_ph": float(treated_ph) if treated_ph is not None else None,
        "result_origin": origin
    }, on_conflict="experiment_id,dose_mg_l").execute()
    return code

def import_excel(file):
    book = pd.ExcelFile(file)
    sheet = "Jar Test Data" if "Jar Test Data" in book.sheet_names else book.sheet_names[0]
    df = pd.read_excel(file, sheet_name=sheet, header=3 if sheet == "Jar Test Data" else 0)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Required columns missing: " + ", ".join(missing))
    for col in REQUIRED + ["Treated pH"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    count = 0
    for i, row in df.iterrows():
        exp_code = str(row.get("Experiment ID", "")).strip()
        if not exp_code or exp_code.lower() == "nan":
            exp_code = f"IMPORT-{i+1:04d}"
        source = str(row.get("Source Sheet", "Historical workbook"))
        tph = row.get("Treated pH")
        save_observation(exp_code, source, row[REQUIRED[0]], row[REQUIRED[1]],
                         row[REQUIRED[2]], row[REQUIRED[3]],
                         None if pd.isna(tph) else tph, "imported")
        count += 1
    return count

st.sidebar.header("Database")
st.sidebar.caption("Connected storage: Supabase PostgreSQL")
if db_ready():
    st.sidebar.success("Supabase connection configured")
else:
    st.sidebar.warning("Add the two Streamlit Secrets before using database features.")

with st.sidebar:
    st.header("Historical data")
    st.write("Upload the Nkana Water Intelligent Jar Test Workbook (.xlsx). Existing experiment/dose pairs are updated rather than duplicated.")
    upload = st.file_uploader("Historical workbook", type=["xlsx"])
    if upload and st.button("Import historical observations", type="primary", disabled=not db_ready()):
        try:
            n = import_excel(upload)
            st.success(f"Imported or updated {n} dose observations.")
            st.rerun()
        except Exception as exc:
            st.error(f"Import failed: {exc}")

try:
    data = load_observations()
    db_error = None
except Exception as exc:
    data = pd.DataFrame(columns=[
        "id", "experiment_id", "source", "raw_turbidity", "raw_ph",
        "alum_dose", "treated_turbidity", "treated_ph", "record_origin"
    ])
    db_error = str(exc)

if db_error:
    st.error(f"Could not read the Supabase tables: {db_error}")

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
            outside = any(value < X[col].min() or value > X[col].max()
                          for value, col in zip([raw_t, raw_ph, dose], cols))
            if outside:
                st.warning("One or more inputs are outside historical ranges. This is extrapolation; confirm with a jar test.")
            st.info(f"Model fitted on {len(usable)} complete real/imported observations. This is decision support, not a dosing instruction.")
            get_supabase().table("predictions").insert({
                "raw_turbidity_ntu": raw_t, "raw_ph": raw_ph, "dose_mg_l": dose,
                "predicted_turbidity_ntu": pred, "model_name": "ExtraTreesRegressor",
                "model_version": "baseline-1", "input_within_training_range": not outside
            }).execute()

with record_tab:
    st.subheader("Append a measured result")
    with st.form("record"):
        exp = st.text_input("Experiment ID (leave blank to generate one)")
        source = st.text_input("Source / location", value="Manual entry")
        a, b, c, d = st.columns(4)
        rt = a.number_input("Raw turbidity (NTU)", min_value=0.0, value=50.0)
        ph = b.number_input("Raw pH", min_value=0.0, max_value=14.0, value=7.0)
        ad = c.number_input("Alum dose (mg/L)", min_value=0.0, value=30.0)
        tt = d.number_input("Treated turbidity (NTU)", min_value=0.0, value=5.0)
        tph = st.number_input("Treated pH (optional; zero means not recorded)", min_value=0.0, max_value=14.0, value=0.0)
        submit = st.form_submit_button("Save measured observation", disabled=not db_ready())
    if submit:
        try:
            code = save_observation(exp, source, rt, ph, ad, tt, tph or None, "measured")
            st.success(f"Saved to Supabase under experiment {code}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

with audit_tab:
    st.subheader("Historical record")
    if data.empty:
        st.info("No observations yet. Import the historical workbook or record a measured jar test.")
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
st.caption("Decision support only. Verify candidate doses with a fresh jar test and current plant conditions, alum strength, pH, alkalinity and approved procedures.")
