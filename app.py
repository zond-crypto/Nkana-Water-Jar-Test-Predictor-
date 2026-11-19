from __future__ import annotations

from datetime import datetime, timezone
import hmac
import io
import math
import uuid

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from supabase import Client, create_client
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict

st.set_page_config(page_title="Nkana Water | Jar Test Intelligence", page_icon="💧", layout="wide")
st.title("Nkana Water | Jar Test Intelligence")
st.caption("Historical jar-test learning • Alum dose optimisation • Dose-response simulation • Plant chemical planning")

# Shared-password gate. Keep APP_PASSWORD and SUPABASE_SERVICE_ROLE_KEY in Streamlit Secrets only.
app_password = st.secrets.get("APP_PASSWORD", "")
if not app_password:
    st.error("Access is locked. Add APP_PASSWORD to Streamlit Secrets.")
    st.stop()
if not st.session_state.get("authenticated", False):
    with st.form("login"):
        st.subheader("Laboratory access")
        entered = st.text_input("Application password", type="password")
        if st.form_submit_button("Sign in"):
            if hmac.compare_digest(entered, app_password):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()
with st.sidebar:
    if st.button("Sign out"):
        st.session_state["authenticated"] = False
        st.rerun()

@st.cache_resource
def get_db() -> Client:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit Secrets.")
    return create_client(url, key)

def db_status():
    try:
        get_db()
        return True
    except Exception as exc:
        st.sidebar.error(f"Database unavailable: {exc}")
        return False

def load_data():
    db = get_db()
    exps = db.table("experiments").select(
        "id,experiment_code,experiment_date,source_location,water_source,raw_turbidity_ntu,raw_ph,raw_temperature_c,raw_alkalinity_mg_l_caco3,plant_flow_m3_day,operating_hours_per_day,shift_length_hours,alum_bag_mass_kg,notes"
    ).order("experiment_date", desc=True).execute().data or []
    jars = db.table("jar_test_results").select(
        "id,experiment_id,dose_mg_l,treated_turbidity_ntu,treated_ph,residual_aluminium_mg_l,settling_time_min,observation_notes,result_origin,created_at"
    ).execute().data or []
    if not exps or not jars:
        return pd.DataFrame()
    a, b = pd.DataFrame(exps), pd.DataFrame(jars)
    d = b.merge(a, left_on="experiment_id", right_on="id", suffixes=("_jar", "_exp"))
    d = d.rename(columns={
        "experiment_code":"experiment_id", "experiment_date":"date",
        "source_location":"source", "raw_turbidity_ntu":"raw_turbidity",
        "raw_ph":"raw_ph", "raw_temperature_c":"temperature",
        "raw_alkalinity_mg_l_caco3":"alkalinity", "dose_mg_l":"dose",
        "treated_turbidity_ntu":"treated_turbidity", "treated_ph":"treated_ph",
        "result_origin":"origin", "plant_flow_m3_day":"flow",
        "operating_hours_per_day":"operating_hours", "shift_length_hours":"shift_hours",
        "alum_bag_mass_kg":"bag_mass"
    })
    keep = ["id_jar","experiment_id","date","source","water_source","raw_turbidity","raw_ph",
            "temperature","alkalinity","dose","treated_turbidity","treated_ph","residual_aluminium_mg_l",
            "settling_time_min","observation_notes","origin","flow","operating_hours","shift_hours","bag_mass","notes"]
    return d[[c for c in keep if c in d.columns]].rename(columns={"id_jar":"result_id"})

def add_experiment(code, date, source, raw, ph, jars, temperature=None, alkalinity=None,
                   flow=None, operating_hours=None, shift_hours=None, bag_mass=50.0, notes=""):
    db = get_db()
    code = code.strip() or f"JT-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:5].upper()}"
    exists = db.table("experiments").select("id").eq("experiment_code", code).limit(1).execute().data or []
    if exists:
        raise ValueError(f"Experiment ID {code} already exists. Use a new ID; records are not silently overwritten.")
    payload = {
        "experiment_code":code, "experiment_date":date.isoformat() if hasattr(date,"isoformat") else str(date),
        "source_location":source, "water_source":source, "raw_turbidity_ntu":float(raw),
        "raw_ph":float(ph) if ph is not None else None,
        "raw_temperature_c":float(temperature) if temperature is not None else None,
        "raw_alkalinity_mg_l_caco3":float(alkalinity) if alkalinity is not None else None,
        "plant_flow_m3_day":float(flow) if flow is not None else None,
        "operating_hours_per_day":float(operating_hours) if operating_hours is not None else None,
        "shift_length_hours":float(shift_hours) if shift_hours is not None else None,
        "alum_bag_mass_kg":float(bag_mass) if bag_mass is not None else None,
        "notes":notes
    }
    created = db.table("experiments").insert(payload).execute().data
    exp_id = created[0]["id"]
    rows = []
    for dose, turb, tph in jars:
        rows.append({"experiment_id":exp_id,"dose_mg_l":float(dose),
                     "treated_turbidity_ntu":float(turb),
                     "treated_ph":float(tph) if tph is not None else None,
                     "result_origin":"measured"})
    db.table("jar_test_results").insert(rows).execute()
    return code

def interpolate_dose(group, target):
    j = group.dropna(subset=["dose","treated_turbidity"]).sort_values("dose")
    j = j[j["treated_turbidity"] > 0]
    if len(j) < 2:
        return None
    # Return the lowest tested/interpolated dose that crosses the settled turbidity target.
    for i in range(len(j)):
        row = j.iloc[i]
        if row.treated_turbidity <= target:
            if i == 0:
                return float(row.dose)
            prev = j.iloc[i-1]
            if prev.treated_turbidity == row.treated_turbidity:
                return float(row.dose)
            frac = (float(prev.treated_turbidity)-target)/(float(prev.treated_turbidity)-float(row.treated_turbidity))
            return float(prev.dose + frac*(row.dose-prev.dose))
    return None

def dose_training(data, target, year_from):
    real = data[data.origin.isin(["measured","imported"])].copy()
    real = real[real.date.notna()]
    real["year"] = pd.to_datetime(real.date, errors="coerce").dt.year
    real = real[real.year >= year_from]
    rows=[]
    for code,g in real.groupby("experiment_id"):
        raw = pd.to_numeric(g.raw_turbidity, errors="coerce").dropna()
        if raw.empty: continue
        d = interpolate_dose(g, target)
        if d is not None and d > 0:
            rows.append({"experiment_id":code,"date":g.date.iloc[0],"raw":float(raw.iloc[0]),
                         "dose":d,"selected":pd.to_numeric(g.get("coagulant_strength_mg_l",pd.Series(dtype=float)),errors="coerce").dropna().iloc[0] if "coagulant_strength_mg_l" in g and pd.to_numeric(g.coagulant_strength_mg_l,errors="coerce").notna().any() else np.nan})
    return pd.DataFrame(rows)

def fit_powerlaw(train):
    x=np.log(train.raw.to_numpy(float)); y=np.log(train.dose.to_numpy(float))
    n=len(x)
    if n<5: return None
    xb=x.mean(); yb=y.mean(); sxx=np.sum((x-xb)**2)
    if sxx<=1e-12: return None
    slope=np.sum((x-xb)*(y-yb))/sxx
    intercept=yb-slope*xb
    resid=y-(intercept+slope*x)
    sigma=float(np.sqrt(np.sum(resid**2)/max(n-2,1)))
    r2=float(1-np.sum(resid**2)/np.sum((y-yb)**2)) if np.sum((y-yb)**2)>0 else 0.0
    return {"a":intercept,"b":slope,"sigma":sigma,"xb":xb,"sxx":sxx,"n":n,
            "r2":r2,"xmin":float(train.raw.min()),"xmax":float(train.raw.max()),
            "rmse_log":float(np.sqrt(np.mean(resid**2)))}

def predict_dose(model, raw):
    x=math.log(raw)
    mean=model["a"]+model["b"]*x
    # Approximate 95% prediction interval; intentionally conservative for small historical datasets.
    se=model["sigma"]*math.sqrt(1+1/model["n"]+(x-model["xb"])**2/model["sxx"])
    margin=2.0*se
    return (math.exp(mean), math.exp(mean-margin), math.exp(mean+margin))

def bootstrap_required_doses(train, raw, n=1000, seed=42):
    rng=np.random.default_rng(seed)
    values=[]
    if len(train)<5: return np.array(values)
    for _ in range(n):
        sample=train.iloc[rng.integers(0,len(train),len(train))].reset_index(drop=True)
        m=fit_powerlaw(sample)
        if m is None: continue
        try:
            value=predict_dose(m,raw)[0]
            if np.isfinite(value) and value>0: values.append(value)
        except Exception:
            continue
    return np.asarray(values,dtype=float)

def loo_dose(train):
    if len(train)<6: return None
    errors=[]
    for i in range(len(train)):
        sub=train.drop(train.index[i])
        m=fit_powerlaw(sub)
        if not m: continue
        pred=predict_dose(m,float(train.iloc[i].raw))[0]
        actual=float(train.iloc[i].dose)
        errors.append(abs(pred-actual)/actual)
    return float(np.mean(errors)) if errors else None

def fit_hill(group):
    j=group.dropna(subset=["dose","treated_turbidity"])
    j=j[(j.dose>0)&(j.treated_turbidity>0)].sort_values("dose")
    if len(j)<4: return None
    raw=float(j.raw_turbidity.iloc[0]); ds=j.dose.to_numpy(float); ys=j.treated_turbidity.to_numpy(float)
    floor=max(0.05,min(float(ys.min())*0.8, float(raw)*0.25))
    best=None
    # Search a simple monotonic Hill curve. Fit uses real jars only.
    for f in np.linspace(0, min(float(ys.min())*0.8, raw*0.3), 5):
        for d50 in np.geomspace(max(ds.min()*0.35,0.1),ds.max()*3,35):
            for h in np.arange(1.0,8.1,0.5):
                pred=f+(raw-f)/(1+(ds/d50)**h)
                loss=float(np.mean((np.log(np.maximum(ys,0.01))-np.log(np.maximum(pred,0.01)))**2))
                if best is None or loss<best[0]: best=(loss,f,d50,h)
    return {"raw":raw,"floor":best[1],"d50":best[2],"hill":best[3],"rmse":math.sqrt(best[0]),"experiment_id":str(group.experiment_id.iloc[0])}

def hill_curve(raw,dose,floor,d50,h):
    return floor+(raw-floor)/(1+(dose/d50)**h)

def import_historical_workbook(uploaded):
    """
    Import the original NWSC multi-sheet workbook OR a normalized flat workbook.
    Original workbook layout is read by cell position to preserve its actual values:
      C2 = experiment date; B5 = raw turbidity; B6 = raw pH;
      rows 11 onward: A = dose, B = treated turbidity, C = treated pH.
    Missing '*' results are stored as NULL, never converted to zero.
    """
    book = pd.ExcelFile(uploaded)
    db = get_db()
    inserted = 0
    created_experiments = 0
    skipped_sheets = []

    def num(value):
        if value is None or (isinstance(value, str) and value.strip() in {"", "*", "-", "—", "N/A", "NA"}):
            return None
        try:
            result = float(str(value).replace(",", "").strip())
            return result if np.isfinite(result) else None
        except Exception:
            return None

    for sheet in book.sheet_names:
        frame = pd.read_excel(uploaded, sheet_name=sheet, header=None, dtype=object)
        if frame.empty or len(frame.index) < 6:
            continue

        # The source workbook has no flat header row. Read its fixed worksheet layout.
        raw = num(frame.iat[4, 1]) if frame.shape[0] > 4 and frame.shape[1] > 1 else None
        raw_ph = num(frame.iat[5, 1]) if frame.shape[0] > 5 and frame.shape[1] > 1 else None
        date_raw = frame.iat[1, 2] if frame.shape[0] > 1 and frame.shape[1] > 2 else None

        # Also accept a normalized sheet with explicit column names.
        if raw is None:
            for header_row in range(min(6, len(frame.index))):
                headers = [str(v).strip().lower() if v is not None else "" for v in frame.iloc[header_row].tolist()]
                lookup = {name: i for i, name in enumerate(headers)}
                def locate(*names):
                    for name in names:
                        if name.lower() in lookup:
                            return lookup[name.lower()]
                    return None
                raw_i = locate("raw turbidity (ntu)", "raw turbidity", "raw ntu")
                dose_i = locate("alum dose (mg/l)", "alum dose", "dose (mg/l)", "dose")
                turb_i = locate("treated turbidity (ntu)", "treated turbidity", "settled turbidity (ntu)")
                if raw_i is not None and dose_i is not None and turb_i is not None:
                    records = []
                    ph_i = locate("raw ph", "raw-water ph", "raw water ph")
                    tph_i = locate("treated ph", "settled ph")
                    date_i = locate("date", "experiment date", "test date")
                    for row_i in range(header_row + 1, len(frame.index)):
                        row = frame.iloc[row_i]
                        dose_v = num(row.iloc[dose_i]) if dose_i < len(row) else None
                        turb_v = num(row.iloc[turb_i]) if turb_i < len(row) else None
                        if dose_v is None:
                            continue
                        records.append((row_i + 1, dose_v, turb_v,
                                        num(row.iloc[tph_i]) if tph_i is not None and tph_i < len(row) else None))
                    raw = num(frame.iloc[header_row + 1, raw_i]) if header_row + 1 < len(frame.index) else None
                    raw_ph = num(frame.iloc[header_row + 1, ph_i]) if ph_i is not None and header_row + 1 < len(frame.index) else None
                    if date_i is not None and header_row + 1 < len(frame.index):
                        date_raw = frame.iloc[header_row + 1, date_i]
                    break
            else:
                skipped_sheets.append(sheet)
                continue
        else:
            records = []
            for row_i in range(10, len(frame.index)):
                row = frame.iloc[row_i]
                dose_v = num(row.iloc[0]) if len(row) > 0 else None
                if dose_v is None:
                    continue
                turb_v = num(row.iloc[1]) if len(row) > 1 else None
                tph_v = num(row.iloc[2]) if len(row) > 2 else None
                records.append((row_i + 1, dose_v, turb_v, tph_v))

        if raw is None or raw <= 0 or not records:
            skipped_sheets.append(sheet)
            continue

        if isinstance(date_raw, (datetime, )):
            date_value = date_raw.date().isoformat()
        elif date_raw is not None and str(date_raw).strip():
            parsed_date = pd.to_datetime(date_raw, dayfirst=True, errors="coerce")
            date_value = parsed_date.date().isoformat() if not pd.isna(parsed_date) else datetime.now(timezone.utc).date().isoformat()
        else:
            date_value = datetime.now(timezone.utc).date().isoformat()

        code = str(sheet).strip()
        existing = db.table("experiments").select("id").eq("experiment_code", code).limit(1).execute().data or []
        if existing:
            exp_id = existing[0]["id"]
        else:
            notes = (f"Imported from uploaded original jar-test workbook. Worksheet name: {sheet}. "
                     f"Date cell as recorded: {date_raw!s}. Date was not inferred from worksheet title.")
            rec = db.table("experiments").insert({
                "experiment_code": code,
                "experiment_date": date_value,
                "source_location": "Historical NWSC jar-test workbook",
                "water_source": "Raw water",
                "raw_turbidity_ntu": raw,
                "raw_ph": raw_ph,
                "coagulant_name": "Aluminium sulphate",
                "notes": notes
            }).execute().data
            exp_id = rec[0]["id"]
            created_experiments += 1

        old_rows = db.table("jar_test_results").select("dose_mg_l,observation_notes").eq("experiment_id", exp_id).execute().data or []
        old_markers = {x.get("observation_notes") for x in old_rows}
        old_doses = {round(float(x["dose_mg_l"]), 6) for x in old_rows}
        # Keep one result per dose (database uniqueness rule); prefer a numeric
        # turbidity reading when the workbook repeats a dose on a summary row.
        preferred = {}
        for excel_row, dose_v, turb_v, tph_v in records:
            key = round(float(dose_v), 6)
            if key not in preferred or (preferred[key][2] is None and turb_v is not None):
                preferred[key] = (excel_row, dose_v, turb_v, tph_v)
        pending = []
        for excel_row, dose_v, turb_v, tph_v in preferred.values():
            marker = f"source_sheet={sheet}; source_excel_row={excel_row}; source_file=JAR TEST EXPERIMENT"
            if marker in old_markers or round(float(dose_v), 6) in old_doses:
                continue
            pending.append({
                "experiment_id": exp_id,
                "dose_mg_l": dose_v,
                "treated_turbidity_ntu": turb_v,
                "treated_ph": tph_v,
                "result_origin": "imported",
                "observation_notes": marker
            })
            old_doses.add(round(float(dose_v), 6))
        if pending:
            db.table("jar_test_results").insert(pending).execute()
            inserted += len(pending)

    if skipped_sheets:
        st.warning("Skipped sheets without a recognizable experiment layout: " + ", ".join(skipped_sheets))
    return inserted, created_experiments



def import_bundled_historical_dataset():
    """Idempotently load the committed, source-faithful NWSC dataset into Supabase."""
    path = "data/historical_jar_tests.csv"
    frame = pd.read_csv(path, dtype={"experiment_code": str, "date_cell_original": str})
    required = {"experiment_code","date_iso_parsed","raw_turbidity_ntu","dose_mg_l","source_excel_row"}
    if not required.issubset(frame.columns):
        raise ValueError("Bundled historical dataset is missing required columns.")
    db = get_db()
    inserted = 0
    created = 0
    for code, group in frame.groupby("experiment_code", sort=False):
        first = group.iloc[0]
        exists = db.table("experiments").select("id").eq("experiment_code", str(code)).limit(1).execute().data or []
        if exists:
            exp_id = exists[0]["id"]
        else:
            date_value = str(first["date_iso_parsed"]) if pd.notna(first["date_iso_parsed"]) and str(first["date_iso_parsed"]).strip() else datetime.now(timezone.utc).date().isoformat()
            raw_ph = first.get("raw_ph")
            raw_ph = float(raw_ph) if pd.notna(raw_ph) else None
            date_original = str(first.get("date_cell_original", ""))
            notes = (f"Imported from committed historical dataset derived from JAR TEST EXPERIMENT (1).xlsx. "
                     f"Original worksheet: {code}. Original date cell: {date_original}. "
                     "Worksheet title/date discrepancies are intentionally preserved; '*' values are missing, not zero.")
            rec = db.table("experiments").insert({
                "experiment_code": str(code), "experiment_date": date_value,
                "source_location": "Historical NWSC jar-test workbook",
                "water_source": "Raw water",
                "raw_turbidity_ntu": float(first["raw_turbidity_ntu"]),
                "raw_ph": raw_ph, "coagulant_name": "Aluminium sulphate",
                "notes": notes
            }).execute().data
            exp_id = rec[0]["id"]
            created += 1

        existing_rows = db.table("jar_test_results").select("dose_mg_l,observation_notes").eq("experiment_id", exp_id).execute().data or []
        existing_markers = {row.get("observation_notes") for row in existing_rows}
        existing_doses = {round(float(row["dose_mg_l"]), 6) for row in existing_rows}
        # The database intentionally enforces one row per experiment/dose. The
        # source workbook sometimes repeats the selected dose on a summary row.
        # Prefer a row with a numeric treated-turbidity result over a blank repeat.
        group = group.assign(_has_result=group["treated_turbidity_ntu"].notna()).sort_values(
            "_has_result", ascending=False
        ).drop_duplicates(subset=["dose_mg_l"], keep="first")
        pending = []
        for _, row in group.iterrows():
            excel_row = int(row["source_excel_row"])
            marker = f"source_sheet={code}; source_excel_row={excel_row}; source_file=JAR TEST EXPERIMENT"
            dose_value = float(row["dose_mg_l"])
            if marker in existing_markers or round(dose_value, 6) in existing_doses:
                continue
            def optional_float(value):
                return float(value) if pd.notna(value) and str(value).strip() != "" else None
            pending.append({
                "experiment_id": exp_id,
                "dose_mg_l": dose_value,
                "treated_turbidity_ntu": optional_float(row["treated_turbidity_ntu"]),
                "treated_ph": optional_float(row["treated_ph"]),
                "result_origin": "imported",
                "observation_notes": marker
            })
            existing_markers.add(marker)
            existing_doses.add(round(dose_value, 6))
        if pending:
            db.table("jar_test_results").insert(pending).execute()
            inserted += len(pending)
    return inserted, created


def make_synthetic(curves, count, seed=17):
    rng=np.random.default_rng(seed)
    out=[]
    for n in range(count):
        c=curves[int(rng.integers(0,len(curves)))]
        raw=float(c["raw"]*np.exp(rng.normal(0,0.08)))
        for mult in [0.6,0.8,1.0,1.2,1.5,2.0]:
            dose=float(c["d50"]*mult*np.exp(rng.normal(0,0.05)))
            turb=max(0.05,hill_curve(raw,dose,c["floor"],c["d50"],c["hill"])*np.exp(rng.normal(0,c["rmse"])))
            out.append({"synthetic_test":f"SYN-{n+1:02d}","raw_turbidity_ntu":round(raw,2),
                        "dose_mg_l":round(dose,2),"settled_turbidity_ntu":round(turb,2),
                        "record_origin":"synthetic — not measured"})
    return pd.DataFrame(out)

ready=db_status()
st.sidebar.markdown("**Storage:** Supabase PostgreSQL")
if ready: st.sidebar.success("Database connection configured")
st.sidebar.caption("Only real/imported measurements train the dose model. Synthetic records are never stored as measured results.")

with st.sidebar:
    st.divider()
    st.subheader("Historical NWSC dataset")
    st.caption("The original uploaded workbook has been transcribed into a version-controlled dataset in this repository.")
    if st.button("Load committed historical data into Supabase",disabled=not ready,type="primary"):
        try:
            n, created = import_bundled_historical_dataset()
            st.success(f"Loaded {n} jar rows across {created} new experiments. Re-running is safe; source rows are de-duplicated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Historical data load failed: {exc}")
    st.divider()
    st.subheader("Import another historical workbook")
    workbook=st.file_uploader("Upload additional NWSC workbook (.xlsx)",type=["xlsx"],key="history_import")
    if workbook and st.button("Import to permanent database",disabled=not ready,type="primary"):
        try:
            n, created = import_historical_workbook(workbook)
            st.success(f"Imported {n} jar rows across {created} new experiments. Existing source-row records were skipped on repeat imports.")
            st.rerun()
        except Exception as exc:
            st.error(f"Import failed: {exc}")

# Historical bootstrap/reconciliation: run on every app start, not only when the
# experiments table is completely empty. This repairs partial first imports safely.
# Each jar row is de-duplicated by its original worksheet + Excel row marker.
if ready:
    try:
        seeded_rows, seeded_experiments = import_bundled_historical_dataset()
        if seeded_rows or seeded_experiments:
            st.sidebar.success(
                f"Historical data synchronised: {seeded_experiments} new experiments, "
                f"{seeded_rows} new jar rows."
            )
        else:
            st.sidebar.caption("Committed historical baseline is already synchronised.")
    except Exception as seed_exc:
        st.sidebar.error(f"Historical data synchronisation failed: {seed_exc}")

try:
    data=load_data() if ready else pd.DataFrame()
    load_error=None
except Exception as exc:
    data=pd.DataFrame(); load_error=str(exc)
if load_error: st.error(f"Database read failed: {load_error}")

if not data.empty:
    real_data=data[data.origin.isin(["measured","imported"])]
else: real_data=data

m1,m2,m3,m4=st.columns(4)
# Defensive dashboard counts: tolerate nullable identifiers/origin values from Postgres.
if data.empty:
    experiment_count = 0
    synthetic_count = 0
else:
    experiment_count = int(data["experiment_id"].dropna().astype(str).nunique())
    synthetic_count = int(data["origin"].fillna("").astype(str).eq("synthetic").sum())
m1.metric("Dose observations", f"{len(data):,}")
m2.metric("Experiments", f"{experiment_count:,}")
m3.metric("Measured/imported results", f"{len(real_data):,}")
m4.metric("Synthetic records in database", f"{synthetic_count:,}")

tabs=st.tabs(["Dose optimiser","Predict treated turbidity","Record jar test","History & validation","Data quality & exports"])

with tabs[0]:
    st.subheader("Aluminium sulphate dose optimiser")
    st.caption("Learns from complete experiment-level outcomes: the lowest dose that reached your selected settled-water turbidity target.")
    a,b,c=st.columns(3)
    raw=a.number_input("Current raw turbidity (NTU)",min_value=0.01,value=50.0,step=1.0,key="opt_raw")
    target=b.number_input("Target settled turbidity (NTU)",min_value=0.1,value=6.0,step=0.5,key="opt_target")
    year=c.selectbox("Training period",["All tests","2021 onward","2023 onward"],index=0,key="opt_year")
    year_from={"All tests":2000,"2021 onward":2021,"2023 onward":2023}[year]
    train=dose_training(data,target,year_from) if not data.empty else pd.DataFrame()
    model=fit_powerlaw(train) if not train.empty else None
    if model is None:
        st.warning(f"Not enough experiments reached ≤ {target:g} NTU. Need at least five usable experiments. Import historical jar tests or record more tests.")
    elif raw<=target:
        st.info("Raw turbidity is already at or below the selected target. This model cannot justify an alum dose.")
    elif not (model["xmin"]<=raw<=model["xmax"]):
        st.error(f"Raw turbidity {raw:g} NTU is outside the training range ({model['xmin']:.1f}–{model['xmax']:.1f} NTU). No dose recommendation is issued.")
        st.caption("Do not extrapolate this model for operational dosing. Run a jar test.")
    else:
        central,low,high=predict_dose(model,raw)
        bootstrap=bootstrap_required_doses(train,raw,1000,42)
        if len(bootstrap)>=100:
            central=float(np.median(bootstrap))
            q80=float(np.quantile(bootstrap,0.80))
            q90=float(np.quantile(bootstrap,0.90))
        else:
            q80,q90=high,high
        loo=loo_dose(train)
        confidence="Low"
        if model["n"]>=15 and loo is not None and loo<=0.20: confidence="Higher (historical validation only)"
        elif model["n"]>=8 and loo is not None and loo<=0.35: confidence="Moderate"
        st.markdown(f'<div style="font-size:38px;font-weight:700">{central:.1f} mg/L</div>',unsafe_allow_html=True)
        st.caption(f"Estimated dose to reach ≤ {target:g} NTU. Approximate regression 95% prediction interval: {low:.1f}–{high:.1f} mg/L. Confidence: {confidence}.")
        if len(bootstrap)>=100:
            st.markdown(f"**Bootstrap dose scenarios:** median {central:.1f} mg/L · 80th percentile {q80:.1f} mg/L · 90th percentile {q90:.1f} mg/L")
            candidate=st.number_input("Test a candidate dose (mg/L) — estimated chance of reaching target",min_value=0.0,value=float(round(central,1)),step=0.5,key="candidate_dose")
            chance=float(np.mean(bootstrap<=candidate)*100)
            st.metric("Historical-model chance at candidate dose",f"{chance:.0f}%")
            st.caption("Bootstrap resamples whole experiments, not individual jars. This is an empirical model probability, not a guarantee for today's water.")
        st.warning("This is a starting-dose estimate only. Confirm with a fresh jar test before changing plant dosing.")
        f1,f2,f3=st.columns(3)
        f1.metric("Experiments used",model["n"])
        f2.metric("Model R²",f'{model["r2"]:.2f}')
        f3.metric("Leave-one-experiment-out error",f'{loo*100:.0f}%' if loo is not None else "Not available")
        flow=st.number_input("Plant flow (m³/day)",min_value=0.0,value=62940.0,step=100.0)
        op_hours=st.number_input("Operating hours per day",min_value=0.1,max_value=24.0,value=24.0,step=0.5)
        shift=st.number_input("Shift length (hours)",min_value=0.1,max_value=24.0,value=8.0,step=0.5)
        bag=st.number_input("Alum bag mass (kg)",min_value=0.1,value=50.0,step=1.0)
        if flow>0:
            kg_day=central*flow/1000
            bags=kg_day/bag
            shift_bags=bags*shift/op_hours
            st.subheader("Theoretical alum consumption")
            x1,x2,x3=st.columns(3)
            x1.metric("Alum per day",f"{kg_day:,.1f} kg")
            x2.metric("Bags per day",f"{bags:,.2f}")
            x3.metric("Bags per shift",f"{shift_bags:,.2f}")
            st.caption(f"Whole bags for planning: {math.ceil(bags)} per day and {math.ceil(shift_bags)} per shift. Actual consumption depends on solution strength, feed calibration and plant operation.")
        if st.button("Save this dose estimate to prediction history",type="primary"):
            get_db().table("predictions").insert({
                "raw_turbidity_ntu":raw,"raw_ph":None,"dose_mg_l":central,
                "predicted_turbidity_ntu":None,"target_turbidity_ntu":target,
                "predicted_dose_mg_l":central,"dose_low_mg_l":low,"dose_high_mg_l":high,
                "plant_flow_m3_day":flow,"alum_bag_mass_kg":bag,
                "model_name":"Experiment-level log-log dose-to-target","model_version":"dose-optimiser-2",
                "input_within_training_range":True,"confidence_level":confidence,
                "prediction_notes":f"{model['n']} experiments; LOEO relative error={loo}"
            }).execute()
            st.success("Dose estimate saved in the prediction history.")
    if not train.empty:
        st.markdown("**Historical dose required versus raw turbidity**")
        fig=go.Figure()
        fig.add_scatter(x=train.raw,y=train.dose,mode="markers+text",text=train.experiment_id,
                        textposition="top center",name="Measured experiment-derived dose")
        if model:
            xx=np.geomspace(model["xmin"],model["xmax"],100)
            yy=[math.exp(model["a"]+model["b"]*math.log(x)) for x in xx]
            fig.add_scatter(x=xx,y=yy,mode="lines",name="Log-log trend")
        fig.update_layout(xaxis_title="Raw turbidity (NTU)",yaxis_title="Dose to target (mg/L)",height=360)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(train.sort_values("date",ascending=False),use_container_width=True,hide_index=True)

with tabs[1]:
    st.subheader("Treated turbidity prediction")
    st.caption("Predicts a treated turbidity at a specified alum dose. This is separate from the dose-to-target optimiser.")
    x1,x2,x3=st.columns(3)
    pr=x1.number_input("Raw turbidity (NTU)",min_value=0.0,value=50.0,key="pred_raw")
    pp=x2.number_input("Raw-water pH",min_value=0.0,max_value=14.0,value=7.0,key="pred_ph")
    pdose=x3.number_input("Alum dose (mg/L)",min_value=0.0,value=30.0,key="pred_dose")
    usable=data[data.origin.isin(["measured","imported"])].dropna(subset=["raw_turbidity","raw_ph","dose","treated_turbidity"]) if not data.empty else pd.DataFrame()
    if st.button("Estimate treated turbidity",type="primary"):
        if len(usable)<8:
            st.warning("At least eight complete real/imported observations with raw pH are required.")
        else:
            cols=["raw_turbidity","raw_ph","dose"]
            X=usable[cols]; y=usable.treated_turbidity
            model2=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,random_state=42)
            model2.fit(X,y)
            pred=float(model2.predict(pd.DataFrame([[pr,pp,pdose]],columns=cols))[0])
            outside=any(v<X[k].min() or v>X[k].max() for v,k in zip([pr,pp,pdose],cols))
            st.metric("Estimated treated turbidity",f"{max(0,pred):.2f} NTU")
            if outside: st.error("One or more inputs are outside the historical training range. Treat as unverified; run a jar test.")
            else: st.info("Inputs fall within the recorded feature ranges. This does not guarantee the target will be achieved.")
            st.caption(f"Model uses {len(usable)} measured/imported jar observations. Synthetic records are excluded.")

with tabs[2]:
    st.subheader("Record a real jar-test experiment")
    with st.form("new_experiment"):
        a,b,c=st.columns(3)
        code=a.text_input("Experiment ID (optional)")
        date=b.date_input("Experiment date",value=datetime.now().date())
        source=c.text_input("Source / sampling location",value="Raw water")
        a,b,c=st.columns(3)
        rt=a.number_input("Raw turbidity (NTU)",min_value=0.0,value=50.0)
        ph=b.number_input("Raw pH",min_value=0.0,max_value=14.0,value=7.0)
        flow=c.number_input("Plant flow (m³/day)",min_value=0.0,value=0.0)
        a,b,c=st.columns(3)
        temp=a.number_input("Raw temperature (°C; 0 = not recorded)",min_value=0.0,max_value=60.0,value=0.0)
        alk=b.number_input("Raw alkalinity (mg/L as CaCO₃; 0 = not recorded)",min_value=0.0,value=0.0)
        bag=b.number_input("Alum bag mass (kg)",min_value=0.0,value=50.0)
        op=a.number_input("Operating hours/day",min_value=0.0,max_value=24.0,value=24.0)
        sh=c.number_input("Shift length (hours)",min_value=0.0,max_value=24.0,value=8.0)
        st.markdown("**Jar results — one row per jar: dose, settled turbidity, treated pH (optional)**")
        raw_jars=st.text_area("Example: 20, 8.2, 7.1  (one jar per line)",height=150)
        notes=st.text_area("Observations: floc formation, mixing, settling time, unusual conditions")
        save=st.form_submit_button("Save experiment to permanent database",type="primary",disabled=not ready)
    if save:
        parsed=[]
        try:
            for line in raw_jars.splitlines():
                if not line.strip(): continue
                vals=[v.strip() for v in line.replace(";",",").split(",")]
                if len(vals)<2: vals=line.split()
                if len(vals)<2: raise ValueError("Each jar row needs at least dose and settled turbidity.")
                dose_v=float(vals[0]); turb_v=float(vals[1]); tph=float(vals[2]) if len(vals)>2 and vals[2] else None
                if dose_v<0 or turb_v<0: raise ValueError("Dose and turbidity cannot be negative.")
                parsed.append((dose_v,turb_v,tph))
            if rt<=0 or not parsed: raise ValueError("Enter raw turbidity above zero and at least one jar result.")
            exp_code=add_experiment(code,date,source,rt,ph,parsed,temp or None,alk or None,flow or None,op or None,sh or None,bag or None,notes)
            st.success(f"Experiment {exp_code} and {len(parsed)} jar results saved.")
            st.rerun()
        except Exception as exc: st.error(f"Could not save experiment: {exc}")

with tabs[3]:
    st.subheader("Experiment history")
    if data.empty:
        st.info("The database has no jar-test results yet. Record a test or import the historical workbook.")
    else:
        f1,f2=st.columns(2)
        origins=f1.multiselect("Record origin",sorted(data.origin.dropna().unique()),default=sorted(data.origin.dropna().unique()))
        search=f2.text_input("Search experiment ID or source")
        view=data[data.origin.isin(origins)].copy()
        if search: view=view[view.experiment_id.astype(str).str.contains(search,case=False,na=False)|view.source.astype(str).str.contains(search,case=False,na=False)]
        st.dataframe(view.sort_values("date",ascending=False),use_container_width=True,hide_index=True)
        st.download_button("Export filtered results CSV",view.to_csv(index=False).encode("utf-8"),"nkana_jar_test_history.csv","text/csv")
    st.subheader("Prediction history")
    try:
        preds=get_db().table("predictions").select("*").order("created_at",desc=True).limit(250).execute().data or []
        if preds:
            st.dataframe(pd.DataFrame(preds),use_container_width=True,hide_index=True)
            st.markdown("**Record the actual result for a saved prediction**")
            outstanding=[p for p in preds if p.get("actual_turbidity_ntu") is None]
            if outstanding:
                labels={f"{p['id']} | {p.get('created_at','')[:16]} | dose {p.get('dose_mg_l','n/a')} mg/L":p["id"] for p in outstanding}
                with st.form("prediction_outcome"):
                    selected=st.selectbox("Prediction record",list(labels.keys()))
                    actual=st.number_input("Measured settled turbidity (NTU)",min_value=0.0,value=5.0,step=0.1)
                    outcome=st.form_submit_button("Save measured outcome")
                if outcome:
                    get_db().table("predictions").update({"actual_turbidity_ntu":actual}).eq("id",labels[selected]).execute()
                    st.success("Actual result saved. It will be available for prediction-performance review.")
                    st.rerun()
            else:
                st.caption("All displayed predictions already have measured outcomes, or no predictions are awaiting validation.")
        else: st.caption("No saved predictions yet.")
    except Exception as exc: st.error(f"Could not load prediction history: {exc}")

with tabs[4]:
    st.subheader("Data quality checks")
    if data.empty:
        st.info("No records available to audit.")
    else:
        checks=[]
        for code,g in data.groupby("experiment_id"):
            if g.raw_turbidity.isna().any(): checks.append((code,"Missing raw turbidity"))
            if g.dose.isna().any(): checks.append((code,"Missing alum dose"))
            if g.treated_turbidity.isna().any(): checks.append((code,"Missing treated turbidity"))
            if g.dose.duplicated().any(): checks.append((code,"Duplicate dose within experiment"))
            if (g.treated_turbidity<0).any(): checks.append((code,"Negative treated turbidity"))
            if (g.raw_turbidity<=0).any(): checks.append((code,"Raw turbidity zero or negative"))
            if g.origin.nunique()>1: checks.append((code,"Mixed data origins in one experiment"))
        if checks:
            st.warning(f"{len(checks)} data-quality flag(s) found.")
            st.dataframe(pd.DataFrame(checks,columns=["Experiment","Issue"]),use_container_width=True,hide_index=True)
        else: st.success("No basic data-quality flags found.")
        st.markdown("**Distribution of real measured/imported jar results**")
        real=data[data.origin.isin(["measured","imported"])]
        if not real.empty:
            fig=go.Figure()
            for code,g in real.groupby("experiment_id"):
                fig.add_scatter(x=g.dose,y=g.treated_turbidity,mode="lines+markers",name=str(code))
            fig.update_layout(xaxis_title="Alum dose (mg/L)",yaxis_title="Settled turbidity (NTU)",height=420)
            st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Trend simulator / synthetic data**")
    st.caption("Synthetic results are resampled from historical experiment curves. They are useful for visualising plausible trends, not for increasing the number of real observations or validating a model.")
    if not data.empty:
        curves=[]
        for _,g in data[data.origin.isin(["measured","imported"])].groupby("experiment_id"):
            c=fit_hill(g)
            if c: curves.append(c)
        if len(curves)>=4:
            n=st.slider("Number of synthetic experiment curves",1,40,10)
            if st.button("Generate synthetic jar-test scenarios"):
                syn=make_synthetic(curves,n)
                st.dataframe(syn,use_container_width=True,hide_index=True)
                st.download_button("Download synthetic scenarios CSV",syn.to_csv(index=False).encode("utf-8"),"synthetic_jar_test_scenarios_NOT_MEASURED.csv","text/csv")
        else:
            st.info("At least four historical experiments with four or more numeric jars are needed for the trend simulator.")

st.divider()
st.caption("Decision-support software only. Never use a predicted dose as an automatic dosing instruction. Confirm with jar testing, pH/alkalinity checks, settled and filtered turbidity, residual aluminium where required, and approved NWSC operating procedures.")
