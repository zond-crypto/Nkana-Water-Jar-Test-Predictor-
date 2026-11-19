from __future__ import annotations
from datetime import datetime, timezone
import hmac, math, uuid
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from supabase import create_client
from model_engine import clean_real, fit_curves, predict_treated, quality_flags
from alum_optimizer import optimise_alum_dose, chemical_plan

st.set_page_config(page_title="Nkana Water | Treatment Intelligence",page_icon="💧",layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{color-scheme:dark;--ink:#e7eef7;--muted:#a7b6c8;--teal:#38c6c0;--blue:#8bd5e1;--line:#34465b;--panel:#172638;--surface:#101b2a}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:#0b1420;color:var(--ink)}
[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#0b1420;color:var(--ink)}
.block-container{max-width:1450px;padding-top:1.1rem;padding-bottom:3rem}
h1,h2,h3,h4,p,li,label,.stMarkdown,.stCaption{font-family:'DM Sans',sans-serif;color:var(--ink)}
h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.025em}
small,[data-testid="stCaptionContainer"]{color:#b5c3d2!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101f31,#123c50)}
[data-testid="stSidebar"] *{color:#edf7fb!important}
[data-testid="stMetric"]{background:#172638;border:1px solid #34465b;padding:1rem;border-radius:14px;box-shadow:0 4px 18px #0005}
[data-testid="stMetricLabel"],[data-testid="stMetricDelta"]{color:#c2d0df!important}
[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;color:#8fe0e2!important;font-weight:800}
.stTabs [data-baseweb="tab"]{font-weight:700;padding:0 .9rem;color:#c4d1df}
.stTabs [aria-selected="true"]{color:#63d8d0!important;border-bottom:3px solid #38c6c0}
.stButton>button,.stDownloadButton>button{border-radius:9px;min-height:2.6rem;font-weight:700;background:#20364b;color:#f2f7fb;border:1px solid #496078}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#55d1c8;color:#fff}
.stButton>button[kind="primary"]{background:#087f83;border-color:#38c6c0;color:white}
input,textarea,[data-baseweb="input"],[data-baseweb="select"],[data-baseweb="textarea"]{background:#142337!important;color:#f2f7fb!important;border-color:#52677d!important}
input,textarea{color:#f2f7fb!important}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"]{background:#172638!important;color:#f2f7fb!important}
[data-baseweb="option"],[role="option"]{color:#edf4fb!important}
[data-testid="stDataFrame"],[data-testid="stTable"]{background:#142337;color:#edf4fb}
[data-testid="stAlert"]{background:#172638;color:#edf4fb;border:1px solid #40556b}
hr{border-color:#34465b!important}
.nw-hero{padding:1.55rem 1.7rem;border-radius:18px;color:white;background:linear-gradient(115deg,#101f31,#16445a 60%,#087f83);margin-bottom:1.25rem;border:1px solid #31556b}
.nw-kicker{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:#bde8eb;font-weight:700}
.nw-title{font:800 clamp(1.6rem,3vw,2.4rem)/1.15 Manrope,sans-serif;color:white;margin:.4rem 0}
.nw-sub{color:#d8ebee;max-width:850px;margin:0}.nw-chip{display:inline-block;margin:.8rem .35rem 0 0;padding:.28rem .65rem;border:1px solid #ffffff55;border-radius:99px;color:#f2fbfc;font-size:.75rem}
@media(max-width:700px){.block-container{padding-left:.8rem;padding-right:.8rem}.nw-hero{padding:1.15rem;border-radius:14px}}
</style>
<div class="nw-hero"><div class="nw-kicker">Nkana Water Supply and Sanitation Company · Central Laboratory</div>
<div class="nw-title">Water Treatment Intelligence</div>
<p class="nw-sub">Aluminium sulphate decision support · experiment-level learning · dose-response prediction · chemical planning · model monitoring</p>
<span class="nw-chip">REAL DATA FIRST</span><span class="nw-chip">EXPLAINABLE MODELS</span><span class="nw-chip">AUDITABLE HISTORY</span></div>
""",unsafe_allow_html=True)

# Keep credentials in Streamlit Cloud → App settings → Secrets. Never commit them.
try:
    password = str(st.secrets.get("APP_PASSWORD", "")).strip()
    url = str(st.secrets.get("SUPABASE_URL", "")).strip()
    key = str(st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
except Exception:
    password = url = key = ""

if not password:
    st.error("Application sign-in is not configured. The administrator must add APP_PASSWORD in Streamlit Cloud → App settings → Secrets.")
    st.stop()

if not st.session_state.get("authenticated", False):
    st.subheader("Laboratory access")
    with st.form("login", clear_on_submit=True):
        entered = st.text_input("Application password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        if hmac.compare_digest(entered.strip().encode("utf-8"), password.encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Password not accepted. Check the application password with the administrator; it is not your GitHub or Streamlit account password.")
    st.stop()

if not url or not key:
    st.error("Database connection is not configured. The administrator must add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit Cloud → App settings → Secrets.")
    st.stop()

@st.cache_resource
def db():
    return create_client(url, key)
with st.sidebar:
    st.markdown("### NKANA WATER");st.caption("CENTRAL LABORATORY · INTERNAL DECISION SUPPORT")
    st.success("Database connection configured")
    if st.button("Sign out"):st.session_state.authenticated=False;st.rerun()
    st.divider();st.caption("Only measured/imported observations train models. Synthetic scenarios are never stored as measured results.")

@st.cache_data(ttl=60, show_spinner=False)
def load():
    ex=db().table("experiments").select("id,experiment_code,experiment_date,source_location,water_source,raw_turbidity_ntu,raw_ph,raw_temperature_c,raw_alkalinity_mg_l_caco3,plant_flow_m3_day,operating_hours_per_day,shift_length_hours,alum_bag_mass_kg,notes").order("experiment_date",desc=True).execute().data or []
    jars=db().table("jar_test_results").select("id,experiment_id,dose_mg_l,treated_turbidity_ntu,treated_ph,residual_aluminium_mg_l,settling_time_min,observation_notes,result_origin,created_at").execute().data or []
    if not ex or not jars:return pd.DataFrame()
    a,b=pd.DataFrame(ex),pd.DataFrame(jars);d=b.merge(a,left_on="experiment_id",right_on="id",suffixes=("_jar","_exp"))
    # Explicit mapping prevents duplicate experiment_id columns and the previous dashboard TypeError.
    return pd.DataFrame({"result_id":d["id_jar"],"experiment_id":d["experiment_code"],"date":d["experiment_date"],
      "source":d["source_location"],"water_source":d["water_source"],"raw_turbidity":d["raw_turbidity_ntu"],
      "raw_ph":d["raw_ph"],"temperature":d["raw_temperature_c"],"alkalinity":d["raw_alkalinity_mg_l_caco3"],
      "flow":d["plant_flow_m3_day"],"operating_hours":d["operating_hours_per_day"],"shift_hours":d["shift_length_hours"],
      "bag_mass":d["alum_bag_mass_kg"],"dose":d["dose_mg_l"],"treated_turbidity":d["treated_turbidity_ntu"],
      "treated_ph":d["treated_ph"],"residual_aluminium_mg_l":d["residual_aluminium_mg_l"],
      "settling_time_min":d["settling_time_min"],"observation_notes":d["observation_notes"],"origin":d["result_origin"],"notes":d["notes"]})

@st.cache_data(ttl=86400, show_spinner=False)
def sync_committed_history():
    """Idempotently seed the committed source transcription into the durable database."""
    frame=pd.read_csv("data/historical_jar_tests.csv",dtype={"experiment_code":str,"date_cell_original":str})
    required={"experiment_code","date_iso_parsed","raw_turbidity_ntu","dose_mg_l","source_excel_row"}
    if not required.issubset(frame.columns):raise ValueError("Historical baseline CSV has unexpected columns.")
    for code,g in frame.groupby("experiment_code",sort=False):
        first=g.iloc[0]
        found=db().table("experiments").select("id").eq("experiment_code",str(code)).limit(1).execute().data or []
        if found:expid=found[0]["id"]
        else:
            datev=str(first["date_iso_parsed"]) if pd.notna(first["date_iso_parsed"]) else datetime.now(timezone.utc).date().isoformat()
            phv=float(first["raw_ph"]) if pd.notna(first.get("raw_ph")) else None
            note=("Imported from source transcription of JAR TEST EXPERIMENT workbook. "
                  f"Original sheet={code}; original date cell={first.get('date_cell_original','')}. "
                  "Source date discrepancies are preserved; blank/'*' results remain missing.")
            created=db().table("experiments").insert({"experiment_code":str(code),"experiment_date":datev,
              "source_location":"Historical NWSC jar-test workbook","water_source":"Raw water",
              "raw_turbidity_ntu":float(first["raw_turbidity_ntu"]),"raw_ph":phv,
              "coagulant_name":"Aluminium sulphate","notes":note}).execute().data
            expid=created[0]["id"]
        old=db().table("jar_test_results").select("dose_mg_l").eq("experiment_id",expid).execute().data or []
        doses={round(float(x["dose_mg_l"]),6) for x in old}
        g=g.assign(_has=g["treated_turbidity_ntu"].notna()).sort_values("_has",ascending=False).drop_duplicates("dose_mg_l")
        pending=[]
        for _,row in g.iterrows():
            dosev=float(row["dose_mg_l"])
            if round(dosev,6) in doses:continue
            def opt(v):return float(v) if pd.notna(v) else None
            pending.append({"experiment_id":expid,"dose_mg_l":dosev,
              "treated_turbidity_ntu":opt(row["treated_turbidity_ntu"]),"treated_ph":opt(row.get("treated_ph")),
              "result_origin":"imported","observation_notes":f"source_sheet={code}; source_excel_row={int(row['source_excel_row'])}; baseline=v3"})
            doses.add(round(dosev,6))
        if pending:db().table("jar_test_results").insert(pending).execute()

try:
    sync_committed_history()
except Exception as seed_error:
    st.sidebar.warning(f"Historical baseline sync needs review: {seed_error}")

try:data=load()
except Exception as exc:
    st.error("Could not load the laboratory database. Check Supabase connection and table columns.");st.exception(exc);st.stop()
real=clean_real(data) if not data.empty else data
nexp=int(data.experiment_id.dropna().astype(str).nunique()) if not data.empty else 0
m1,m2,m3,m4=st.columns(4)
m1.metric("Jar observations",f"{len(data):,}");m2.metric("Experiments",f"{nexp:,}")
m3.metric("Measured / imported",f"{len(real):,}")
@st.cache_data(ttl=30, show_spinner=False)
def load_predictions():
    return db().table("predictions").select("*").order("created_at",desc=True).limit(500).execute().data or []

preds=[]
try:
    preds=load_predictions()
except Exception as exc:
    st.warning("Saved recommendations could not be loaded. Other laboratory functions remain available.")
    st.caption(f"Database read: {type(exc).__name__}")
m4.metric("Predictions awaiting validation",f"{sum(p.get('actual_turbidity_ntu') is None for p in preds):,}")
tabs=st.tabs(["Dose optimiser","Treated-water predictor","Record experiment","History & learning","Quality & governance"])

with tabs[0]:
    st.subheader("Aluminium sulphate dose optimiser")
    st.caption("One target-dose estimate per independent experiment. No synthetic observations enter the model.")
    a,b=st.columns(2)
    raw=a.number_input("Current raw turbidity (NTU)",min_value=.01,value=50.,step=1.)
    target=b.number_input("Target settled turbidity (NTU)",min_value=.1,value=6.,step=.5)
    result,train=optimise_alum_dose(data,raw,target,2000) if not data.empty else ({"blocked":True,"reason":"No historical observations are available."},pd.DataFrame())
    if result.get("blocked"):
        st.warning(result["reason"])
        if not train.empty:
            st.caption(f"{len(train)} independent experiments reached this target. The system requires enough data to validate a prediction.")
    else:
        x,y,z=st.columns(3)
        x.metric("Recommended starting dose",f"{result['dose_mg_l']:.1f} mg/L")
        y.metric("Empirical 90% error band",f"{result['low_mg_l']:.1f}–{result['high_mg_l']:.1f} mg/L")
        z.metric("Confidence",result["confidence"])
        val=result["validation"]
        st.caption(f"Selected method: {result['method']} · {result['n_experiments']} independent experiments · {result['nearby_experiments']} similar raw-water tests.")
        a,b,c=st.columns(3)
        a.metric("LOEO median absolute % error",f"{val['median_ape_pct']:.1f}%")
        b.metric("LOEO mean absolute % error",f"{val['mape_pct']:.1f}%")
        c.metric("Predictions within ±20%",f"{val['within_20_pct']:.0f}%")
        st.dataframe(pd.DataFrame(val["methods"])[["method","n","mae_mg_l","median_ape_pct","mape_pct","within_20_pct"]],hide_index=True,use_container_width=True)
        st.caption("Validation holds out complete experiments. The error band is calibrated from those held-out errors; it is not a guarantee for a new water source.")
        st.divider()
        st.markdown("**Plant chemical planning**")
        a,b,c=st.columns(3)
        flow=a.number_input("Plant flow (m³/day)",min_value=.01,value=62940.,step=100.)
        hours=b.number_input("Operating hours/day",min_value=.1,max_value=24.,value=24.)
        bag=c.number_input("Alum bag mass (kg)",min_value=.1,value=50.,step=1.)
        a,b=st.columns(2)
        shift=a.number_input("Shift length (hours)",min_value=.1,max_value=24.,value=8.)
        stock=b.number_input("Prepared stock strength (g/L, optional)",min_value=0.,value=0.,step=1.,help="Leave at zero if stock concentration is not measured.")
        plan=chemical_plan(result["dose_mg_l"],flow,bag,hours,shift,stock if stock>0 else None)
        a,b,c=st.columns(3)
        a.metric("Theoretical alum",f"{plan['kg_day']:,.1f} kg/day")
        b.metric("Bag requirement",f"{plan['bags_day']:,.2f} bags/day")
        c.metric("Shift requirement",f"{plan['kg_shift']:,.1f} kg/shift")
        d,e=st.columns(2)
        d.metric("Alum bags per shift",f"{plan['bags_shift']:.2f} bags",
                 help="Theoretical bag-equivalent consumption for the selected shift length.")
        e.metric("Whole bags to stage per shift",f"{math.ceil(plan['bags_shift']):,} bags",
                 help="Rounded up to ensure sufficient stock is available for the shift.")
        st.caption(f"Daily planning: {plan['bags_day']:.2f} bag-equivalents/day; stage {plan['whole_bags_day']} whole bags/day. Shift calculation uses the selected operating hours/day and shift length.")
        if stock>0:
            st.metric("Calculated stock-solution feed",f"{plan['solution_feed_l_h']:,.1f} L/hour")
        st.warning("Dose is expressed as mg/L of the same aluminium-sulphate product basis used in the historical jar tests. Verify stock preparation, pump calibration and product specification before translating this to plant feed.")
        if st.button("Save dose recommendation",type="primary"):
            db().table("predictions").insert({"raw_turbidity_ntu":raw,"target_turbidity_ntu":target,"dose_mg_l":result["dose_mg_l"],
              "predicted_dose_mg_l":result["dose_mg_l"],"dose_low_mg_l":result["low_mg_l"],"dose_high_mg_l":result["high_mg_l"],
              "plant_flow_m3_day":flow,"alum_bag_mass_kg":bag,"model_name":result["method"],"model_version":"NWSC-ATI-4.0",
              "input_within_training_range":True,"confidence_level":result["confidence"],
              "prediction_notes":f"target={target}; experiments={result['n_experiments']}; validation={val}"}).execute()
            load_predictions.clear()
            st.success("Dose recommendation saved for outcome validation.")
    if not train.empty:
        fig=go.Figure()
        fig.add_scatter(x=train.raw,y=train.dose,mode="markers+text",text=train.experiment_id,textposition="top center",name="Experiment target dose")
        fig.update_layout(template="plotly_dark",paper_bgcolor="#0b1420",plot_bgcolor="#101b2a",font_color="#e7eef7",xaxis_title="Raw turbidity (NTU)",yaxis_title=f"Alum dose to reach ≤ {target:g} NTU (mg/L)",height=390)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Each dot represents one independent experiment after monotone dose-response smoothing. The historical source contains date-label inconsistencies, so v4 deliberately uses all eligible experiments rather than filtering by year. Inspect unusual experiments before relying on the model.")
    st.info("pH, alkalinity and temperature are deliberately not used as prediction features yet: their historical coverage is too incomplete for a fair validation. We will add them only when enough comparable, verified measurements have accumulated.")

with tabs[1]:
    st.subheader("Treated-water response predictor")
    st.caption("Fits the complete dose-response curve of each historical experiment, then compares current raw water with similar experiments.")
    a,b,c=st.columns(3);r=a.number_input("Raw turbidity (NTU)",min_value=.01,value=50.,key="curve_raw")
    p=b.number_input("Raw pH (optional)",min_value=0.,max_value=14.,value=7.,key="curve_ph")
    dose=c.number_input("Candidate alum dose (mg/L)",min_value=0.,value=30.,key="curve_dose")
    curves=fit_curves(data) if not data.empty else []
    if st.button("Predict treated turbidity",type="primary"):
        result=predict_treated(curves,r,dose,p)
        if not result:st.warning("At least four experiments with four or more numeric jar readings are required.")
        elif result.get("blocked"):st.error(result["reason"])
        else:
            a,b,c=st.columns(3);a.metric("Weighted estimate",f"{result['estimate']:.2f} NTU")
            b.metric("Historical 10th–90th range",f"{result['low']:.2f}–{result['high']:.2f} NTU");c.metric("Comparable curves",result["n_neighbours"])
            st.caption("Range is the spread across selected historical experiment curves, not a formal 95% interval.")
            st.write("Comparable experiments: "+", ".join(result["neighbour_ids"]))
            st.warning("Verify with a fresh jar test. The estimate is not a guarantee for today's water.")
    if curves:
        grid=np.geomspace(max(.1,min(x["dose_min"] for x in curves)),max(x["dose_max"] for x in curves),80)
        fig=go.Figure()
        for q in curves:fig.add_scatter(x=grid,y=[q["floor"]+(q["raw"]-q["floor"])/(1+(v/q["d50"])**q["hill"]) for v in grid],mode="lines",name=q["experiment_id"])
        fig.update_layout(template="plotly_dark",paper_bgcolor="#0b1420",plot_bgcolor="#101b2a",font_color="#e7eef7",xaxis_title="Alum dose (mg/L)",yaxis_title="Modelled settled turbidity (NTU)",height=420)
        st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    st.subheader("Record a jar-test experiment")
    with st.form("record_experiment"):
        a,b,c=st.columns(3);code=a.text_input("Experiment ID (optional)");date=b.date_input("Experiment date",value=datetime.now().date())
        source=c.text_input("Source / sampling location",value="Raw water")
        a,b,c=st.columns(3);rawv=a.number_input("Raw turbidity (NTU)",min_value=.01,value=50.)
        ph=b.number_input("Raw pH (0 if not recorded)",min_value=0.,max_value=14.,value=7.)
        flow=c.number_input("Plant flow (m³/day; 0 if not recorded)",min_value=0.,value=0.)
        a,b,c=st.columns(3);temp=a.number_input("Temperature °C (0 if not recorded)",min_value=0.,max_value=60.,value=0.)
        alk=b.number_input("Alkalinity mg/L as CaCO₃ (0 if not recorded)",min_value=0.,value=0.)
        bag=c.number_input("Alum bag mass kg",min_value=.1,value=50.)
        jars=st.text_area("One jar per line: dose, settled turbidity, treated pH (optional)",placeholder="15, 8.2, 7.1\n20, 5.4, 7.0\n25, 3.1, 6.9",height=140)
        notes=st.text_area("Observations: floc, mixing, settling, unusual conditions")
        save=st.form_submit_button("Save experiment to permanent database",type="primary")
    if save:
        try:
            parsed=[]
            for line in jars.splitlines():
                if not line.strip():continue
                vals=[x.strip() for x in line.replace(";",",").split(",")]
                if len(vals)<2:vals=line.split()
                if len(vals)<2:raise ValueError("Each jar row requires dose and settled turbidity.")
                dv=float(vals[0]);tv=float(vals[1]);tp=float(vals[2]) if len(vals)>2 and vals[2] else None
                if dv<0 or tv<0:raise ValueError("Dose and turbidity cannot be negative.")
                parsed.append((dv,tv,tp))
            if len(parsed)<3:raise ValueError("Enter at least three jar readings.")
            expcode=code.strip() or f"JT-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:5]}"
            if db().table("experiments").select("id").eq("experiment_code",expcode).limit(1).execute().data:raise ValueError("Experiment ID already exists.")
            created=db().table("experiments").insert({"experiment_code":expcode,"experiment_date":date.isoformat(),"source_location":source,
              "water_source":"Raw water","raw_turbidity_ntu":rawv,"raw_ph":ph if ph>0 else None,
              "raw_temperature_c":temp if temp>0 else None,"raw_alkalinity_mg_l_caco3":alk if alk>0 else None,
              "plant_flow_m3_day":flow if flow>0 else None,"alum_bag_mass_kg":bag,"coagulant_name":"Aluminium sulphate","notes":notes}).execute().data
            expid=created[0]["id"]
            db().table("jar_test_results").insert([{"experiment_id":expid,"dose_mg_l":dv,"treated_turbidity_ntu":tv,
              "treated_ph":tp,"result_origin":"measured"} for dv,tv,tp in parsed]).execute()
            load.clear()
            st.success(f"Saved {expcode} with {len(parsed)} jar readings.");st.rerun()
        except Exception as exc:st.error(f"Could not save experiment: {exc}")

with tabs[3]:
    st.subheader("Historical experiment register")
    if data.empty:st.info("No database observations found.")
    else:
        a,b=st.columns(2);origins=a.multiselect("Data origin",sorted(data.origin.dropna().unique()),default=sorted(data.origin.dropna().unique()))
        search=b.text_input("Find experiment")
        view=data[data.origin.isin(origins)].copy()
        if search:view=view[view.experiment_id.astype(str).str.contains(search,case=False,na=False)]
        st.dataframe(view.sort_values("date",ascending=False),use_container_width=True,hide_index=True)
        st.download_button("Export observations CSV",view.to_csv(index=False).encode(),"nkana_water_jar_history.csv","text/csv")
    st.subheader("Recommendation outcome register")
    if preds:
        st.dataframe(pd.DataFrame(preds),use_container_width=True,hide_index=True)
        waiting=[q for q in preds if q.get("actual_turbidity_ntu") is None]
        if waiting:
            opts={f"{q.get('id')} · {str(q.get('created_at',''))[:16]} · {q.get('dose_mg_l')} mg/L":q for q in waiting}
            with st.form("outcome"):
                label=st.selectbox("Select recommendation",list(opts))
                actual=st.number_input("Measured settled turbidity (NTU)",min_value=0.,value=5.,step=.1)
                record=st.form_submit_button("Record actual result")
            if record:
                db().table("predictions").update({"actual_turbidity_ntu":actual}).eq("id",opts[label]["id"]).execute()
                load_predictions.clear()
                st.success("Outcome saved.");st.rerun()
    else:st.caption("No saved recommendations yet.")
    verified=[q for q in preds if q.get("actual_turbidity_ntu") is not None and q.get("predicted_turbidity_ntu") is not None]
    if verified:
        err=[abs(float(q["actual_turbidity_ntu"])-float(q["predicted_turbidity_ntu"])) for q in verified]
        st.metric("Validated treated-turbidity MAE",f"{np.mean(err):.2f} NTU")
    dose_outcomes=[q for q in preds if q.get("actual_turbidity_ntu") is not None and q.get("target_turbidity_ntu") is not None and q.get("predicted_dose_mg_l") is not None]
    if dose_outcomes:
        achieved=sum(float(q["actual_turbidity_ntu"])<=float(q["target_turbidity_ntu"]) for q in dose_outcomes)
        st.metric("Saved dose recommendations that achieved target",f"{achieved/len(dose_outcomes)*100:.0f}%",help=f"{achieved} of {len(dose_outcomes)} recommendations with recorded actual outcomes met their saved target.")

with tabs[4]:
    st.subheader("Data quality and model governance")
    flags=quality_flags(data) if not data.empty else pd.DataFrame()
    if flags.empty:st.success("No rule-based data-quality flags found.")
    else:st.warning(f"{len(flags)} flags require review.");st.dataframe(flags,use_container_width=True,hide_index=True)
    if not data.empty:
        st.metric("Real observations eligible for model fitting",len(real))
        comp=[]
        for col,label in [("raw_ph","Raw pH"),("temperature","Temperature"),("alkalinity","Alkalinity"),("treated_ph","Treated pH"),("residual_aluminium_mg_l","Residual aluminium"),("settling_time_min","Settling time")]:
            n=int(real[col].notna().sum()) if col in real else 0
            comp.append({"Measurement":label,"Available results":n,"Completeness %":round(100*n/max(len(real),1),1)})
        st.dataframe(pd.DataFrame(comp),hide_index=True,use_container_width=True)
    st.markdown("""
**Governance rules**
- Each jar-test experiment is one independent unit for model validation.
- Measured/imported data only; synthetic scenarios are never training records.
- Predictions are blocked outside the historical raw-turbidity range.
- Missing pH, alkalinity and temperature remain missing; no invented measurements.
- New measurements are retained and can be used for later model evaluation.
- No automatic dosing control. Confirm each candidate with a fresh jar test and NWSC procedures.
""")
st.divider()
st.caption("NWSC Water Treatment Intelligence · Alum module · v4.0 · Laboratory decision support only, not an automatic dosing controller.")
