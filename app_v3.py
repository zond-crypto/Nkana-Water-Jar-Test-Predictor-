from __future__ import annotations
from datetime import datetime, timezone
import hmac, math, uuid
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from supabase import create_client
from model_engine import clean_real, experiment_dose_table, fit_log_model, predict_log_model, leave_one_experiment_out, fit_curves, predict_treated, quality_flags

st.set_page_config(page_title="Nkana Water | Treatment Intelligence",page_icon="💧",layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#142d40;--muted:#627789;--teal:#087f8c;--blue:#16425f;--line:#dce6ec}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:#f4f7f9;color:var(--ink)}
.block-container{max-width:1450px;padding-top:1.1rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.025em;color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#102f43,#14536a)}[data-testid="stSidebar"] *{color:#eff9fb!important}
[data-testid="stMetric"]{background:white;border:1px solid var(--line);padding:1rem;border-radius:14px;box-shadow:0 4px 16px #142d4009}
[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;color:var(--blue);font-weight:800}
.stTabs [data-baseweb="tab"]{font-weight:700;padding:0 .9rem}.stTabs [aria-selected="true"]{color:var(--teal)!important;border-bottom:3px solid var(--teal)}
.stButton>button,.stDownloadButton>button{border-radius:9px;min-height:2.6rem;font-weight:700}
.stButton>button[kind="primary"]{background:var(--teal);border-color:var(--teal)}
.nw-hero{padding:1.55rem 1.7rem;border-radius:18px;color:white;background:linear-gradient(115deg,#102f43,#14536a 60%,#087f8c);margin-bottom:1.25rem}
.nw-kicker{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:#bde8eb;font-weight:700}
.nw-title{font:800 clamp(1.6rem,3vw,2.4rem)/1.15 Manrope,sans-serif;color:white;margin:.4rem 0}
.nw-sub{color:#d8ebee;max-width:850px;margin:0}.nw-chip{display:inline-block;margin:.8rem .35rem 0 0;padding:.28rem .65rem;border:1px solid #ffffff40;border-radius:99px;color:#f2fbfc;font-size:.75rem}
@media(max-width:700px){.block-container{padding-left:.8rem;padding-right:.8rem}.nw-hero{padding:1.15rem;border-radius:14px}}
</style>
<div class="nw-hero"><div class="nw-kicker">Nkana Water Supply and Sanitation Company · Central Laboratory</div>
<div class="nw-title">Water Treatment Intelligence</div>
<p class="nw-sub">Aluminium sulphate decision support · experiment-level learning · dose-response prediction · chemical planning · model monitoring</p>
<span class="nw-chip">REAL DATA FIRST</span><span class="nw-chip">EXPLAINABLE MODELS</span><span class="nw-chip">AUDITABLE HISTORY</span></div>
""",unsafe_allow_html=True)

# Credentials must be stored in Streamlit Cloud Secrets, never in GitHub.
password=st.secrets.get("APP_PASSWORD","");url=st.secrets.get("SUPABASE_URL","");key=st.secrets.get("SUPABASE_SERVICE_ROLE_KEY","")
if not password or not url or not key:
    st.error("Configuration incomplete. Add APP_PASSWORD, SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to Streamlit Cloud → App settings → Secrets.");st.stop()
if not st.session_state.get("authenticated"):
    with st.form("login"):
        st.subheader("Laboratory access");entered=st.text_input("Application password",type="password")
        if st.form_submit_button("Sign in",type="primary"):
            if hmac.compare_digest(entered,password):st.session_state.authenticated=True;st.rerun()
            st.error("Incorrect password.")
    st.stop()
@st.cache_resource
def db():return create_client(url,key)
with st.sidebar:
    st.markdown("### NKANA WATER");st.caption("CENTRAL LABORATORY · INTERNAL DECISION SUPPORT")
    st.success("Database connection configured")
    if st.button("Sign out"):st.session_state.authenticated=False;st.rerun()
    st.divider();st.caption("Only measured/imported observations train models. Synthetic scenarios are never stored as measured results.")

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

try:data=load()
except Exception as exc:
    st.error("Could not load the laboratory database. Check Supabase connection and table columns.");st.exception(exc);st.stop()
real=clean_real(data) if not data.empty else data
nexp=int(data.experiment_id.dropna().astype(str).nunique()) if not data.empty else 0
m1,m2,m3,m4=st.columns(4)
m1.metric("Jar observations",f"{len(data):,}");m2.metric("Experiments",f"{nexp:,}")
m3.metric("Measured / imported",f"{len(real):,}")
preds=[]
try:preds=db().table("predictions").select("*").order("created_at",desc=True).limit(500).execute().data or []
except Exception:pass
m4.metric("Predictions awaiting validation",f"{sum(p.get('actual_turbidity_ntu') is None for p in preds):,}")
tabs=st.tabs(["Dose optimiser","Treated-water predictor","Record experiment","History & learning","Quality & governance"])

with tabs[0]:
    st.subheader("Aluminium sulphate dose optimiser")
    st.caption("One interpolated dose-to-target value per historical experiment. Jars from one experiment are never counted as independent experiments.")
    a,b,c=st.columns(3)
    raw=a.number_input("Current raw turbidity (NTU)",min_value=.01,value=50.,step=1.)
    target=b.number_input("Target settled turbidity (NTU)",min_value=.1,value=6.,step=.5)
    years=c.selectbox("Training period",["All tests","2021 onward","2023 onward"])
    yr={"All tests":2000,"2021 onward":2021,"2023 onward":2023}[years]
    train=experiment_dose_table(data,target,yr) if not data.empty else pd.DataFrame()
    model=fit_log_model(train) if not train.empty else None
    if model is None:st.warning("Insufficient independent experiments reached this target. Need at least five usable experiments.")
    elif raw<=target:st.info("Raw turbidity is already at or below target. The historical model cannot justify an alum dose.")
    elif not model["xmin"]<=raw<=model["xmax"]:st.error(f"No recommendation: {raw:g} NTU is outside historical training range {model['xmin']:.1f}–{model['xmax']:.1f} NTU.")
    else:
        central,lo,hi=predict_log_model(model,raw);cv,cv_rows=leave_one_experiment_out(train)
        conf="Low"
        if cv and model["n"]>=15 and cv["mape_pct"]<=20:conf="Higher (historical validation only)"
        elif cv and model["n"]>=8 and cv["mape_pct"]<=35:conf="Moderate"
        st.metric("Modelled starting dose",f"{central:.1f} mg/L")
        st.caption(f"Approximate 95% prediction interval: {lo:.1f}–{hi:.1f} mg/L · {model['n']} experiments · confidence {conf}.")
        if cv:
            x,y,z=st.columns(3);x.metric("LOEO dose MAPE",f"{cv['mape_pct']:.1f}%");y.metric("Within ±20%",f"{cv['within_20_pct']:.0f}%");z.metric("Validation experiments",cv["n"])
        else:st.info("Not enough independent experiments for leave-one-experiment-out validation.")
        if len(train)>=8:
            st.markdown("**Whole-experiment bootstrap scenarios**")
            rng=np.random.default_rng(42);vals=[]
            for _ in range(700):
                sample=train.iloc[rng.integers(0,len(train),len(train))];bm=fit_log_model(sample)
                if bm and bm["xmin"]<=raw<=bm["xmax"]:vals.append(predict_log_model(bm,raw)[0])
            if len(vals)>=100:
                q50,q80,q90=np.quantile(vals,[.5,.8,.9]);x,y,z=st.columns(3)
                x.metric("Median",f"{q50:.1f} mg/L");y.metric("80th percentile",f"{q80:.1f} mg/L");z.metric("90th percentile",f"{q90:.1f} mg/L")
        st.divider();st.markdown("**Chemical planning**")
        a,b,c=st.columns(3);flow=a.number_input("Plant flow (m³/day)",min_value=0.,value=62940.,step=100.)
        hours=b.number_input("Operating hours/day",min_value=.1,max_value=24.,value=24.)
        bag=c.number_input("Alum bag mass (kg)",min_value=.1,value=50.,step=1.)
        shift=st.number_input("Shift length (hours)",min_value=.1,max_value=24.,value=8.)
        if flow>0:
            kg=central*flow/1000;bags=kg/bag;sb=bags*shift/hours
            x,y,z=st.columns(3);x.metric("Theoretical alum kg/day",f"{kg:,.1f}");y.metric("Bags/day",f"{bags:,.2f}");z.metric("Bags/shift",f"{sb:,.2f}")
            st.caption(f"Whole-bag planning: {math.ceil(bags)} bags/day and {math.ceil(sb)} bags/shift. Actual use depends on solution strength, feeder calibration and process losses.")
        st.warning("Decision support only. Confirm every candidate with a fresh jar test.")
        if st.button("Save dose recommendation",type="primary"):
            db().table("predictions").insert({"raw_turbidity_ntu":raw,"target_turbidity_ntu":target,"dose_mg_l":central,
              "predicted_dose_mg_l":central,"dose_low_mg_l":lo,"dose_high_mg_l":hi,"plant_flow_m3_day":flow,
              "alum_bag_mass_kg":bag,"model_name":"Experiment-level log-log dose model","model_version":"NWSC-ATI-3.0",
              "input_within_training_range":True,"confidence_level":conf,"prediction_notes":f"n={model['n']}; LOEO={cv}"}).execute()
            st.success("Recommendation saved for later outcome validation.")
    if not train.empty:
        fig=go.Figure();fig.add_scatter(x=train.raw,y=train.dose,mode="markers+text",text=train.experiment_id,textposition="top center",name="Historical dose")
        if model:
            xx=np.geomspace(model["xmin"],model["xmax"],100);yy=[math.exp(model["a"]+model["b"]*math.log(v)) for v in xx]
            fig.add_scatter(x=xx,y=yy,mode="lines",name="Log–log trend")
        fig.update_layout(xaxis_title="Raw turbidity (NTU)",yaxis_title=f"Dose to reach ≤ {target:g} NTU (mg/L)",height=360)
        st.plotly_chart(fig,use_container_width=True)

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
        fig.update_layout(xaxis_title="Alum dose (mg/L)",yaxis_title="Modelled settled turbidity (NTU)",height=420)
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
                st.success("Outcome saved.");st.rerun()
    else:st.caption("No saved recommendations yet.")
    verified=[q for q in preds if q.get("actual_turbidity_ntu") is not None and q.get("predicted_turbidity_ntu") is not None]
    if verified:
        err=[abs(float(q["actual_turbidity_ntu"])-float(q["predicted_turbidity_ntu"])) for q in verified]
        st.metric("Validated treated-turbidity MAE",f"{np.mean(err):.2f} NTU")

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
st.caption("NWSC Water Treatment Intelligence · Alum module · v3.0 · Laboratory decision support only, not an automatic dosing controller.")
