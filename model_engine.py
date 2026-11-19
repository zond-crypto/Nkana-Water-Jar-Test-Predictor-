"""Experiment-level modelling for Nkana Water alum jar tests."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REAL_ORIGINS={"measured","imported"}

def clean_real(data):
    if data.empty:return data.copy()
    d=data[data.origin.fillna("").isin(REAL_ORIGINS)].copy()
    for c in ["raw_turbidity","dose","treated_turbidity","raw_ph","treated_ph","temperature","alkalinity"]:
        if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    return d[(d.raw_turbidity>0)&(d.dose>=0)]

def target_dose(group,target):
    j=group.dropna(subset=["dose","treated_turbidity"]).sort_values("dose")
    j=j[(j.dose>0)&(j.treated_turbidity>=0)]
    if len(j)<2:return None
    for i,row in enumerate(j.itertuples()):
        if row.treated_turbidity<=target:
            if i==0:return float(row.dose)
            p=j.iloc[i-1]; y0=float(p.treated_turbidity); y1=float(row.treated_turbidity)
            if y0<=target or y0==y1:return float(row.dose)
            return float(p.dose+(y0-target)/(y0-y1)*(row.dose-p.dose))
    return None

def experiment_dose_table(data,target,year_from=2000):
    d=clean_real(data)
    if d.empty:return pd.DataFrame()
    d["date_parsed"]=pd.to_datetime(d.date,errors="coerce")
    d=d[d.date_parsed.dt.year.fillna(0)>=year_from]
    out=[]
    for code,g in d.groupby("experiment_id",dropna=True):
        raw=g.raw_turbidity.dropna()
        if raw.empty:continue
        dose=target_dose(g,target)
        if dose is not None and np.isfinite(dose) and dose>0:
            out.append({"experiment_id":str(code),"date":str(g.date.iloc[0]),"raw":float(raw.iloc[0]),"dose":float(dose),
              "raw_ph":float(g.raw_ph.dropna().iloc[0]) if g.raw_ph.notna().any() else np.nan})
    return pd.DataFrame(out)

def fit_log_model(train):
    if len(train)<5 or train.raw.nunique()<3:return None
    x=np.log(train.raw.to_numpy(float));y=np.log(train.dose.to_numpy(float));xb=float(x.mean());yb=float(y.mean())
    sxx=float(np.sum((x-xb)**2))
    if sxx<=1e-10:return None
    b=float(np.sum((x-xb)*(y-yb))/sxx);a=yb-b*xb;res=y-(a+b*x);n=len(x)
    sigma=float(np.sqrt(np.sum(res**2)/max(n-2,1)));sst=float(np.sum((y-yb)**2))
    return {"a":a,"b":b,"sigma":sigma,"xb":xb,"sxx":sxx,"n":n,"r2":float(1-np.sum(res**2)/sst) if sst else 0.0,
      "xmin":float(train.raw.min()),"xmax":float(train.raw.max())}

def predict_log_model(m,raw):
    x=math.log(float(raw));mu=m["a"]+m["b"]*x
    se=m["sigma"]*math.sqrt(1+1/m["n"]+(x-m["xb"])**2/m["sxx"])
    return math.exp(mu),math.exp(mu-1.96*se),math.exp(mu+1.96*se)

def leave_one_experiment_out(train):
    if len(train)<6:return None,pd.DataFrame()
    rows=[]
    for idx in train.index:
        m=fit_log_model(train.drop(index=idx))
        if m is None:continue
        r=train.loc[idx];pred=predict_log_model(m,float(r.raw))[0];actual=float(r.dose)
        rows.append({"experiment_id":r.experiment_id,"actual":actual,"predicted":pred,
          "absolute_error":abs(pred-actual),"absolute_pct_error":abs(pred-actual)/actual*100})
    v=pd.DataFrame(rows)
    if v.empty:return None,v
    return {"n":len(v),"mae_mg_l":float(v.absolute_error.mean()),
      "mape_pct":float(v.absolute_pct_error.mean()),"within_20_pct":float((v.absolute_pct_error<=20).mean()*100)},v

def fit_hill_curve(group):
    j=group.dropna(subset=["dose","treated_turbidity"]).copy()
    j=j[(j.dose>0)&(j.treated_turbidity>=0)].sort_values("dose")
    if len(j)<4 or j.dose.nunique()<4:return None
    raw=float(j.raw_turbidity.iloc[0]);x=j.dose.to_numpy(float);y=j.treated_turbidity.to_numpy(float)
    f0=max(0.,min(float(y.min()),raw*.2));d0=float(np.median(x))
    def fn(p,d):return p[0]+(raw-p[0])/(1+(d/p[1])**p[2])
    def residual(p):return np.log1p(fn(p,x))-np.log1p(y)
    try:
        fit=least_squares(residual,[f0,d0,2.],bounds=([0.,max(.01,x.min()*.1),.4],[max(.1,min(raw*.8,float(y.max())+.1)),x.max()*10,10]),loss="soft_l1",max_nfev=1200)
        floor,d50,h=map(float,fit.x);pred=fn(fit.x,x)
        rmse=float(np.sqrt(np.mean((np.log1p(pred)-np.log1p(y))**2)))
        return {"experiment_id":str(group.experiment_id.iloc[0]),"raw":raw,
          "raw_ph":float(group.raw_ph.dropna().iloc[0]) if group.raw_ph.notna().any() else None,
          "floor":floor,"d50":d50,"hill":h,"rmse":rmse,"dose_min":float(x.min()),"dose_max":float(x.max())}
    except Exception:return None

def curve_value(c,dose):
    return c["floor"]+(c["raw"]-c["floor"])/(1+(float(dose)/c["d50"])**c["hill"])

def fit_curves(data):
    d=clean_real(data)
    return [c for _,g in d.groupby("experiment_id") if (c:=fit_hill_curve(g)) is not None]

def predict_treated(curves,raw,dose,raw_ph=None):
    if len(curves)<4:return None
    raws=np.array([c["raw"] for c in curves],float)
    if raw<raws.min() or raw>raws.max():
        return {"blocked":True,"reason":f"Raw turbidity outside curve support ({raws.min():.1f}–{raws.max():.1f} NTU)."}
    dist=[]
    for c in curves:
        v=abs(math.log(max(raw,.01))-math.log(max(c["raw"],.01)))
        if raw_ph is not None and c["raw_ph"] is not None:v+=.20*abs(float(raw_ph)-c["raw_ph"])
        dist.append(v)
    idx=np.argsort(dist)[:min(7,len(curves))];chosen=[curves[i] for i in idx]
    w=np.array([1/(dist[i]+.05)**2 for i in idx]);w/=w.sum()
    vals=np.array([curve_value(c,dose) for c in chosen])
    return {"blocked":False,"estimate":float(np.sum(vals*w)),"low":float(np.quantile(vals,.1)),
      "high":float(np.quantile(vals,.9)),"n_neighbours":len(chosen),"neighbour_ids":[c["experiment_id"] for c in chosen]}

def quality_flags(data):
    d=clean_real(data);flags=[]
    if d.empty:return pd.DataFrame(columns=["experiment_id","issue"])
    for code,g in d.groupby("experiment_id"):
        if g.raw_turbidity.nunique()>1:flags.append((code,"Raw turbidity differs within experiment"))
        if g.dose.duplicated().any():flags.append((code,"Duplicate dose row"))
        if g.treated_turbidity.isna().any():flags.append((code,"Missing treated turbidity"))
        if (g.treated_turbidity<0).any():flags.append((code,"Negative treated turbidity"))
        if g.raw_ph.notna().any() and ((g.raw_ph.dropna()<0)|(g.raw_ph.dropna()>14)).any():flags.append((code,"Raw pH outside 0–14"))
    return pd.DataFrame(flags,columns=["experiment_id","issue"])
