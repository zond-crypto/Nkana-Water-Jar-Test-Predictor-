import pandas as pd
from model_engine import target_dose,fit_log_model,predict_log_model,fit_hill_curve,curve_value

def test_target_interpolates_between_jars():
 d=pd.DataFrame({"dose":[10,20,30],"treated_turbidity":[12,8,4]})
 assert abs(target_dose(d,6)-25)<1e-9

def test_target_none_if_not_reached():
 d=pd.DataFrame({"dose":[10,20,30],"treated_turbidity":[20,15,10]})
 assert target_dose(d,5) is None

def test_log_model_returns_positive_prediction_and_interval():
 d=pd.DataFrame({"raw":[10,20,30,40,50,60],"dose":[8,12,16,20,24,28]})
 m=fit_log_model(d);p,lo,hi=predict_log_model(m,35)
 assert 0<lo<p<hi

def test_hill_curve_declines_with_dose():
 d=pd.DataFrame({"experiment_id":["A"]*5,"raw_turbidity":[50]*5,"raw_ph":[7]*5,
 "dose":[5,10,20,30,40],"treated_turbidity":[30,20,10,6,4]})
 c=fit_hill_curve(d)
 assert c is not None and curve_value(c,40)<curve_value(c,5)
