"""Validated alum dose optimisation from independent jar-test experiments.

Only measured/imported observations are eligible. Synthetic records are excluded.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REAL_ORIGINS = {"measured", "imported"}


def _real(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    d = data[data.get("origin", pd.Series(index=data.index, dtype=object)).fillna("").isin(REAL_ORIGINS)].copy()
    for col in ("raw_turbidity", "dose", "treated_turbidity", "raw_ph"):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def isotonic_decreasing(values):
    """Unweighted pool-adjacent-violators fit constrained to be non-increasing."""
    y = np.asarray(values, dtype=float)
    if len(y) == 0:
        return y
    means, weights, starts, ends = [], [], [], []
    for i, value in enumerate(-y):
        means.append(float(value)); weights.append(1.0); starts.append(i); ends.append(i + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            w = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            means[-2:] = [m]; weights[-2:] = [w]
            ends[-2:] = [ends[-1]]
            starts.pop()
    fitted = np.empty(len(y), dtype=float)
    for mean, start, end in zip(means, starts, ends):
        fitted[start:end] = -mean
    return fitted


def experiment_target_dose(group: pd.DataFrame, target: float):
    """Estimate first dose reaching target after monotone smoothing of replicate means."""
    j = group.dropna(subset=["dose", "treated_turbidity"]).copy()
    j = j[(j.dose > 0) & (j.treated_turbidity >= 0)]
    if j.empty:
        return None
    j = j.groupby("dose", as_index=False).treated_turbidity.mean().sort_values("dose")
    if len(j) < 2:
        return None
    doses = j.dose.to_numpy(float)
    observed = j.treated_turbidity.to_numpy(float)
    fitted = isotonic_decreasing(observed)
    hit = np.flatnonzero(fitted <= float(target))
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(doses[0])
    y0, y1 = fitted[i - 1], fitted[i]
    if y0 <= target or abs(y0 - y1) < 1e-12:
        return float(doses[i])
    fraction = (y0 - target) / (y0 - y1)
    return float(doses[i - 1] + fraction * (doses[i] - doses[i - 1]))


def build_experiment_table(data: pd.DataFrame, target: float, year_from: int = 2000):
    d = _real(data)
    if d.empty:
        return pd.DataFrame(columns=["experiment_id", "date", "raw", "dose", "raw_ph"])
    d["date_parsed"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d.date_parsed.dt.year.fillna(0) >= int(year_from)]
    rows = []
    for code, group in d.groupby("experiment_id", dropna=True):
        raw_values = group.raw_turbidity.dropna()
        if raw_values.empty:
            continue
        raw = float(raw_values.median())
        if raw <= float(target):
            continue
        dose = experiment_target_dose(group, target)
        if dose is None or not np.isfinite(dose) or dose <= 0:
            continue
        ph = group.raw_ph.dropna()
        rows.append({"experiment_id": str(code), "date": str(group.date.iloc[0]),
                     "raw": raw, "dose": dose, "raw_ph": float(ph.iloc[0]) if not ph.empty else np.nan})
    return pd.DataFrame(rows)


def _fit_loglinear(train: pd.DataFrame):
    if len(train) < 5 or train.raw.nunique() < 3:
        return None
    x = np.log(train.raw.to_numpy(float))
    y = np.log(train.dose.to_numpy(float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    x0 = float(np.mean(x))
    def residual(p):
        return p[0] + p[1] * (x - x0) - y
    try:
        fit = least_squares(residual, [float(np.mean(y)), 0.5], loss="soft_l1", f_scale=0.15)
        return {"intercept": float(fit.x[0]), "slope": float(fit.x[1]), "x0": x0,
                "xmin": float(train.raw.min()), "xmax": float(train.raw.max())}
    except Exception:
        return None


def _predict_loglinear(model, raw: float):
    return float(math.exp(model["intercept"] + model["slope"] * (math.log(float(raw)) - model["x0"])))


def _predict_neighbours(train: pd.DataFrame, raw: float, k: int = 5):
    if train.empty:
        return None
    dist = np.abs(np.log(train.raw.to_numpy(float)) - math.log(float(raw)))
    ix = np.argsort(dist)[:min(k, len(train))]
    # Inverse-distance weighting; log-dose averaging limits domination by large doses.
    weights = 1.0 / np.maximum(dist[ix], 0.04) ** 2
    vals = np.log(train.dose.to_numpy(float)[ix])
    return float(math.exp(np.average(vals, weights=weights)))


def _score_predictions(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ape = np.abs(p - a) / np.maximum(a, 1e-9) * 100
    return {"n": int(len(a)), "mae_mg_l": float(np.mean(np.abs(p-a))),
            "median_ape_pct": float(np.median(ape)), "mape_pct": float(np.mean(ape)),
            "within_20_pct": float(np.mean(ape <= 20) * 100)}


def leave_one_experiment_out(train: pd.DataFrame):
    """Compare robust power-law and local-neighbour predictions, holding out whole tests."""
    if len(train) < 7:
        return None, pd.DataFrame()
    rows = []
    for idx in train.index:
        held = train.loc[idx]
        remaining = train.drop(index=idx)
        model = _fit_loglinear(remaining)
        candidates = {
            "Robust power law": _predict_loglinear(model, float(held.raw)) if model else None,
            "Similar experiments": _predict_neighbours(remaining, float(held.raw))
        }
        for name, pred in candidates.items():
            if pred is not None and np.isfinite(pred):
                rows.append({"experiment_id": held.experiment_id, "method": name,
                             "actual": float(held.dose), "predicted": float(pred)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return None, detail
    scores = []
    for name, group in detail.groupby("method"):
        score = _score_predictions(group.actual, group.predicted)
        score["method"] = name
        scores.append(score)
    summary = pd.DataFrame(scores).sort_values(["median_ape_pct", "mae_mg_l"])
    best = summary.iloc[0].to_dict()
    best["methods"] = summary.to_dict("records")
    return best, detail


def optimise_alum_dose(data: pd.DataFrame, raw: float, target: float, year_from: int = 2000):
    """Return a model-selected starting dose and empirical LOEO error band."""
    table = build_experiment_table(data, target, year_from)
    if len(table) < 7:
        return {"blocked": True, "reason": f"Only {len(table)} independent experiments reached the target; at least 7 are required for validation."}, table
    if raw <= target:
        return {"blocked": True, "reason": "Raw turbidity is already at or below the selected settled-water target."}, table
    if raw < table.raw.min() or raw > table.raw.max():
        return {"blocked": True, "reason": f"Raw turbidity is outside the validated training range ({table.raw.min():.1f}–{table.raw.max():.1f} NTU)."}, table
    validation, details = leave_one_experiment_out(table)
    if validation is None:
        return {"blocked": True, "reason": "Whole-experiment validation could not be completed."}, table
    model = _fit_loglinear(table)
    linear = _predict_loglinear(model, raw) if model else None
    local = _predict_neighbours(table, raw)
    selected = validation["method"]
    central = linear if selected == "Robust power law" else local
    if central is None or not np.isfinite(central):
        return {"blocked": True, "reason": "No stable dose estimate could be fitted."}, table
    selected_errors = details[details.method == selected].copy()
    log_errors = np.abs(np.log(selected_errors.predicted.to_numpy(float) / selected_errors.actual.to_numpy(float)))
    # Empirical 90% multiplicative error band from held-out complete experiments.
    q = float(np.quantile(log_errors, 0.90, method="higher")) if len(log_errors) >= 7 else float(np.max(log_errors))
    low, high = central / math.exp(q), central * math.exp(q)
    near = int((np.abs(np.log(table.raw / float(raw))) <= math.log(1.4)).sum())
    confidence = "Low"
    if len(table) >= 15 and validation["median_ape_pct"] <= 20 and near >= 3:
        confidence = "Moderate"
    if len(table) >= 20 and validation["median_ape_pct"] <= 12 and near >= 5:
        confidence = "Higher (still requires jar-test confirmation)"
    return {"blocked": False, "dose_mg_l": float(central), "low_mg_l": float(low),
            "high_mg_l": float(high), "method": selected, "validation": validation,
            "n_experiments": int(len(table)), "nearby_experiments": near,
            "confidence": confidence, "training_min": float(table.raw.min()),
            "training_max": float(table.raw.max()), "table": table}, table


def chemical_plan(dose_mg_l: float, flow_m3_day: float, bag_mass_kg: float = 50.0,
                  operating_hours: float = 24.0, shift_hours: float = 8.0,
                  stock_strength_g_l: float | None = None):
    """Translate product dose to theoretical product mass and optional stock-solution feed."""
    if min(dose_mg_l, flow_m3_day, bag_mass_kg, operating_hours, shift_hours) <= 0:
        raise ValueError("Dose, flow, bag mass and operating hours must be positive.")
    kg_day = float(dose_mg_l) * float(flow_m3_day) / 1000.0
    result = {"kg_day": kg_day, "bags_day": kg_day / bag_mass_kg,
              "whole_bags_day": int(math.ceil(kg_day / bag_mass_kg)),
              "kg_shift": kg_day * shift_hours / operating_hours,
              "bags_shift": kg_day * shift_hours / operating_hours / bag_mass_kg}
    if stock_strength_g_l is not None:
        if stock_strength_g_l <= 0:
            raise ValueError("Stock strength must be greater than zero.")
        # 1 mg/L × 1 m³/h = 1 g/h.
        result["solution_feed_l_h"] = float(dose_mg_l) * (flow_m3_day / operating_hours) / float(stock_strength_g_l)
    return result
