import numpy as np
import pandas as pd
from alum_optimizer import isotonic_decreasing, experiment_target_dose, build_experiment_table, optimise_alum_dose, chemical_plan


def test_isotonic_decreasing_removes_local_rebound():
    fitted = isotonic_decreasing([12.0, 8.0, 9.0, 4.0])
    assert np.all(np.diff(fitted) <= 1e-12)
    assert np.allclose(fitted, [12.0, 8.5, 8.5, 4.0])


def test_target_dose_interpolates_monotone_smoothed_curve():
    group = pd.DataFrame({"dose": [10, 20, 30, 40], "treated_turbidity": [12, 7, 8, 3]})
    dose = experiment_target_dose(group, 6.0)
    assert 20 < dose < 40


def test_synthetic_rows_are_excluded():
    rows = []
    for i in range(8):
        for dose, turb in [(10, 12), (20, 7), (30, 4)]:
            rows.append({"experiment_id": f"E{i}", "date": "2025-01-01",
                         "raw_turbidity": 20 + i * 10, "dose": dose,
                         "treated_turbidity": turb, "raw_ph": 7.0,
                         "origin": "synthetic" if i == 0 else "measured"})
    table = build_experiment_table(pd.DataFrame(rows), 6)
    assert len(table) == 7
    assert "E0" not in set(table.experiment_id)


def test_chemical_plan_mass_balance():
    plan = chemical_plan(30, 62940, 50, 24, 8, 10)
    assert np.isclose(plan["kg_day"], 1888.2)
    assert np.isclose(plan["bags_day"], 37.764)
    assert np.isclose(plan["kg_shift"], 629.4)
    assert np.isclose(plan["solution_feed_l_h"], 7867.5)


def test_optimizer_blocks_extrapolation():
    rows = []
    for i in range(10):
        raw = 20 + i * 10
        for dose, turb in [(10, 15), (20, 8), (30, 4)]:
            rows.append({"experiment_id": f"E{i}", "date": "2025-01-01",
                         "raw_turbidity": raw, "dose": dose,
                         "treated_turbidity": turb, "raw_ph": 7.0,
                         "origin": "measured"})
    result, _ = optimise_alum_dose(pd.DataFrame(rows), 150, 6)
    assert result["blocked"]
