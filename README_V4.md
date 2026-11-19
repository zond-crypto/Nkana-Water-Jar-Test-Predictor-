# Nkana Water Treatment Intelligence — Alum Optimiser v4.0

## Scope
This release focuses exclusively on aluminium sulphate dose optimisation and dosing prediction.

## What changed
- Added `alum_optimizer.py`, a separate, testable optimisation engine.
- Reconstructs one target-reaching dose per independent jar-test experiment.
- Averages replicate dose readings and applies a monotone non-increasing isotonic fit to reduce local turbidity rebounds before interpolating the target dose.
- Compares a robust log-log power-law model with a similar-experiment weighted-neighbour baseline.
- Selects the model using leave-one-experiment-out validation; each entire experiment is held out.
- Reports median absolute percentage error, mean absolute percentage error, ±20% hit rate and an empirical multiplicative error band derived from held-out errors.
- Blocks predictions when raw turbidity is outside the historical support, when raw turbidity is already at/below target, or when there are too few independent experiments.
- Adds product mass, bag, shift and optional stock-solution feed calculations.
- Keeps synthetic data out of model training and validation.

## App architecture
- `app.py`: stable Streamlit entry point.
- `app_v4.py`: v4 user interface and Supabase integration.
- `alum_optimizer.py`: dose-response smoothing, model comparison, validation and chemical calculations.
- `model_engine.py`: existing treated-turbidity curve predictor, retained for comparison.
- `tests/test_alum_optimizer.py`: v4 regression tests.
- `tests/test_model_engine.py`: existing model tests.
- Supabase PostgreSQL remains the durable data store; this release requires no schema migration.

## Model governance and limitations
- Prediction feature: raw turbidity only. Historical pH, alkalinity and temperature coverage is incomplete, so these are not included as model features yet.
- Target dose is derived from measured jar results. Isotonic smoothing reduces non-monotone noise but cannot correct transcription errors or poor jar-test practice.
- Model comparison uses leave-one-experiment-out validation. A small historical dataset can still produce unstable estimates.
- The empirical error band describes historical held-out performance, not a guaranteed confidence interval for a new source or unusual water chemistry.
- Dose units must use the same alum-product basis as the historical jar-test stock preparation. Confirm stock strength, product specification and dosing-pump calibration before converting to plant feed.
- Synthetic observations are not stored as measured results and are never used to fit or validate the optimiser.
- The application is decision support only. Confirm candidate doses with a fresh jar test and approved NWSC procedures. It does not automatically control dosing equipment.

## Deploy
The Streamlit Community Cloud main file remains `app.py`. The entry point now loads v4. Keep `APP_PASSWORD`, `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in Streamlit Secrets only. Never commit credentials or operational secrets.

## Test
```bash
pip install -r requirements.txt
pytest -q
```
