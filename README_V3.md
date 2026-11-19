# Nkana Water Treatment Intelligence — v3.0

## What is new
- Polished Streamlit interface for the Alum module.
- Fixed the joined-data mapping by constructing a unique DataFrame explicitly; avoids the duplicate-column / dashboard metric TypeError from the deployed build.
- Separate treated-turbidity predictor fits a monotonic Hill response curve for each complete historical experiment, then uses nearby whole-experiment curves.
- Dose optimiser uses one interpolated dose-to-target point per experiment.
- Leave-one-experiment-out validation holds out complete experiments, never individual jars.
- Strictly blocks estimates outside observed raw-turbidity support.
- Adds feature-completeness, outcome capture, quality flags and exportable history.
- Synthetic data remain illustrative and are never used to train or validate.

## Architecture
- app_v3.py: Streamlit UI, authentication, Supabase data access, records, reporting.
- model_engine.py: pure experiment-level model functions.
- tests/test_model_engine.py: regression tests.
- Supabase PostgreSQL remains the durable system of record. Keep APP_PASSWORD, SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit Cloud Secrets only.
- GitHub stores source; Streamlit Community Cloud runs the app.

## Deploy
The existing deployed entry point app.py now forwards to app_v3.py, so no main-file setting change should be necessary. Streamlit Community Cloud should redeploy from the new commit. Keep existing secrets, reboot if needed, then verify experiment and jar counts. Run tests with pytest -q.

## Data and limits
No database migration is required. Keep the repository private and never commit credentials. The historical data are sparse for pH, temperature and alkalinity; the first dose model therefore uses raw turbidity only and does not invent missing variables. Improve coverage through routine real measurements. Predictions are decision support only; confirm every candidate with a current jar test and approved NWSC operating procedures.
