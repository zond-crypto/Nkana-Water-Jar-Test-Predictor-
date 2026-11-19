# Nkana Water Jar Test Predictor

Streamlit decision-support application for historical jar-test dose-response analysis and treated-turbidity estimation.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Use Streamlit Community Cloud, select this private repository, branch `main`, and main file `app.py`.

## Import historical data
In the app sidebar, upload `Nkana_Water_Intelligent_Jar_Test_Workbook.xlsx`. The app reads the `Jar Test Data` sheet and appends numeric dose-response observations. Avoid importing the same workbook repeatedly.

## Important limitations
- The model is a preliminary Extra Trees regression model using raw turbidity, raw pH and alum dose. It is not yet a validated operational dosing model.
- Cross-validation metrics are indicative only; experiments from the same source workbook may be correlated.
- Predictions outside the historical input ranges are flagged and should not be relied upon without jar testing.
- SQLite on Streamlit hosting is not durable storage. Configure a managed persistent database before continuous operational use or multi-user deployment.
- Do not upload confidential or identifiable operational records to a public repository. Keep this repository private.
- Confirm every candidate dose with a fresh jar test and approved operational procedures.
