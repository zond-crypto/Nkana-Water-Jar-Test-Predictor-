# Isolated Streamlit frontend diagnostic

This test is intentionally separate from the Nkana app. It imports no project
modules, reads no Streamlit secrets, and performs no Supabase/database calls.

## Create the test app

1. Open https://share.streamlit.io/ in Chrome and sign in with the same account
   that owns the existing Nkana app.
2. Choose **Create app** (or **New app**) and select **Deploy a public app from
   GitHub**.
3. Repository: `zond-crypto/Nkana-Water-Jar-Test-Predictor-`
4. Branch: `main`
5. Main file path: `diagnostic_app.py`
6. Give it a distinct name such as `nkana-streamlit-diagnostic`, then deploy.
7. Do not edit, delete, or change the existing Nkana app or its secrets.

## Interpret the result

- If this diagnostic page renders, its text and buttons work, but the original
  app remains black: the Streamlit frontend can render in this account and the
  problem is specific to the original app/configuration/runtime.
- If this diagnostic app is also black or blank: the issue is likely at the
  Streamlit Cloud/browser/frontend delivery layer, not the Nkana application
  code or Supabase.
- If deployment fails, capture the deployment logs. Do not share secret values.

The page should display **SUCCESS: Streamlit rendered this page**, the installed
Streamlit version, a text input, and a Test rerun button.
