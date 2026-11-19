# Historical NWSC jar-test dataset

Source: user-uploaded `JAR TEST EXPERIMENT (1).xlsx`.

The normalized dataset is `historical_jar_tests.csv`. It is included in the repository so that a fresh deployment can seed the Supabase database without the operator uploading the workbook again.

## Extracted inventory

- 31 non-empty experiment worksheets
- 193 dose rows, including rows with missing treated-turbidity readings
- 157 numeric treated-turbidity readings
- Missing values represented as empty CSV fields; the workbook's `*` marker is not converted to zero
- Original worksheet title, date-cell text, parsed date, and source Excel row are retained

## Important source-data cautions

- Worksheet titles and date cells do not always agree. The app retains the recorded date cell as the experiment date and stores the original text in the experiment notes; it does not silently infer a replacement date from the worksheet title.
- Examples requiring source-owner review include `16 DEC 2019` with date cell `16/12/2020`, `24 Dec 2020` with `24/01/2026`, `9 NOV 2026` with `2022-09-11`, and `5 OCT 2026` with `2026-05-10`.
- The `16 DEC 2019` worksheet records 0.1 NTU at 32 mg/L. This value is retained as written in the uploaded workbook; verify against the original laboratory notebook before relying on it.
- Several worksheets contain repeated dose entries and/or rows without a numeric treated-turbidity result. These are preserved with source row identifiers rather than deleted or treated as zero.

## Import behaviour

On first launch, if the Supabase `experiments` table is empty, the app automatically imports this committed dataset. The sidebar also provides a manual import action. Imported results are marked `imported`, not `measured`; both can be used as historical training data. Synthetic data is kept separate and is not used to train the model.

Imports are idempotent by source worksheet and source Excel row. Re-running the bundled import should not duplicate rows that were imported by the matching workbook importer.

## Governance

This is a transcription of the uploaded workbook, not an independent verification or correction of the original laboratory records. Confirm date anomalies, unusual values, and missing entries with the controlled laboratory source before using them in formal reporting or operational decisions.
