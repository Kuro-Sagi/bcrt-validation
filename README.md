# bcrt-validation

Clean, script-first pipeline to validate 6 bespoke CRT items (bCRT) alongside CRT-2,
integrate LIWC from two writing tasks (Q1, Q2), and select either all 6 or the best 4 items.

## Quick start

1) Put your Qualtrics CSV in `data/raw/`.
2) Update `config/config.yaml` if paths differ.
3) Run the pipeline step-by-step:
   ```bash
   python src/00_load_and_qc.py
   python src/01_score_items.py
   python src/02_make_splits.py
   python src/03_item_stats.py
   python src/04_reliability.py
   python src/05_liwc_io.py   # export texts, then run LIWC externally, then re-run to merge
   python src/06_external_validity.py
   python src/07_selection_rule.py
   python src/99_report.py
   ```

Artifacts: tables/figures saved under `reports/`.
