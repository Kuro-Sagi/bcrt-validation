
import yaml, pandas as pd
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))
reports_tables = Path(cfg["paths"]["reports"]) / "tables"
reports_figs = Path(cfg["paths"]["reports"]) / "figures"
reports_figs.mkdir(parents=True, exist_ok=True)

# Just collate key CSVs; plotting can be added later if desired.
print("Report files:")
for fname in ["per_item_stats.csv","reliability.csv","selection_scores.csv","selection_decision.csv","nfc_correlations.csv","liwc_regressions.csv"]:
    p = reports_tables / fname
    print("-", p, "exists" if p.exists() else "MISSING")
