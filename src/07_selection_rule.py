
import yaml, pandas as pd, numpy as np
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))
reports_tables = Path(cfg["paths"]["reports"]) / "tables"
processed_dir = Path(cfg["paths"]["processed"])
interim_dir = Path(cfg["paths"]["interim"])
reports_tables.mkdir(parents=True, exist_ok=True)

per_item = pd.read_csv(reports_tables / "per_item_stats.csv")

# Utility z-sum with penalties
bounds = cfg["analysis"]["difficulty_bounds"]
min_r = cfg["analysis"]["min_itemrest_r"]
min_intuit = cfg["analysis"]["min_intuitive_pct"]
soft, hard = cfg["analysis"]["exposure_penalty_thresholds"]

df = per_item.copy()
for col in ["p_naive","r_itemrest_comb","r_itemrest_crt2","pct_intuitive_errors"]:
    mu = df[col].mean()
    sd = df[col].std(ddof=0) or 1.0
    df[f"z_{col}"] = (df[col]-mu)/sd

df["penalty"] = 0.0
df.loc[(df["p_naive"]<bounds[0]) | (df["p_naive"]>bounds[1]), "penalty"] -= 2
df.loc[df["r_itemrest_comb"]<min_r, "penalty"] -= 2
df.loc[df["pct_intuitive_errors"]<min_intuit, "penalty"] -= 1
df.loc[df["exposure_delta_p"]>=soft, "penalty"] -= 1
df.loc[df["exposure_delta_p"]>=hard, "penalty"] -= 2

df["utility"] = df[["z_p_naive","z_r_itemrest_comb","z_r_itemrest_crt2","z_pct_intuitive_errors"]].sum(axis=1) + df["penalty"]
df = df.sort_values("utility", ascending=False)
df.to_csv(reports_tables / "selection_scores.csv", index=False)

# Decide
all_clear = (
    df["p_naive"].between(bounds[0], bounds[1]).all() and
    (df["r_itemrest_comb"]>=min_r).all() and
    (df["pct_intuitive_errors"]>=min_intuit).all()
)
choice = "keep_all_6" if all_clear else "keep_best_4"
keep_items = df["item"].tolist() if all_clear else df["item"].head(4).tolist()

# If best-4, compute bcrt4_total from long-form
if choice == "keep_best_4":
    bcrt4_cols = [c for c in keep_items if c.startswith("bcrt")]
    item_long = pd.read_csv(interim_dir / "scored_long.csv")
    bcrt4 = (item_long[item_long["item_id"].isin(bcrt4_cols)]
                .pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first")
                .fillna(0).astype(int).sum(axis=1))
    person = pd.read_csv(processed_dir / "person_wide.csv").set_index("pid")
    person["bcrt4_total"] = bcrt4
    person["bcrt4_total"] = person["bcrt4_total"].fillna(0).astype(int)
    person.reset_index().to_csv(processed_dir / "person_wide.csv", index=False)

pd.Series({"decision": choice, "keep_items": ",".join(keep_items)}).to_csv(reports_tables / "selection_decision.csv")
print(f"Selection: {choice}; items: {keep_items}")
