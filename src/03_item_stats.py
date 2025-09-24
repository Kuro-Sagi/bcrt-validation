# src/03_item_stats.py
"""
Item-level diagnostics and figures.

Outputs (tables):
- reports/tables/per_item_stats_naive.csv    (naive participants only)
- reports/tables/per_item_stats_full.csv     (all participants)

Outputs (figures):
- reports/figures/item_rest_r_combined.png
- reports/figures/response_types_stacked_naive.png
- reports/figures/response_types_stacked_full.png
- reports/figures/scatter_r_vs_intuitive.png
- reports/figures/scatter_r_vs_p.png
- reports/figures/scatter_exposure_vs_p.png
- reports/figures/compare_r_naive_vs_full.png
- reports/figures/compare_p_naive_vs_full.png
- reports/figures/icc/icc_<item>.png               (empirical ICCs; naïve)
"""

import yaml, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- tiny helpers ----------
def point_biserial(y_bin, x_cont):
    y = pd.Series(y_bin).astype(float)
    x = pd.Series(x_cont).astype(float)
    if y.nunique(dropna=True) < 2: return np.nan
    if x.std(ddof=1) == 0 or np.isnan(x.std(ddof=1)): return np.nan
    return float(np.corrcoef(y, x)[0, 1])

def _label_points(ax, x, y, labels):
    for xi, yi, lab in zip(np.asarray(x), np.asarray(y), labels):
        if np.isnan(xi) or np.isnan(yi): continue
        ax.annotate(str(lab), (xi, yi), textcoords="offset points", xytext=(3, 3), ha="left", fontsize=8)

# ---------- load ----------
cfg = yaml.safe_load(open("config/config.yaml"))
interim = Path(cfg["paths"]["interim"])
processed = Path(cfg["paths"]["processed"])
reports = Path(cfg["paths"]["reports"])
reports_tables = reports / "tables"
reports_figs = reports / "figures"
reports_tables.mkdir(parents=True, exist_ok=True)
reports_figs.mkdir(parents=True, exist_ok=True)
(fig_icc_dir := reports_figs / "icc").mkdir(parents=True, exist_ok=True)

crt2 = cfg["columns"]["crt2_items"]
bcrt = cfg["columns"]["bcrt_items"]
all_items = crt2 + bcrt

item_long = pd.read_csv(interim / "scored_long.csv")
# wide 0/1 matrix
pivot = (item_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first")
         .reindex(columns=all_items).fillna(0).astype(int))

# totals
tot_crt2 = pivot[crt2].sum(axis=1) if crt2 else pd.Series(0, index=pivot.index)
tot_bcrt = pivot[bcrt].sum(axis=1) if bcrt else pd.Series(0, index=pivot.index)
tot_comb = tot_crt2 + tot_bcrt

# splits
def load_pids_naive():
    pw = processed / "person_naive.csv"
    if pw.exists():
        return pd.read_csv(pw)["pid"].tolist()
    # fallback: derive naïve from item_long (everyone with seen==0 for all items)
    seen_by_pid = item_long.groupby("pid")["seen"].max()
    return seen_by_pid[seen_by_pid == 0].index.tolist()

naive_pids = load_pids_naive()
naive_pids = [p for p in naive_pids if p in pivot.index]
full_pids = pivot.index.tolist()

# ---------- per-item stats: NAIVE ----------
rows = []
for item in all_items:
    fam = "CRT2" if item in crt2 else "BCRT"

    # naïve difficulty
    p_naive = pivot.loc[naive_pids, item].mean() if naive_pids else np.nan

    # rested totals (naïve)
    r_comb = point_biserial(pivot.loc[naive_pids, item],
                            (tot_comb - pivot[item]).loc[naive_pids]) if naive_pids else np.nan

    # CRT2-anchored rest-total (minus self for CRT2 items; unchanged for BCRT)
    anchor_crt2 = (tot_crt2 - pivot[item]) if item in crt2 else tot_crt2
    r_crt2 = point_biserial(pivot.loc[naive_pids, item], anchor_crt2.loc[naive_pids]) if naive_pids else np.nan

    # BCRT-anchored rest-total (minus self for BCRT items; unchanged for CRT2)
    anchor_bcrt = (tot_bcrt - pivot[item]) if item in bcrt else tot_bcrt
    r_bcrt = point_biserial(pivot.loc[naive_pids, item], anchor_bcrt.loc[naive_pids]) if naive_pids else np.nan

    # naïve intuitive% (among errors)
    sub_n = item_long[(item_long["item_id"] == item) & (item_long["pid"].isin(naive_pids))]
    n_err = int((sub_n["correct"] == 0).sum())
    n_int = int((sub_n["resp_type"] == 1).sum())
    pct_intuit = (n_int / n_err) if n_err > 0 else np.nan

    # exposure inflation on full sample (use long form so seen is per response)
    sub_f = item_long[item_long["item_id"] == item]
    # Within the full sample (all respondents), compute difficulty split by exposure
    # p_full_seen  = mean P(correct) among exposed respondents
    # p_full_unseen= mean P(correct) among unexposed respondents
    p_full_seen = sub_f.loc[sub_f["seen"] == 1, "correct"].mean()
    p_full_unseen = sub_f.loc[sub_f["seen"] == 0, "correct"].mean()
    # Exposure inflation (full): how much easier the item is when seen vs unseen
    delta_p = (p_full_seen - p_full_unseen) if (not np.isnan(p_full_seen) and not np.isnan(p_full_unseen)) else np.nan

    rows.append({
        "item": item, "family": fam,
        "n_naive": len(naive_pids),
        "p_naive": p_naive,
        "r_itemrest_crt2": r_crt2,
        "r_itemrest_bcrt": r_bcrt,
        "r_itemrest_comb": r_comb,
        "pct_intuitive_errors": pct_intuit,
        "p_full_unseen": p_full_unseen,
        "p_full_seen": p_full_seen,
        "exposure_delta_p": delta_p
    })

stats_naive = pd.DataFrame(rows)
# Harmonise column names for naive table
if not stats_naive.empty:
    stats_naive = stats_naive.rename(columns={
        "n_naive": "n",
        "p_naive": "p",
    })
    # Order columns consistently
    col_order = [
        "item","family","n","p",
        "r_itemrest_crt2","r_itemrest_bcrt","r_itemrest_comb",
        "pct_intuitive_errors","p_full_unseen","p_full_seen","exposure_delta_p"
    ]
    stats_naive = stats_naive[[c for c in col_order if c in stats_naive.columns]]
stats_naive.to_csv(reports_tables / "per_item_stats_naive.csv", index=False)
print("Saved per_item_stats_naive.csv")

# ---------- per-item stats: FULL (all participants) ----------
rows_f = []
for item in all_items:
    fam = "CRT2" if item in crt2 else "BCRT"
    p_full = pivot.loc[full_pids, item].mean() if full_pids else np.nan
    r_comb_f = point_biserial(pivot.loc[full_pids, item], (tot_comb - pivot[item]).loc[full_pids]) if full_pids else np.nan
    anchor_crt2_f = (tot_crt2 - pivot[item]) if item in crt2 else tot_crt2
    r_crt2_f = point_biserial(pivot.loc[full_pids, item], anchor_crt2_f.loc[full_pids]) if full_pids else np.nan
    anchor_bcrt_f = (tot_bcrt - pivot[item]) if item in bcrt else tot_bcrt
    r_bcrt_f = point_biserial(pivot.loc[full_pids, item], anchor_bcrt_f.loc[full_pids]) if full_pids else np.nan
    rows_f.append({
        "item": item, "family": fam,
        "n_full": len(full_pids),
        "p_full": p_full,
        "r_itemrest_crt2_full": r_crt2_f,
        "r_itemrest_bcrt_full": r_bcrt_f,
        "r_itemrest_comb_full": r_comb_f,
    })
stats_full = pd.DataFrame(rows_f)
# Harmonise column names for full table
if not stats_full.empty:
    stats_full = stats_full.rename(columns={
        "n_full": "n",
        "p_full": "p",
        "r_itemrest_crt2_full": "r_itemrest_crt2",
        "r_itemrest_bcrt_full": "r_itemrest_bcrt",
        "r_itemrest_comb_full": "r_itemrest_comb",
    })
    col_order = [
        "item","family","n","p",
        "r_itemrest_crt2","r_itemrest_bcrt","r_itemrest_comb",
        "pct_intuitive_errors","p_unseen","p_seen","exposure_delta_p"
    ]
    # Some of these may be NaN if not applicable; keep uniform order when present
    stats_full = stats_full[[c for c in col_order if c in stats_full.columns]]
stats_full.to_csv(reports_tables / "per_item_stats_full.csv", index=False)
print("Saved per_item_stats_full.csv")

# ---------- Figures ----------
# 1) Discrimination bar (combined r; naïve)
sd = stats_naive.sort_values("r_itemrest_comb")
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(sd["item"], sd["r_itemrest_comb"])
ax.set_xlabel("Rested item–total r (Combined10, naïve)")
ax.set_ylabel("Item")
ax.set_title("Item discrimination")
plt.tight_layout()
plt.savefig(reports_figs / "item_rest_r_combined.png", dpi=200)
plt.close(fig)

# 2a) Stacked response-types (naïve)
resp_n = item_long[item_long["pid"].isin(naive_pids)].copy()
resp_n["resp_type_label"] = resp_n["resp_type"].map({2: "correct", 1: "intuitive", 0: "other"})
ct = (resp_n.groupby(["item_id", "resp_type_label"]).size()
      .unstack(fill_value=0).reindex(index=all_items, fill_value=0))
ct_pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(ct_pct))
for lab in ["correct", "intuitive", "other"]:
    vals = ct_pct[lab].values if lab in ct_pct.columns else np.zeros(len(ct_pct))
    ax.bar(range(len(ct_pct)), vals, bottom=bottom, label=lab)
    bottom = bottom + vals
ax.set_xticks(range(len(ct_pct)))
ax.set_xticklabels(ct_pct.index.tolist(), rotation=45, ha="right")
ax.set_ylabel("Proportion")
ax.set_title("Response-type proportions (naïve)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(reports_figs / "response_types_stacked_naive.png", dpi=200)
plt.close(fig)

# 2b) Stacked response-types (full)
resp_f = item_long.copy()
resp_f["resp_type_label"] = resp_f["resp_type"].map({2: "correct", 1: "intuitive", 0: "other"})
ct_f = (resp_f.groupby(["item_id", "resp_type_label"]).size()
        .unstack(fill_value=0).reindex(index=all_items, fill_value=0))
ct_f_pct = ct_f.div(ct_f.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(ct_f_pct))
for lab in ["correct", "intuitive", "other"]:
    vals = ct_f_pct[lab].values if lab in ct_f_pct.columns else np.zeros(len(ct_f_pct))
    ax.bar(range(len(ct_f_pct)), vals, bottom=bottom, label=lab)
    bottom = bottom + vals
ax.set_xticks(range(len(ct_f_pct)))
ax.set_xticklabels(ct_f_pct.index.tolist(), rotation=45, ha="right")
ax.set_ylabel("Proportion")
ax.set_title("Response-type proportions (full)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(reports_figs / "response_types_stacked_full.png", dpi=200)
plt.close(fig)

# 3) Scatter diagnostics with labels (naïve)
# r vs intuitive%
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_naive["r_itemrest_comb"], stats_naive["pct_intuitive_errors"], s=40)
_label_points(ax, stats_naive["r_itemrest_comb"], stats_naive["pct_intuitive_errors"], stats_naive["item"])
ax.set_xlabel("Rested item–total r (Combined10, naïve)")
ax.set_ylabel("Intuitive error proportion (naïve)")
ax.set_title("Intuitive errors vs discrimination")
plt.tight_layout()
plt.savefig(reports_figs / "scatter_r_vs_intuitive.png", dpi=200)
plt.close(fig)

# r vs difficulty p
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_naive["r_itemrest_comb"], stats_naive["p"], s=40)
_label_points(ax, stats_naive["r_itemrest_comb"], stats_naive["p"], stats_naive["item"])
ax.set_xlabel("Rested item–total r (Combined10, naïve)")
ax.set_ylabel("Difficulty p (naïve)")
ax.set_title("Difficulty vs discrimination")
plt.tight_layout()
plt.savefig(reports_figs / "scatter_r_vs_p.png", dpi=200)
plt.close(fig)

# exposure Δp vs p
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_naive["exposure_delta_p"], stats_naive["p"], s=40)
_label_points(ax, stats_naive["exposure_delta_p"], stats_naive["p"], stats_naive["item"])
ax.axvline(0.0, linewidth=1)
ax.set_xlabel("Exposure inflation Δp (full: seen − unseen)")
ax.set_ylabel("Difficulty p (naïve)")
ax.set_title("Exposure inflation vs difficulty")
plt.tight_layout()
plt.savefig(reports_figs / "scatter_exposure_vs_p.png", dpi=200)
plt.close(fig)

# 3b) Naïve vs full comparisons
# r comparison
fig, ax = plt.subplots(figsize=(6.5, 5))
merged_rf = stats_naive.merge(stats_full, on="item", how="inner", suffixes=("_naive","_full"))
ax.scatter(merged_rf["r_itemrest_comb_naive"], merged_rf["r_itemrest_comb_full"], s=40)
for i, row in merged_rf.iterrows():
    if pd.notna(row["r_itemrest_comb_naive"]) and pd.notna(row["r_itemrest_comb_full"]):
        ax.annotate(row["item"], (row["r_itemrest_comb_naive"], row["r_itemrest_comb_full"]),
                    textcoords="offset points", xytext=(3,3), fontsize=8)
lims = [np.nanmin([merged_rf["r_itemrest_comb_naive"].min(), merged_rf["r_itemrest_comb_full"].min(), -0.05]),
        np.nanmax([merged_rf["r_itemrest_comb_naive"].max(), merged_rf["r_itemrest_comb_full"].max(), 1.0])]
ax.plot(lims, lims, linestyle="--", linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Rested r (naïve)")
ax.set_ylabel("Rested r (full)")
ax.set_title("Discrimination: naïve vs full")
plt.tight_layout()
plt.savefig(reports_figs / "compare_r_naive_vs_full.png", dpi=200)
plt.close(fig)

# p comparison
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(merged_rf["p_naive"], merged_rf["p_full"], s=40)
for i, row in merged_rf.iterrows():
    if pd.notna(row["p_naive"]) and pd.notna(row["p_full"]):
        ax.annotate(row["item"], (row["p_naive"], row["p_full"]),
                    textcoords="offset points", xytext=(3,3), fontsize=8)
lims = [np.nanmin([merged_rf["p_naive"].min(), merged_rf["p_full"].min(), 0.0]),
        np.nanmax([merged_rf["p_naive"].max(), merged_rf["p_full"].max(), 1.0])]
ax.plot(lims, lims, linestyle="--", linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Difficulty p (naïve)")
ax.set_ylabel("Difficulty p (full)")
ax.set_title("Difficulty: naïve vs full")
plt.tight_layout()
plt.savefig(reports_figs / "compare_p_naive_vs_full.png", dpi=200)
plt.close(fig)

# 4) Empirical ICCs (naïve): deciles of rested Combined10 (z)
if naive_pids:
    rest_all = (tot_comb - pivot).loc[naive_pids]  # each column is rested w.r.t that item
    for item in all_items:
        abil = rest_all[item].astype(float)
        s = abil.std(ddof=1)
        z = (abil - abil.mean()) / (s if s else 1.0)
        # deciles
        q = np.linspace(0, 1, 11)
        cuts = np.unique(np.quantile(z, q))
        if len(cuts) < 3:  # guard
            continue
        bins = pd.cut(z, bins=cuts, include_lowest=True, duplicates='drop')
        p = pivot.loc[naive_pids, item].reindex(z.index).groupby(bins).mean()
        centers = [b.mid for b in p.index.categories] if hasattr(p.index, 'categories') else np.arange(len(p))

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(centers, p.values, marker="o")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Ability (rested Combined10, z; deciles)")
        ax.set_ylabel("P(correct)")
        ax.set_title(f"Empirical ICC (naïve): {item}")
        plt.tight_layout()
        plt.savefig(fig_icc_dir / f"icc_{item}.png", dpi=200)
        plt.close(fig)

print(f"Saved figures to {reports_figs} and per-item ICCs to {fig_icc_dir}")