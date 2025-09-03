import yaml, pandas as pd, numpy as np
from pathlib import Path
from utils import point_biserial
import matplotlib.pyplot as plt

# Helper: label scatter points with item ids
def _label_points(ax, x, y, labels):
    import numpy as _np
    for xi, yi, lab in zip(x, y, labels):
        if _np.isnan(xi) or _np.isnan(yi):
            continue
        ax.annotate(str(lab), (xi, yi), textcoords="offset points", xytext=(3, 3), ha="left", fontsize=8)

cfg = yaml.safe_load(open("config/config.yaml"))
interim_dir = Path(cfg["paths"]["interim"])
processed_dir = Path(cfg["paths"]["processed"])
reports = Path(cfg["paths"]["reports"])
reports_tables = Path(reports) / "tables"
reports_tables.mkdir(parents=True, exist_ok=True)
reports_figs = Path(reports) / "figures"
reports_figs.mkdir(parents=True, exist_ok=True)
fig_icc_dir = reports_figs / "icc"
fig_icc_dir.mkdir(parents=True, exist_ok=True)

crt2 = cfg["columns"]["crt2_items"]
bcrt = cfg["columns"]["bcrt_items"]
all_items = crt2 + bcrt

item_long = pd.read_csv(interim_dir / "scored_long.csv")
person_naive = pd.read_csv(processed_dir / "person_naive.csv")
person_full = pd.read_csv(processed_dir / "person_full.csv")

# Construct totals for resting
# Rebuild from item_long to ensure consistency
pivot = item_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first")
pivot = pivot.reindex(columns=all_items)
pivot = pivot.fillna(0).astype(int)
tot_crt2 = pivot[crt2].sum(axis=1)
tot_bcrt = pivot[bcrt].sum(axis=1)
tot_comb = tot_crt2 + tot_bcrt

rows = []
for item in all_items:
    # Naive subset
    naive_pids = person_naive["pid"].values
    p_naive = pivot.loc[naive_pids, item].mean()
    # Rested totals
    rest_comb_naive = tot_comb.loc[naive_pids] - pivot.loc[naive_pids, item]
    rest_crt2_naive = tot_crt2.loc[naive_pids] - (pivot.loc[naive_pids, item] if item in crt2 else 0)
    r_comb = point_biserial(pivot.loc[naive_pids, item], rest_comb_naive)
    r_crt2 = point_biserial(pivot.loc[naive_pids, item], rest_crt2_naive)
    # Intuitive proportion among errors
    err_mask = (item_long["item_id"]==item) & (item_long["pid"].isin(naive_pids)) & (item_long["correct"]==0)
    n_err = err_mask.sum()
    n_intuit = item_long.loc[err_mask, "resp_type"].eq(1).sum()
    pct_intuit = (n_intuit / n_err) if n_err>0 else np.nan
    # Exposure inflation on full set
    sub_full = item_long[item_long["item_id"]==item]
    p_seen = sub_full[sub_full["seen"]==1]["correct"].mean()
    p_unseen = sub_full[sub_full["seen"]==0]["correct"].mean()
    delta_p = (p_seen - p_unseen) if (not np.isnan(p_seen) and not np.isnan(p_unseen)) else np.nan
    rows.append({
        "item": item, "p_naive": p_naive, "r_itemrest_comb": r_comb, "r_itemrest_crt2": r_crt2,
        "pct_intuitive_errors": pct_intuit, "exposure_delta_p": delta_p
    })

stats_df = pd.DataFrame(rows)
stats_df.to_csv(reports_tables / "per_item_stats.csv", index=False)
print("Saved per_item_stats.csv")

# --- Full-sample unseen-only metrics (pooled across all participants who have NOT seen the item) ---
rows_unseen = []
for item in all_items:
    mask_u = (item_long["item_id"] == item) & (item_long["seen"] == 0)
    unseen_pids = item_long.loc[mask_u, "pid"].unique()
    if len(unseen_pids) == 0:
        rows_unseen.append({"item": item, "p_unseen": np.nan, "r_itemrest_comb_unseen": np.nan,
                            "pct_intuitive_errors_unseen": np.nan})
        continue
    p_unseen = pivot.loc[unseen_pids, item].mean()
    rest_comb_unseen = (tot_comb - pivot[item]).loc[unseen_pids]
    r_unseen = point_biserial(pivot.loc[unseen_pids, item], rest_comb_unseen)
    sub_u = item_long.loc[mask_u]
    n_err_u = (sub_u["correct"] == 0).sum()
    n_intuit_u = (sub_u["resp_type"] == 1).sum()
    pct_intuit_u = (n_intuit_u / n_err_u) if n_err_u > 0 else np.nan
    rows_unseen.append({
        "item": item,
        "p_unseen": p_unseen,
        "r_itemrest_comb_unseen": r_unseen,
        "pct_intuitive_errors_unseen": pct_intuit_u
    })

stats_unseen = pd.DataFrame(rows_unseen)
stats_both = stats_df.merge(stats_unseen, on="item", how="left")
stats_both.to_csv(reports_tables / "per_item_stats_with_unseen.csv", index=False)
print("Saved per_item_stats_with_unseen.csv (naïve + unseen-only full)")

# === Figures ===
# 1) Forest/bar plot of rested item–total r on Combined10
sd = stats_df.sort_values("r_itemrest_comb")
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(sd["item"], sd["r_itemrest_comb"])
ax.set_xlabel("Rested item–total r (Combined10)")
ax.set_ylabel("Item")
ax.set_title("Item discrimination")
plt.tight_layout()
(reports_figs / "item_rest_r_combined.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "item_rest_r_combined.png", dpi=200)
plt.close(fig)

# 2) Stacked proportions of response types (naïve set only)
naive_pids_set = set(person_naive["pid"]) if "pid" in person_naive.columns else set()
resp = item_long[item_long["pid"].isin(naive_pids_set)].copy()
resp["resp_type_label"] = resp["resp_type"].map({2: "correct", 1: "intuitive", 0: "other"})
ct = (resp.groupby(["item_id", "resp_type_label"]).size()
         .unstack(fill_value=0)
         .reindex(index=all_items, fill_value=0))
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
ax.set_title("Response-type proportions (naïve set)")
ax.legend(frameon=False)
plt.tight_layout()
(reports_figs / "response_types_stacked.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "response_types_stacked.png", dpi=200)
plt.close(fig)

# 2b) Stacked proportions for unseen-only across full sample
resp_u = item_long[item_long["seen"] == 0].copy()
resp_u["resp_type_label"] = resp_u["resp_type"].map({2: "correct", 1: "intuitive", 0: "other"})
ct_u = (resp_u.groupby(["item_id", "resp_type_label"]).size()
           .unstack(fill_value=0)
           .reindex(index=all_items, fill_value=0))
ct_u_pct = ct_u.div(ct_u.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(ct_u_pct))
for lab in ["correct", "intuitive", "other"]:
    vals = ct_u_pct[lab].values if lab in ct_u_pct.columns else np.zeros(len(ct_u_pct))
    ax.bar(range(len(ct_u_pct)), vals, bottom=bottom, label=lab)
    bottom = bottom + vals
ax.set_xticks(range(len(ct_u_pct)))
ax.set_xticklabels(ct_u_pct.index.tolist(), rotation=45, ha="right")
ax.set_ylabel("Proportion")
ax.set_title("Response-type proportions (unseen-only, full sample)")
ax.legend(frameon=False)
plt.tight_layout()
(reports_figs / "response_types_stacked_unseen_full.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "response_types_stacked_unseen_full.png", dpi=200)
plt.close(fig)

print(f"Saved figures to {reports_figs}")

# 3) Scatter diagnostics
# r vs intuitive%
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_df["r_itemrest_comb"], stats_df["pct_intuitive_errors"], s=40)
_label_points(ax, stats_df["r_itemrest_comb"].values, stats_df["pct_intuitive_errors"].values, stats_df["item"].values)
ax.set_xlabel("Rested item–total r (Combined10)")
ax.set_ylabel("Intuitive error proportion")
ax.set_title("Intuitive errors vs. discrimination")
plt.tight_layout()
(reports_figs / "scatter_r_vs_intuitive.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "scatter_r_vs_intuitive.png", dpi=200)
plt.close(fig)

# r vs difficulty p
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_df["r_itemrest_comb"], stats_df["p_naive"], s=40)
_label_points(ax, stats_df["r_itemrest_comb"].values, stats_df["p_naive"].values, stats_df["item"].values)
ax.set_xlabel("Rested item–total r (Combined10)")
ax.set_ylabel("Difficulty p (naïve)")
ax.set_title("Difficulty vs. discrimination")
plt.tight_layout()
(reports_figs / "scatter_r_vs_p.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "scatter_r_vs_p.png", dpi=200)
plt.close(fig)

# exposure inflation vs p
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_df["exposure_delta_p"], stats_df["p_naive"], s=40)
_label_points(ax, stats_df["exposure_delta_p"].values, stats_df["p_naive"].values, stats_df["item"].values)
ax.axvline(0.0, color="black", linewidth=1)
ax.set_xlabel("Exposure inflation Δp (seen − unseen)")
ax.set_ylabel("Difficulty p (naïve)")
ax.set_title("Exposure inflation vs. difficulty")
plt.tight_layout()
(reports_figs / "scatter_exposure_vs_p.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "scatter_exposure_vs_p.png", dpi=200)
plt.close(fig)

# 3b) Comparison plots: naïve vs unseen-only
# r comparison
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_both["r_itemrest_comb"], stats_both["r_itemrest_comb_unseen"], s=40)
lims = [np.nanmin([ax.get_xlim()[0], ax.get_ylim()[0], -0.05]), np.nanmax([ax.get_xlim()[1], ax.get_ylim()[1], 1.0])]
ax.plot(lims, lims, linestyle="--", linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Rested r (naïve)")
ax.set_ylabel("Rested r (unseen-only full)")
ax.set_title("Discrimination: naïve vs unseen-only")
plt.tight_layout()
(reports_figs / "compare_r_naive_vs_unseen.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "compare_r_naive_vs_unseen.png", dpi=200)
plt.close(fig)

# p comparison
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(stats_both["p_naive"], stats_both["p_unseen"], s=40)
lims = [np.nanmin([ax.get_xlim()[0], ax.get_ylim()[0], 0.0]), np.nanmax([ax.get_xlim()[1], ax.get_ylim()[1], 1.0])]
ax.plot(lims, lims, linestyle="--", linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Difficulty p (naïve)")
ax.set_ylabel("Difficulty p (unseen-only full)")
ax.set_title("Difficulty: naïve vs unseen-only")
plt.tight_layout()
(reports_figs / "compare_p_naive_vs_unseen.png").unlink(missing_ok=True)
plt.savefig(reports_figs / "compare_p_naive_vs_unseen.png", dpi=200)
plt.close(fig)

# 4) Empirical ICCs (no IRT fit): p(correct) by ability deciles (rested Combined10), naïve set
naive_pids = set(person_naive["pid"]) if "pid" in person_naive.columns else set()
rest_score = (tot_comb - pivot[all_items]).loc[list(naive_pids)]  # rested per item

for item in all_items:
    # ability proxy = rested Combined10 (z)
    ability = rest_score[item].astype(float)
    if ability.empty:
        continue
    z = (ability - ability.mean()) / (ability.std(ddof=1) if ability.std(ddof=1) else 1.0)
    # deciles
    q = np.linspace(0, 1, 11)
    cuts = np.quantile(z, q)
    # handle potential duplicate cutpoints
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        continue
    bins = pd.cut(z, bins=cuts, include_lowest=True, duplicates='drop')
    y = pivot.loc[list(naive_pids), item].reindex(z.index)
    icc = y.groupby(bins).mean()
    centers = [b.mid for b in icc.index.categories] if hasattr(icc.index, 'categories') else np.arange(len(icc))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(centers, icc.values, marker="o")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Ability (rested Combined10, z)")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"Empirical ICC: {item}")
    plt.tight_layout()
    out = fig_icc_dir / f"icc_{item}.png"
    out.unlink(missing_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)

print(f"Saved additional figures to {reports_figs} and per-item ICCs to {fig_icc_dir}")
