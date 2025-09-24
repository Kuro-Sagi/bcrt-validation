import yaml, pandas as pd, numpy as np
from pathlib import Path
from utils import kr20

# ------------------ config & paths ------------------
cfg = yaml.safe_load(open("config/config.yaml"))
interim_dir = Path(cfg["paths"]["interim"]) 
processed_dir = Path(cfg["paths"]["processed"]) 
reports_dir = Path(cfg["paths"]["reports"]) 
reports_tables = reports_dir / "tables"
reports_tables.mkdir(parents=True, exist_ok=True)

crt2 = cfg["columns"]["crt2_items"]
bcrt = cfg["columns"]["bcrt_items"]
all_items = crt2 + bcrt
B = int(cfg.get("analysis", {}).get("bootstrap_resamples", 0) or 0)

# ------------------ data ------------------
item_long = pd.read_csv(interim_dir / "scored_long.csv")
pivot = (item_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first")
         .reindex(columns=all_items).fillna(0).astype(int))

# naïve split (prefer precomputed file; otherwise derive)
naive_file = processed_dir / "person_naive.csv"
if naive_file.exists():
    naive_pids = pd.read_csv(naive_file)["pid"].tolist()
else:
    seen_by_pid = item_long.groupby("pid")["seen"].max()
    naive_pids = seen_by_pid[seen_by_pid == 0].index.tolist()

full_pids = pivot.index.tolist()

# ------------------ helpers ------------------
def _kr20_ci_boot(block: pd.DataFrame, B: int):
    """Percentile bootstrap CI for KR-20 over persons (rows). Returns (lo, hi)."""
    if B <= 0 or block.shape[0] < 2:
        return (np.nan, np.nan)
    n = block.shape[0]
    vals = []
    for _ in range(B):
        idx = np.random.randint(0, n, size=n)
        vals.append(kr20(block.iloc[idx]))
    lo, hi = (np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5))
    return (float(lo), float(hi))

def _try_parse_kept_from_decision(df: pd.DataFrame):
    dfc = df.copy()
    cols = [c.lower() for c in dfc.columns]
    dfc.columns = cols
    if "item" in cols:
        for flag in ["keep", "kept", "selected"]:
            if flag in cols:
                return dfc.loc[dfc[flag].astype(str).str.lower().isin(["1","true","yes","y"]) , "item"].tolist()
    # single-row field like 'kept_items'
    for c in cols:
        if "kept" in c and isinstance(dfc[c].iloc[0], str):
            items = [x.strip() for x in str(dfc[c].iloc[0]).split(",") if x.strip()]
            if items:
                return items
    return []

def infer_bcrt4_items():
    """Try to find 4 kept BCRT items from selection files; fallback to top-4 by anchored r."""
    src_note = ""
    # 1) selection_decision.csv
    f = reports_tables / "selection_decision.csv"
    if f.exists():
        try:
            kept = _try_parse_kept_from_decision(pd.read_csv(f))
            kept = [x for x in kept if x in bcrt]
            if len(kept) == 4:
                return kept, "selection_decision.csv"
        except Exception:
            pass
    # 2) selection_scores.csv
    f = reports_tables / "selection_scores.csv"
    if f.exists():
        try:
            ss = pd.read_csv(f)
            # prefer explicit keep/selected flags
            for flag in ["keep","selected","kept"]:
                if flag in ss.columns:
                    kept = ss.loc[ss[flag].astype(str).str.lower().isin(["1","true","yes","y"]) , "item"].tolist()
                    kept = [x for x in kept if x in bcrt]
                    if len(kept) == 4:
                        return kept, "selection_scores.csv keep-flag"
            # else fall back to top-4 score
            score_col = None
            for c in ["selection_score","score","r_itemrest_crt2","r_itemrest_comb"]:
                if c in ss.columns:
                    score_col = c; break
            if score_col is not None:
                sub = ss[ss["item"].isin(bcrt)].sort_values(score_col, ascending=False).head(4)
                if len(sub) == 4:
                    return sub["item"].tolist(), f"selection_scores.csv top4 by {score_col}"
        except Exception:
            pass
    # 3) per_item_stats.csv (anchored r)
    f = reports_tables / "per_item_stats.csv"
    if f.exists():
        pis = pd.read_csv(f)
        if "r_itemrest_crt2" in pis.columns:
            sub = pis[pis["item"].isin(bcrt)].copy()
            # tie-break: higher intuitive%, then closeness to p=0.5
            if "pct_intuitive_errors" in sub.columns and "p_naive" in sub.columns:
                sub["tie"] = (sub["pct_intuitive_errors"].fillna(0)) - (sub["p_naive"] - 0.5).abs()
            else:
                sub["tie"] = 0
            sub = sub.sort_values(["r_itemrest_crt2","tie"], ascending=[False, False]).head(4)
            if len(sub) == 4:
                return sub["item"].tolist(), "per_item_stats top4 by r_itemrest_crt2"
    return [], ""

bcrt4_items, bcrt4_source = infer_bcrt4_items()
if bcrt4_items:
    pd.DataFrame({"bcrt4_items": bcrt4_items, "source": [bcrt4_source]*len(bcrt4_items)}).to_csv(
        reports_tables / "bcrt4_items_inferred.csv", index=False)

# ------------------ reliability blocks ------------------

def _rel_block(block: pd.DataFrame, label: str, sample: str):
    k = block.shape[1]
    n = block.shape[0]
    alpha = kr20(block)
    lo, hi = _kr20_ci_boot(block, B)
    return {"sample": sample, "label": label, "k": k, "n": n, "KR20": alpha, "KR20_lo": lo, "KR20_hi": hi}

rows = []

# FULL sample
base_full = pivot.copy()
rows.append(_rel_block(base_full[crt2], "CRT-2", "full"))
rows.append(_rel_block(base_full[bcrt], "BCRT-6", "full"))
rows.append(_rel_block(base_full[all_items], "Combined-10", "full"))
if bcrt4_items:
    rows.append(_rel_block(base_full[bcrt4_items], "BCRT-4", "full"))
    rows.append(_rel_block(base_full[crt2 + bcrt4_items], "Combined-8", "full"))

# FULL: drop-one deltas (CRT2 and BCRT)
base_alpha_full = kr20(base_full[all_items])
for item in crt2:
    alt = base_full[all_items].drop(columns=[item])
    rows.append({"sample":"full","label":f"Combined-10 minus {item}",
                 "k": alt.shape[1], "n": alt.shape[0], "KR20": kr20(alt),
                 "Delta_from_base": kr20(alt) - base_alpha_full})
for item in bcrt:
    alt = base_full[all_items].drop(columns=[item])
    rows.append({"sample":"full","label":f"Combined-10 minus {item}",
                 "k": alt.shape[1], "n": alt.shape[0], "KR20": kr20(alt),
                 "Delta_from_base": kr20(alt) - base_alpha_full})
if bcrt4_items:
    base_c8 = base_full[crt2 + bcrt4_items]
    base_c8_alpha = kr20(base_c8)
    for item in bcrt4_items:
        alt = base_c8.drop(columns=[item])
        rows.append({"sample":"full","label":f"Combined-8 minus {item}",
                     "k": alt.shape[1], "n": alt.shape[0], "KR20": kr20(alt),
                     "Delta_from_base": kr20(alt) - base_c8_alpha})

# NAÏVE sample
if naive_pids:
    base_naive = pivot.loc[naive_pids]
    rows.append(_rel_block(base_naive[crt2], "CRT-2", "naive"))
    rows.append(_rel_block(base_naive[bcrt], "BCRT-6", "naive"))
    rows.append(_rel_block(base_naive[all_items], "Combined-10", "naive"))
    if bcrt4_items:
        rows.append(_rel_block(base_naive[bcrt4_items], "BCRT-4", "naive"))
        rows.append(_rel_block(base_naive[crt2 + bcrt4_items], "Combined-8", "naive"))

# write
out = pd.DataFrame(rows)
out.to_csv(reports_tables / "reliability.csv", index=False)
print("Saved reliability.csv with full & naïve KR-20, drop-one deltas, and optional BCRT-4/Combined-8.")
