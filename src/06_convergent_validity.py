# src/07_liwc_families.py  
# Comprehensive LIWC/NFC convergent validity analysis
# Includes proper NFC scoring (9 forward, 9 reverse items) and LIWC family regressions
# with HC3 SEs, 95% CI, standardised betas, and family-wise adjustment (BH-FDR).
# Outputs:
#   reports/tables/liwc_predict_families_combined.csv (both datasets)
#   reports/tables/liwc_predict_families_full.csv    (full dataset)  
#   reports/tables/liwc_predict_families_naive.csv   (naive dataset)
#   reports/tables/nfc_correlations.csv              (NFC-CRT correlations)

import os
import pandas as pd, numpy as np, statsmodels.api as sm
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path(".")
PROCESSED = BASE/"data"/"processed"
INTERIM = BASE/"data"/"interim"
TABLES = BASE/"reports"/"tables"
FIGS = BASE/"reports"/"figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Dedicated folders for Mosleh-only LIWC analysis
TABLES_MOS = TABLES / "mosleh_liwc"
FIGS_MOS   = FIGS   / "mosleh_liwc"
TABLES_MOS.mkdir(parents=True, exist_ok=True)
FIGS_MOS.mkdir(parents=True, exist_ok=True)

# Figure roots for GLMM visualisations
FIGS_GLMM = FIGS / "glmm_families"
FIGS_GLMM.mkdir(parents=True, exist_ok=True)
FIGS_MOS_GLMM = FIGS_MOS / "glmm"
FIGS_MOS_GLMM.mkdir(parents=True, exist_ok=True)

# --- NFC scoring functions ---
def correct_nfc_scoring(df):
    """Apply correct NFC scoring: 9 forward, 9 reverse items"""
    # Correct key: 9 forward, 9 reverse
    forward = [1,2,6,10,11,13,14,15,18]
    reverse = [3,4,5,7,8,9,12,16,17]
    items = [f"nfc_{i}" for i in range(1,19)]
    
    label_to_num = {
        '1 = Extremely Uncharacteristic': 1,
        '2 = Somewhat Uncharacteristic': 2,
        '3 = Uncertain': 3,
        '4 = Somewhat Characteristic': 4,
        '5 = Extremely Characteristic': 5,
    }
    
    # map labels -> 1..5
    num = pd.DataFrame({c: df[c].map(label_to_num) for c in items if c in df.columns})
    
    # reverse only the negative ones
    for i in reverse:
        col = f"nfc_{i}"
        if col in num.columns:
            num[col] = 6 - num[col]
    
    return num.sum(axis=1)

# --------------------------- Families (revised) ---------------------------
FAMILY_SUMMARY = ["Analytic","Clout","Authentic","Tone"]
FAMILY_QTY     = ["WC","WPS","Quant","Number"]
FAMILY_FUNC    = ['article','prep','i','ipron','auxverb','conj','adj']
FAMILY_COG     = ['insight','cause','discrep','tentat','differ','certitude']
FAMILY_AFFECT  = ['emo_pos','emo_neg']

# New families: Time Focus and Motivation
FAMILY_TIME    = ['focuspast','focuspresent']
FAMILY_MOT     = ['want','risk']


# Mosleh et al. (2021) – targeted LIWC variables
# We’ll include variables only if present in the processed data.
# Standard LIWC names we expect: 'insight', 'inhib' (if computed),
# 'emo_neg' (~Negemo), 'emo_pos' (~Posemo). 'moral' and 'political' are
# non-standard in LIWC2015; include them here so they are picked up if you
# computed them upstream.
MOSLEH_TARGETS = [
    'insight',
    'inhib',        # may be missing depending on your LIWC export
    'emo_neg',
    'emo_pos',
    'moral',        # non-standard; will be skipped if absent
    'political'     # non-standard; will be skipped if absent
]
MOSLEH_PREDICTORS = ['crt2_total']  # mirror Mosleh: CRT2 only

# Which families get BH–FDR?  (your policy)
FDR_POLICY = {
    "Summary": True,
    "QuantityLength": True,
    "FunctionAnalytic": True,
    "CognitiveComplexity": True,
    "ToneAffect": True,
    "TimeFocus": True,
    "Motivation": True,
}

# Always run both datasets (no need to ever do just one)
DATASETS = ["full", "naive"]
# Exposure covariate included for FULL runs if present
EXPOSURE_COL = "seen_count"

# Predictor sets
# NOTE: Combined sets use a SINGLE composite predictor column so that
# modelling and figures/tables treat them as one predictor rather than
# two separate coefficients.
PRED_SETS = {
    "crt2_total": ["crt2_total"],
    "bcrt_core3": ["bcrt_core3_total"],
    "bcrt_core4": ["bcrt_core4_total"],
    "crt2_plus_core3": ["crt2_plus_core3_total"],
    "crt2_plus_core4": ["crt2_plus_core4_total"],
}

# ---------------------- Core-3/Core-4 dynamic selection support ----------------------
SEL_DEFAULT_PATHS = [TABLES/"selection_decision.csv", TABLES/"selection_rule"/"selection_decision.csv", BASE/"selection_decision.csv"]

def _norm_item_name(x: str|None) -> str|None:
    if x is None: return None
    s = str(x).strip().lower()
    if not s: return None
    if s.startswith("bcrt_q"): return s
    m = re.search(r"(\d+)", s)
    return f"bcrt_q{int(m.group(1))}" if m else None


def load_core_item_sets():
    """Return dict with keys 'core3' and 'core4' listing item_id names from
    reports/tables/selection_decision.csv produced by 08_selection_rule.py.
    If file is missing or malformed, return {} and let callers skip Core sets.
    """
    sel_path = None
    for p in SEL_DEFAULT_PATHS:
        if p.exists():
            sel_path = p; break
    if sel_path is None:
        try:
            sel_path = next(BASE.rglob("selection_decision.csv"))
        except StopIteration:
            sel_path = None
    if sel_path is None:
        print("[selection] selection_decision.csv not found; Core3/Core4 totals unavailable.")
        return {}
    try:
        sel = pd.read_csv(sel_path)
    except Exception as e:
        print(f"[selection] Could not read {sel_path}: {e}")
        return {}
    cols = [c.lower() for c in sel.columns]
    sel.columns = cols
    if not ("label" in sel.columns and "keep_items" in sel.columns):
        print(f"[selection] {sel_path} missing required columns 'label' and 'keep_items'")
        return {}
    out = {"core3": [], "core4": []}
    for _, r in sel.iterrows():
        lab = str(r.get("label", "")).lower()
        items_raw = str(r.get("keep_items", ""))
        items = [it.strip() for it in items_raw.split(",") if it.strip()]
        items = [_norm_item_name(it) for it in items]
        items = [it for it in items if it]
        if "core-3" in lab or lab.endswith("core3") or lab.endswith("core-3"):
            out["core3"] = items
        elif "core-4" in lab or lab.endswith("core4") or lab.endswith("core-4"):
            out["core4"] = items
    print(f"[selection] Loaded core sets from {sel_path}: core3={out.get('core3')}, core4={out.get('core4')}")
    return out


def ensure_core_totals(df_in: pd.DataFrame) -> pd.DataFrame:
    """Ensure core totals and composite totals exist.
    - Builds `bcrt_core3_total` and `bcrt_core4_total` from the dynamically selected
      items in `reports/tables/selection_decision.csv` (if needed).
    - Always computes composite predictors `crt2_plus_core3_total` and
      `crt2_plus_core4_total` when their components are present.
    If the selection file is absent, leaves any missing totals untouched.
    """
    df = df_in.copy()

    # --- Ensure Core3/Core4 totals exist (only if missing) ---
    need3 = ("bcrt_core3_total" not in df.columns)
    need4 = ("bcrt_core4_total" not in df.columns)
    if need3 or need4:
        sets = load_core_item_sets()
        if sets:
            need_items = set(sets.get("core3", []) + sets.get("core4", []))
            missing_items = [c for c in need_items if c not in df.columns]
            if missing_items:
                try:
                    scored_long = pd.read_csv(BASE/"data"/"interim"/"scored_long.csv")
                    wide = scored_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first").reset_index()
                    df = df.merge(wide, on="pid", how="left")
                except Exception as e:
                    print(f"[selection] Could not merge scored_long for core totals: {e}")
            # construct totals from whatever items are now present
            if need3 and sets.get("core3"):
                have = [c for c in sets["core3"] if c in df.columns]
                if have:
                    df["bcrt_core3_total"] = df[have].sum(axis=1)
                else:
                    print("[selection] Warning: none of the Core-3 items found in data; skipping bcrt_core3_total")
            if need4 and sets.get("core4"):
                have = [c for c in sets["core4"] if c in df.columns]
                if have:
                    df["bcrt_core4_total"] = df[have].sum(axis=1)
                else:
                    print("[selection] Warning: none of the Core-4 items found in data; skipping bcrt_core4_total")

    # --- Always compute composite totals when components are present ---
    try:
        if ("crt2_total" in df.columns) and ("bcrt_core3_total" in df.columns) and ("crt2_plus_core3_total" not in df.columns):
            df["crt2_plus_core3_total"] = df["crt2_total"].astype(float) + df["bcrt_core3_total"].astype(float)
        if ("crt2_total" in df.columns) and ("bcrt_core4_total" in df.columns) and ("crt2_plus_core4_total" not in df.columns):
            df["crt2_plus_core4_total"] = df["crt2_total"].astype(float) + df["bcrt_core4_total"].astype(float)
    except Exception as e:
        print(f"[selection] Could not compute composite totals: {e}")

    return df

# --------------------------- Main analysis loop ---------------------------
all_results = []

for DATASET in DATASETS:
    print(f"\n=== Processing {DATASET.upper()} dataset ===")
    
    # Load data
    if DATASET == "naive":
        liwc_path = PROCESSED/"person_naive_with_liwc.csv"
    else:
        liwc_path = PROCESSED/"person_with_liwc.csv"

    if not liwc_path.exists():
        print(f"WARNING: LIWC file not found: {liwc_path}. Skipping {DATASET} dataset.")
        continue

    pl = pd.read_csv(liwc_path)

    # --- LIWC manifest (sanity check of what is present) ---
    def emit_liwc_manifest(df_in: pd.DataFrame, dataset_label: str):
        # Heuristic: numeric columns that are not obvious IDs/scores
        META = {"pid","participant_id","seen_count","NFC_total","nfc_total","crt2_total","bcrt_core3_total","bcrt_core4_total"}
        # include common BCRT item cols and any non-numeric
        META |= set([c for c in df_in.columns if c.startswith("bcrt_")])
        manifest_rows = []
        for c in df_in.columns:
            if c in META:
                continue
            s = df_in[c]
            if not np.issubdtype(s.dtype, np.number):
                continue
            # Capture Q1/Q2 availability for base names
            base = c.replace("_Q1",""
                    ).replace("_Q2","")
            has_q1 = f"{base}_Q1" in df_in.columns
            has_q2 = f"{base}_Q2" in df_in.columns
            manifest_rows.append({
                "name": c,
                "base": base,
                "has_Q1": has_q1,
                "has_Q2": has_q2,
                "dtype": str(s.dtype),
                "n_nonnull": int(s.notna().sum()),
                "mean": float(np.nanmean(s.values)),
                "std": float(np.nanstd(s.values)),
            })
        man = pd.DataFrame(manifest_rows)
        man = man.sort_values(["base","name"]).reset_index(drop=True)
        (TABLES / f"liwc_variables_manifest_{dataset_label}.csv").write_text("")
        man.to_csv(TABLES / f"liwc_variables_manifest_{dataset_label}.csv", index=False)
        print(f"Wrote LIWC manifest for {dataset_label}:", TABLES / f"liwc_variables_manifest_{dataset_label}.csv")
        # quick count
        bases = man["base"].drop_duplicates().tolist()
        print(f"[LIWC manifest] {dataset_label}: {len(bases)} base variables, {len(man)} total numeric LIWC-like cols")
        return man

    _ = emit_liwc_manifest(pl, DATASET)

    # Apply correct NFC scoring if NFC items are present
    nfc_items = [f"nfc_{i}" for i in range(1,19)]
    nfc_present = any(col in pl.columns for col in nfc_items)
    if nfc_present:
        # Backup original NFC_total if it exists
        if "NFC_total" in pl.columns and "NFC_total_raw" not in pl.columns:
            pl["NFC_total_raw"] = pl["NFC_total"]
        
        # Apply correct scoring
        pl["NFC_total"] = correct_nfc_scoring(pl)
        pl["nfc_total"] = pl["NFC_total"]  # Ensure both naming conventions work
        print(f"Applied correct NFC scoring for {DATASET} dataset (9 forward, 9 reverse items)")
    else:
        print(f"No NFC items found in {DATASET} dataset - skipping NFC correction")

    # Add item correctness → core totals using dynamic selection (08_selection_rule)
    try:
        scored_long = pd.read_csv(BASE/"data"/"interim"/"scored_long.csv")
        wide = scored_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first").reset_index()
        df = pl.merge(wide, on="pid", how="left").copy()
    except Exception:
        df = pl.copy()
    # Respect selection_decision.csv; do not fabricate if missing
    df = ensure_core_totals(df)

    # Helpers (defined inside loop to use current df)
    def avail(col: str) -> bool:
        return (col in df.columns) and (df[col].notna().sum() >= 20) and (df[col].std(skipna=True) > 0)

    # OLS with HC3; returns per-term rows incl. CI and standardised beta
    def run_ols(y: str, Xnames: list):
        if not avail(y):  # skip outcomes with no variance
            return 0, np.nan, []
        X = pd.DataFrame({k: df[k].astype(float) for k in Xnames if avail(k)})
        if X.empty:
            return 0, np.nan, []
        if DATASET == "full" and EXPOSURE_COL in df.columns and avail(EXPOSURE_COL):
            X[EXPOSURE_COL] = df[EXPOSURE_COL].astype(float)
        X = sm.add_constant(X)
        Y = df[y].astype(float)
        if Y.std(skipna=True) == 0 or Y.notna().sum() < 10:
            return 0, np.nan, []
        m = sm.OLS(Y, X, missing="drop").fit(cov_type="HC3")
        sdY = Y.std(skipna=True)
        rows = []
        for k in X.columns:
            if k == "const":
                continue
            beta = m.params.get(k, np.nan)
            se   = m.bse.get(k, np.nan)
            ci_l = beta - 1.96*se
            ci_u = beta + 1.96*se
            sdX = df[k].astype(float).std(skipna=True)
            beta_std = beta * (sdX / sdY) if (sdY>0 and sdX>0) else np.nan
            pval = m.pvalues.get(k, np.nan)
            rows.append((k, beta, se, ci_l, ci_u, beta_std, pval))
        return int(m.nobs), float(m.rsquared), rows

    # Build family rows for this dataset
    def outcomes_for(targets: list):
        # support *_Q1/_Q2 if present; otherwise plain
        tgts = [t for t in targets if t in df.columns]
        tgts += [t+"_Q1" for t in targets if t+"_Q1" in df.columns]
        tgts += [t+"_Q2" for t in targets if t+"_Q2" in df.columns]
        return sorted(set(tgts))

    dataset_rows = []
    for fam_name, fam_targets in [
        ("Summary",            FAMILY_SUMMARY),
        ("QuantityLength",     FAMILY_QTY),
        ("FunctionAnalytic",   FAMILY_FUNC),
        ("CognitiveComplexity",FAMILY_COG),
        ("ToneAffect",         FAMILY_AFFECT),
        ("TimeFocus",          FAMILY_TIME),
        ("Motivation",         FAMILY_MOT),
    ]:
        for y in outcomes_for(fam_targets):
            for set_name, Xnames in PRED_SETS.items():
                n, r2, betas = run_ols(y, Xnames)
                for k, b, se, lo, hi, bstd, p in betas:
                    dataset_rows.append({
                        "dataset": DATASET,
                        "family": fam_name,
                        "outcome": y,
                        "set": set_name,
                        "predictor": k,
                        "beta": b,
                        "se": se,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "beta_std": bstd,
                        "p_raw": p,
                        "r2": r2,
                        "n": n,
                    })
    
    all_results.extend(dataset_rows)
    print(f"Completed {DATASET} dataset: {len(dataset_rows)} rows")

# Combine all datasets
out = pd.DataFrame(all_results)

# Family-wise adjustment policy (BH–FDR where enabled; else copy raw)

def fdr_bh(p: np.ndarray, alpha: float = 0.05):
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    # adjusted p-values (monotone BH)
    q = ranked * n / np.arange(1, n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    padj = np.empty_like(p, dtype=float)
    padj[order] = np.clip(q, 0, 1)
    # significance flag at alpha
    thr = (np.arange(1, n+1) / n) * alpha
    keep = ranked <= thr
    sig = np.zeros(n, dtype=bool)
    if keep.any():
        sig[order[:keep.nonzero()[0].max()+1]] = True
    return padj, sig

out["p_adj"] = np.nan
out["sig_raw"] = out["p_raw"] < 0.05
out["sig_adj"] = False

# Apply BH–FDR narrowly: per (dataset, family, set, predictor)
for ds in out["dataset"].dropna().unique():
    for fam in out["family"].dropna().unique():
        for s in out["set"].dropna().unique():
            for pred in out["predictor"].dropna().unique():
                mask = (out["dataset"]==ds) & (out["family"]==fam) & (out["set"]==s) & (out["predictor"]==pred)
                if not mask.any():
                    continue
                if FDR_POLICY.get(fam, True):
                    adj, sig = fdr_bh(out.loc[mask, "p_raw"].values, alpha=0.05)
                    out.loc[mask, "p_adj"] = adj
                    out.loc[mask, "sig_adj"] = sig
                else:
                    out.loc[mask, "p_adj"] = out.loc[mask, "p_raw"]
                    out.loc[mask, "sig_adj"] = out.loc[mask, "sig_raw"]

# Save combined results and individual dataset files
out.to_csv(TABLES / "liwc_predict_families_combined.csv", index=False)
print("Wrote", TABLES / "liwc_predict_families_combined.csv")

# Also save individual dataset files for backward compatibility
for dataset in DATASETS:
    dataset_data = out[out["dataset"] == dataset]
    if not dataset_data.empty:
        out_file = TABLES / f"liwc_predict_families_{dataset}.csv"
        dataset_data.to_csv(out_file, index=False)
        print("Wrote", out_file)

# --- Primary family analysis (stacked Q1+Q2, clustered by pid, CRT2-only) ---
def stacked_family_cluster(df_in: pd.DataFrame, base: str, xname: str):
    # Build long data with scenario fixed effect
    frames = []
    for scen in ("Q1","Q2"):
        ycol = f"{base}_{scen}"
        if ycol in df_in.columns:
            tmp = df_in[["pid", xname, ycol]].copy()
            tmp = tmp.rename(columns={ycol: "y"})
            tmp["scenario"] = scen
            frames.append(tmp)
    if not frames:
        # Use non-split column if present
        if base in df_in.columns:
            tmp = df_in[["pid", xname, base]].copy()
            tmp = tmp.rename(columns={base: "y"})
            tmp["scenario"] = "Q1"
            frames = [tmp]
        else:
            return None
    long = pd.concat(frames, axis=0, ignore_index=True)
    long = long[np.isfinite(long["y"]) & np.isfinite(long[xname])]
    if long.empty:
        return None
    X = pd.DataFrame({"const": 1.0, xname: long[xname].astype(float), "scenario_Q2": (long["scenario"]=="Q2").astype(float)})
    y = long["y"].astype(float)
    if X[xname].std(ddof=1)==0 or y.std(ddof=1)==0:
        return None
    m = sm.OLS(y, X, missing="drop").fit(cov_type="cluster", cov_kwds={"groups": long["pid"]})
    beta = float(m.params.get(xname, np.nan)); se=float(m.bse.get(xname, np.nan))
    ci_l = beta - 1.96*se; ci_u = beta + 1.96*se
    sdY = y.std(ddof=1); sdX = X[xname].std(ddof=1)
    beta_std = beta * (sdX/sdY) if (sdY>0 and sdX>0) else np.nan
    pval = float(m.pvalues.get(xname, np.nan))
    return {"base": base, "predictor": xname, "beta": beta, "se": se, "ci_lo": ci_l, "ci_hi": ci_u,
            "beta_std": beta_std, "p_raw": pval, "n": int(m.nobs), "r2": float(m.rsquared)}

fam_rows = []
for DATASET in DATASETS:
    if DATASET == "naive":
        liwc_path = PROCESSED/"person_naive_with_liwc.csv"
    else:
        liwc_path = PROCESSED/"person_with_liwc.csv"
    if not liwc_path.exists():
        continue
    dff = pd.read_csv(liwc_path)
    # Ensure CRT present
    if "crt2_total" not in dff.columns or dff["crt2_total"].std(skipna=True)==0:
        continue
    # Work through families as base-variable lists only
    for fam_name, fam_targets in [
        ("Summary",            FAMILY_SUMMARY),
        ("QuantityLength",     FAMILY_QTY),
        ("FunctionAnalytic",   FAMILY_FUNC),
        ("CognitiveComplexity",FAMILY_COG),
        ("ToneAffect",         FAMILY_AFFECT),
        ("TimeFocus",          FAMILY_TIME),
        ("Motivation",         FAMILY_MOT),
    ]:
        for base in fam_targets:
            res = stacked_family_cluster(dff, base, "crt2_total")
            if res is not None:
                res.update({"dataset": DATASET, "family": fam_name, "set": "crt2_total_stacked"})
                fam_rows.append(res)
fam_df = pd.DataFrame(fam_rows)
if not fam_df.empty:
    fam_df["p_adj"], fam_df["sig_adj"] = np.nan, False
    # FDR per (dataset,family) now only spans the # of bases in that family
    for ds in fam_df["dataset"].dropna().unique():
        for fam in fam_df["family"].dropna().unique():
            mask = (fam_df["dataset"]==ds) & (fam_df["family"]==fam)
            if mask.any():
                adj, sig = fdr_bh(fam_df.loc[mask,"p_raw"].values, alpha=0.05)
                fam_df.loc[mask, "p_adj"] = adj
                fam_df.loc[mask, "sig_adj"] = sig
    fam_df.to_csv(TABLES/"liwc_families_stacked_crt2.csv", index=False)
    print("Wrote primary stacked family analysis:", TABLES/"liwc_families_stacked_crt2.csv")
else:
    print("Stacked family analysis produced no rows (missing columns?)")

#
# ----------------------------- Plot helpers -----------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def forest_plot(df_sub: pd.DataFrame, title: str, out_png: Path, xlab: str = "Log-odds (per unit predictor)"):
    # df_sub needs columns: base, beta, ci_lo, ci_hi, p_adj (or p_raw)
    if df_sub.empty:
        return
    d = df_sub.copy()
    # Use adjusted p when available, else raw
    if "p_adj" in d.columns and d["p_adj"].notna().any():
        p_use = d["p_adj"].fillna(d.get("p_raw", np.nan))
    else:
        p_use = d.get("p_raw", pd.Series(np.nan, index=d.index))
    d = d.assign(p_use=p_use)
    d = d.sort_values("beta", ascending=True)

    n = len(d)
    fig_h = max(2.5, 0.35*n + 1.5)
    fig, ax = plt.subplots(figsize=(6.0, fig_h), dpi=300)
    y = np.arange(n)
    ax.hlines(y, d["ci_lo"], d["ci_hi"], linewidth=1)
    ax.plot(d["beta"], y, "o", markersize=4)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["base"].tolist())
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.grid(axis='x', linestyle=':', linewidth=0.5)

    # significance stars using adjusted p if available
    stars = []
    for p in d["p_use"].fillna(1.0):
        if p < 0.001: stars.append("***")
        elif p < 0.01: stars.append("**")
        elif p < 0.05: stars.append("*")
        else: stars.append("")
    # annotate to the right of points
    for yi, xi, st in zip(y, d["beta"], stars):
        if st:
            ax.text(ax.get_xlim()[1], yi, st, va='center', ha='right', fontsize=8)

    # Legend-like caption
    fig.text(0.01, 0.01, "Points = β (log-odds). Bars = 95% CI. * FDR<.05", fontsize=8)

    _ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def heatmap_signed_logp(df_sub: pd.DataFrame, row_key: str, col_key: str, title: str, out_png: Path, sig_mask_col: str = "sig_adj"):
    # expects columns: row_key, col_key, beta, p_adj/p_raw, and optionally a boolean significance column
    if df_sub.empty:
        return
    d = df_sub.copy()
    p = d["p_adj"].fillna(d.get("p_raw", np.nan)) if "p_adj" in d.columns else d.get("p_raw", np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.sign(d["beta"].values) * (-np.log10(np.clip(p.values, 1e-300, 1)))
    d["val"] = val
    pivot = d.pivot(index=row_key, columns=col_key, values="val")
    # plot
    n_rows, n_cols = pivot.shape
    fig_w = max(6.0, 0.38*n_cols + 2.0)
    fig_h = max(3.0, 0.30*n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    im = ax.imshow(pivot.values, aspect='auto', interpolation='none')
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha='right')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("sign(β) × −log10(p)")
    # overlay stars on significant cells if mask present
    if sig_mask_col in d.columns:
        sig = d.pivot(index=row_key, columns=col_key, values=sig_mask_col).reindex(index=pivot.index, columns=pivot.columns)
        for i in range(n_rows):
            for j in range(n_cols):
                if bool(sig.iloc[i, j]) if sig.notna().iloc[i, j] else False:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=9, color="white", weight="bold")
    # gridlines
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    _ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

# --- Combined heatmap helper: columns = all sets/predictor-terms in fixed order ---
def combined_heatmap(df_in: pd.DataFrame, dataset_label: str, family_label: str, out_dir: Path, mosleh: bool = False):
    if df_in.empty:
        return
    # Build a tidy copy with a canonical column mapping
    d = df_in.copy()
    # Column key is set + predictor term (so combos show both coefficients)
    if "predictor" in d.columns:
        d["set_pred"] = d["set"].astype(str) + ":" + d["predictor"].astype(str)
    else:
        d["set_pred"] = d["set"].astype(str)
    # Desired display order & labels
    # Prefer composite columns if present; otherwise fall back to per-term view.
    has_composite = False
    if "predictor" in d.columns:
        comp_keys = {"crt2_plus_core3:crt2_plus_core3_total", "crt2_plus_core4:crt2_plus_core4_total"}
        present_setpred = set((d["set"].astype(str) + ":" + d["predictor"].astype(str)).unique().tolist())
        has_composite = any(k in present_setpred for k in comp_keys)

    if has_composite:
        desired = [
            ("crt2_total", "crt2_total"),
            ("crt2_plus_core3", "crt2_plus_core3_total"),
            ("crt2_plus_core4", "crt2_plus_core4_total"),
            ("bcrt_core3", "bcrt_core3_total"),
            ("bcrt_core4", "bcrt_core4_total"),
        ]
        labels = [
            "CRT2",
            "CRT2+Core3",
            "CRT2+Core4",
            "Core3",
            "Core4",
        ]
    else:
        desired = [
            ("crt2_total", "crt2_total"),
            ("bcrt_core3", "bcrt_core3_total"),
            ("bcrt_core4", "bcrt_core4_total"),
            ("crt2_plus_core3", "crt2_total"),
            ("crt2_plus_core3", "bcrt_core3_total"),
            ("crt2_plus_core4", "crt2_total"),
            ("crt2_plus_core4", "bcrt_core4_total"),
        ]
        labels = [
            "CRT2",
            "Core3",
            "Core4",
            "CRT2+Core3 (CRT2)",
            "CRT2+Core3 (Core3)",
            "CRT2+Core4 (CRT2)",
            "CRT2+Core4 (Core4)",
        ]
    # Map from set_pred key -> label
    keys = [f"{s}:{p}" for (s,p) in desired]
    present_keys = [k for k in keys if k in (d["set_pred"].unique().tolist())]
    present_labels = [lab for k, lab in zip(keys, labels) if k in present_keys]
    if not present_keys:
        return
    # Filter to requested columns only and relabel
    d = d[d["set_pred"].isin(present_keys)].copy()
    label_map = {k: lab for k, lab in zip(present_keys, present_labels)}
    d["set_label"] = d["set_pred"].map(label_map)
    # Construct title and output path
    title = ("Mosleh targets — " if mosleh else "") + f"Combined heatmap ({family_label}) — {dataset_label}"
    out_png = out_dir / f"{'mosleh_' if mosleh else ''}{family_label}_combined_heatmap.png"
    heatmap_signed_logp(d, row_key="base" if "base" in d.columns else "outcome", col_key="set_label", title=title, out_png=out_png)

# --- Utilities for binomial (and beta-binomial) modelling ---
def liwc_counts_from_percent(df_in: pd.DataFrame, base: str, scen: str):
    # derive counts from percent * WC if raw counts not present
    y_pct = f"{base}_{scen}"
    wc    = f"WC_{scen}" if f"WC_{scen}" in df_in.columns else "WC"
    if y_pct not in df_in.columns or wc not in df_in.columns:
        return None
    p = (df_in[y_pct].astype(float) / 100.0).clip(lower=0, upper=1)
    W = df_in[wc].astype(float).clip(lower=0)
    # integer successes; bound by trials
    succ = np.floor(p * W + 0.5)
    succ = np.minimum(succ, W)
    return pd.DataFrame({"pid": df_in["pid"], "successes": succ, "trials": W})

def make_long_counts(df_in: pd.DataFrame, base: str, xname: str):
    frames = []
    for scen in ("Q1","Q2"):
        tmp = liwc_counts_from_percent(df_in, base, scen)
        if tmp is not None and xname in df_in.columns:
            tmp = tmp.join(df_in[[xname]].astype(float))
            tmp["scenario"] = scen
            frames.append(tmp)
    if not frames:
        # try unsplit
        tmp = liwc_counts_from_percent(df_in, base, "")
        if tmp is None:
            return None
        if xname in df_in.columns:
            tmp = tmp.join(df_in[[xname]].astype(float))
        tmp["scenario"] = "Q1"
        frames = [tmp]
    long = pd.concat(frames, axis=0, ignore_index=True)
    long = long[np.isfinite(long["successes"]) & np.isfinite(long["trials"]) & np.isfinite(long[xname])]
    # guard
    long = long[(long["trials"] > 0)]
    return long

def fit_binom_mixed_R(long_df: pd.DataFrame, xname: str):
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        ro.r("suppressMessages(library(glmmTMB))")
    except Exception as e:
        return None, f"R/glmmTMB unavailable: {e}"
    ro.globalenv["dat"] = pandas2ri.py2rpy(long_df)
    ro.r(f"dat$failures <- pmax(dat$trials - dat$successes, 0)")
    # Binomial GLMM
    try:
        ro.r(f"m1 <- glmmTMB(cbind(successes, failures) ~ {xname} + scenario + (1|pid), family=binomial(), data=dat)")
        res = ro.r("summary(m1)")
        # pull coef for xname
        beta = float(ro.r(f"summary(m1)$coefficients$cond['{xname}','Estimate']"))[0]
        se   = float(ro.r(f"summary(m1)$coefficients$cond['{xname}','Std. Error']"))[0]
        return {"beta": beta, "se": se, "family": "binomial"}, None
    except Exception as e:
        binom_err = str(e)
    # Beta-binomial (overdispersion)
    try:
        ro.r(f"m2 <- glmmTMB(cbind(successes, failures) ~ {xname} + scenario + (1|pid), family=betabinomial(), data=dat)")
        beta = float(ro.r(f"summary(m2)$coefficients$cond['{xname}','Estimate']"))[0]
        se   = float(ro.r(f"summary(m2)$coefficients$cond['{xname}','Std. Error']"))[0]
        return {"beta": beta, "se": se, "family": "betabinomial"}, None
    except Exception as e:
        return None, f"binomial failed ({binom_err}); betabinomial failed ({e})"

def fit_binom_cluster(long_df: pd.DataFrame, xname: str):
    # Cluster-robust binomial GLM as a fallback
    import statsmodels.api as sm
    y = long_df[["successes","trials"]].to_numpy()
    X = sm.add_constant(long_df[[xname,]].astype(float))
    try:
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": long_df["pid"]})
        beta = float(m.params[xname]); se = float(m.bse[xname])
        return {"beta": beta, "se": se, "family": "binomial_cluster"}, None
    except Exception as e:
        return None, str(e)


# --- Multi-predictor helpers (shared) ---
def make_long_counts_multi(df_in: pd.DataFrame, base: str, xnames: list):
    frames = []
    for scen in ("Q1","Q2"):
        tmp = liwc_counts_from_percent(df_in, base, scen)
        if tmp is not None:
            cols = [c for c in xnames if c in df_in.columns]
            if not cols:
                continue
            add = df_in[["pid"] + cols].copy()
            tmp = tmp.merge(add, on="pid", how="left")
            tmp["scenario"] = scen
            frames.append(tmp)
    if not frames:
        tmp = liwc_counts_from_percent(df_in, base, "")
        if tmp is None:
            return None
        cols = [c for c in xnames if c in df_in.columns]
        if not cols:
            return None
        add = df_in[["pid"] + cols].copy()
        tmp = tmp.merge(add, on="pid", how="left")
        tmp["scenario"] = "Q1"
        frames = [tmp]
    long = pd.concat(frames, axis=0, ignore_index=True)
    keep = np.isfinite(long["successes"]) & np.isfinite(long["trials"]) & (long["trials"]>0)
    for xn in xnames:
        if xn in long.columns:
            keep &= np.isfinite(long[xn])
    long = long[keep]
    return long


def fit_binom_mixed_R_multi(long_df: pd.DataFrame, xnames: list):
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        ro.r("suppressMessages(library(glmmTMB))")
    except Exception as e:
        return None, f"R/glmmTMB unavailable: {e}"
    ro.globalenv["dat"] = pandas2ri.py2rpy(long_df)
    ro.r("dat$failures <- pmax(dat$trials - dat$successes, 0)")
    rhs = " + ".join([xn for xn in xnames] + ["scenario"]) if xnames else "scenario"
    try:
        ro.r(f"m1 <- glmmTMB(cbind(successes, failures) ~ {rhs} + (1|pid), family=binomial(), data=dat)")
        coefs = {}
        for xn in xnames:
            beta = float(ro.r(f"summary(m1)$coefficients$cond['{xn}','Estimate']"))[0]
            se   = float(ro.r(f"summary(m1)$coefficients$cond['{xn}','Std. Error']"))[0]
            coefs[xn] = {"beta": beta, "se": se, "family": "binomial"}
        return coefs, None
    except Exception as e:
        binom_err = str(e)
    try:
        ro.r(f"m2 <- glmmTMB(cbind(successes, failures) ~ {rhs} + (1|pid), family=betabinomial(), data=dat)")
        coefs = {}
        for xn in xnames:
            beta = float(ro.r(f"summary(m2)$coefficients$cond['{xn}','Estimate']"))[0]
            se   = float(ro.r(f"summary(m2)$coefficients$cond['{xn}','Std. Error']"))[0]
            coefs[xn] = {"beta": beta, "se": se, "family": "betabinomial"}
        return coefs, None
    except Exception as e:
        return None, f"binomial failed ({binom_err}); betabinomial failed ({e})"


def fit_binom_cluster_multi(long_df: pd.DataFrame, xnames: list):
    import statsmodels.api as sm
    y = long_df[["successes","trials"]].to_numpy()
    cols = [xn for xn in xnames if xn in long_df.columns]
    if not cols:
        return None, "no predictors present"
    X = sm.add_constant(long_df[cols].astype(float))
    try:
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": long_df["pid"]})
        coefs = {}
        for xn in cols:
            beta = float(m.params[xn]); se = float(m.bse[xn])
            coefs[xn] = {"beta": beta, "se": se, "family": "binomial_cluster"}
        return coefs, None
    except Exception as e:
        return None, str(e)

# --- Utility: two-sided p from z, robust to missing SciPy/np.erfc ---
def two_sided_p_from_z(z_vals):
    z = np.asarray(z_vals, dtype=float)
    try:
        # Prefer SciPy if available
        from scipy.special import erfc as sp_erfc  # type: ignore
        return sp_erfc(np.abs(z) / np.sqrt(2.0))
    except Exception:
        # Pure-Python fallback via math.erfc, vectorised
        from math import erfc, sqrt
        return np.vectorize(lambda v: erfc(abs(v) / sqrt(2.0)))(z)

# --- Best-in-class: Counts -> (beta-)binomial mixed models (random intercept pid)
# NOTE: Restrict to families for which a binomial count model is appropriate.
GLMM_FAMILIES = [
    ("FunctionAnalytic",   FAMILY_FUNC),
    ("CognitiveComplexity",FAMILY_COG),
    ("ToneAffect",         FAMILY_AFFECT),
    ("TimeFocus",          FAMILY_TIME),
    ("Motivation",         FAMILY_MOT),
]

fam_glmm_rows = []
for DATASET in DATASETS:
    if DATASET == "naive":
        liwc_path = PROCESSED/"person_naive_with_liwc.csv"
    else:
        liwc_path = PROCESSED/"person_with_liwc.csv"
    if not liwc_path.exists():
        continue
    dff = pd.read_csv(liwc_path)
    if "pid" not in dff.columns:
        continue
    # --- ensure Core3/Core4 totals are available for GLMM predictor sets (from selection_decision) ---
    dff = ensure_core_totals(dff)
    for fam_name, fam_targets in GLMM_FAMILIES:
        for base in fam_targets:
            # only run if base is present for at least one scenario
            if not (f"{base}_Q1" in dff.columns or f"{base}_Q2" in dff.columns or base in dff.columns):
                continue
            for set_name, Xnames in PRED_SETS.items():
                # drop predictors with no variance/presence
                xcols = [x for x in Xnames if x in dff.columns and dff[x].std(skipna=True)>0]
                if not xcols:
                    continue
                long = make_long_counts_multi(dff, base, xcols)
                if long is None or long.empty:
                    continue
                # try GLMM via R/glmmTMB (multi-predictor)
                res, err = fit_binom_mixed_R_multi(long, xcols)
                if res is None:
                    # fallback: binomial GLM with pid-clustered SEs
                    res, err2 = fit_binom_cluster_multi(long, xcols)
                    if res is None:
                        print(f"[GLMM families] {DATASET}/{fam_name}/{base}/{set_name} failed: {err} | {err2}")
                        continue
                for xn, est in res.items():
                    beta = est["beta"]; se = est["se"]
                    ci_l = beta - 1.96*se; ci_u = beta + 1.96*se
                    fam_glmm_rows.append({
                        "dataset": DATASET, "family": fam_name, "base": base,
                        "set": set_name, "predictor": xn,
                        "beta": beta, "se": se, "ci_lo": ci_l, "ci_hi": ci_u,
                        "p_raw": np.nan, "model": est["family"], "n": int(len(long))
                    })

fam_glmm_df = pd.DataFrame(fam_glmm_rows)
if not fam_glmm_df.empty:
    fam_glmm_df["z"] = fam_glmm_df["beta"] / fam_glmm_df["se"]
    fam_glmm_df["p_raw"] = two_sided_p_from_z(fam_glmm_df["z"])
    # FDR per (dataset, family, set, predictor) across base variables only
    fam_glmm_df["p_adj"], fam_glmm_df["sig_adj"] = np.nan, False
    for ds in fam_glmm_df["dataset"].dropna().unique():
        for fam in fam_glmm_df["family"].dropna().unique():
            for s in fam_glmm_df["set"].dropna().unique():
                for pred in fam_glmm_df["predictor"].dropna().unique():
                    mask = (fam_glmm_df["dataset"]==ds) & (fam_glmm_df["family"]==fam) & (fam_glmm_df["set"]==s) & (fam_glmm_df["predictor"]==pred)
                    if mask.any():
                        adj, sig = fdr_bh(fam_glmm_df.loc[mask, "p_raw"].values, alpha=0.05)
                        fam_glmm_df.loc[mask, "p_adj"] = adj
                        fam_glmm_df.loc[mask, "sig_adj"] = sig
    fam_glmm_df.to_csv(TABLES/"liwc_families_glmm_crt_sets.csv", index=False)
    print("Wrote GLMM family analysis (multi-set):", TABLES/"liwc_families_glmm_crt_sets.csv")
else:
    print("GLMM family analysis: no rows (missing counts/WC columns)")

# --- Figures for family GLMMs ---
try:
    if 'fam_glmm_df' in globals() and not fam_glmm_df.empty:
        for ds in fam_glmm_df["dataset"].dropna().unique():
            ds_dir = FIGS_GLMM / ds
            _ensure_dir(ds_dir)
            for fam in fam_glmm_df["family"].dropna().unique():
                sub2 = fam_glmm_df[(fam_glmm_df["dataset"]==ds) & (fam_glmm_df["family"]==fam)].copy()
                if sub2.empty:
                    continue
                # Forest plots per predictor set (clear magnitude view)
                for s in sub2["set"].dropna().unique():
                    sub = sub2[sub2["set"]==s].copy()
                    if sub.empty:
                        continue
                    title = f"{fam} — {s} ({ds})"
                    outp = ds_dir / f"{fam}_{s}_forest.png"
                    forest_plot(sub.rename(columns={"base":"base"}), title, outp)
                # Combined heatmap across all sets/predictor-terms (with sig marks)
                combined_heatmap(sub2, dataset_label=ds, family_label=fam, out_dir=ds_dir, mosleh=False)
except Exception as e:
    print(f"Could not render GLMM family figures: {e}")

# --- Mosleh-only LIWC analysis (focused family; separate outputs) ---
def load_person_df(dataset: str):
    if dataset == "naive":
        liwc_path = PROCESSED / "person_naive_with_liwc.csv"
    else:
        liwc_path = PROCESSED / "person_with_liwc.csv"
    if not liwc_path.exists():
        print(f"[Mosleh] Missing {liwc_path}, skipping {dataset}")
        return None
    pl = pd.read_csv(liwc_path)
    # NFC correction (as above)
    nfc_items = [f"nfc_{i}" for i in range(1,19)]
    if any(col in pl.columns for col in nfc_items):
        if "NFC_total" in pl.columns and "NFC_total_raw" not in pl.columns:
            pl["NFC_total_raw"] = pl["NFC_total"]
        pl["NFC_total"] = correct_nfc_scoring(pl)
        pl["nfc_total"] = pl["NFC_total"]
    # merge per-item correctness and compute totals from selection_decision
    try:
        scored_long = pd.read_csv(BASE/"data"/"interim"/"scored_long.csv")
        wide = scored_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first").reset_index()
        df_local = pl.merge(wide, on="pid", how="left").copy()
    except Exception:
        df_local = pl.copy()
    df_local = ensure_core_totals(df_local)
    return df_local

def available_outcomes(df_local: pd.DataFrame, base_targets: list):
    outs = []
    for t in base_targets:
        if t in df_local.columns:
            outs.append(t)
        if f"{t}_Q1" in df_local.columns:
            outs.append(f"{t}_Q1")
        if f"{t}_Q2" in df_local.columns:
            outs.append(f"{t}_Q2")
    return sorted(set(outs))

def run_ols_basic(df_local: pd.DataFrame, y: str, xnames: list):
    # OLS with HC3; returns rows for each predictor in xnames
    if y not in df_local.columns or df_local[y].std(skipna=True) == 0:
        return []
    X = pd.DataFrame({k: df_local[k].astype(float) for k in xnames if k in df_local.columns and df_local[k].std(skipna=True) > 0})
    if X.empty:
        return []
    X = sm.add_constant(X)
    Y = df_local[y].astype(float)
    m = sm.OLS(Y, X, missing="drop").fit(cov_type="HC3")
    sdY = Y.std(skipna=True)
    rows = []
    for k in X.columns:
        if k == "const":
            continue
        beta = float(m.params.get(k, np.nan))
        se   = float(m.bse.get(k, np.nan))
        ci_l = beta - 1.96*se
        ci_u = beta + 1.96*se
        sdX  = df_local[k].astype(float).std(skipna=True)
        beta_std = beta * (sdX / sdY) if (sdY>0 and sdX>0) else np.nan
        pval = float(m.pvalues.get(k, np.nan))
        rows.append({
            "outcome": y, "predictor": k, "beta": beta, "se": se, "ci_lo": ci_l, "ci_hi": ci_u,
            "beta_std": beta_std, "p_raw": pval, "n": int(m.nobs), "r2": float(m.rsquared)
        })
    return rows

def run_stacked_cluster(df_local: pd.DataFrame, base: str, xname: str):
    # Stack Q1/Q2 for a base outcome (e.g., 'insight'), add scenario fixed effect, cluster by pid
    cols = []
    for scen in ("Q1","Q2"):
        col = f"{base}_{scen}"
        if col in df_local.columns:
            tmp = df_local[["pid", xname, col]].copy()
            tmp = tmp.rename(columns={col: "y"})
            tmp["scenario"] = scen
            cols.append(tmp)
    if not cols:
        return None
    long = pd.concat(cols, axis=0, ignore_index=True)
    # Drop rows with missing outcome
    long = long[np.isfinite(long["y"]) & np.isfinite(long[xname])]
    if long.empty:
        return None
    # Design matrix: intercept, xname, scenario_Q2 (Q1 is baseline)
    scen = (long["scenario"] == "Q2").astype(float)
    X = pd.DataFrame({"const": 1.0, xname: long[xname].astype(float), "scenario_Q2": scen})
    y = long["y"].astype(float)
    if X[xname].std(ddof=1) == 0 or y.std(ddof=1) == 0:
        return None
    m = sm.OLS(y, X, missing="drop").fit(cov_type="cluster", cov_kwds={"groups": long["pid"]})
    # Extract xname effect only
    beta = float(m.params.get(xname, np.nan))
    se   = float(m.bse.get(xname, np.nan))
    ci_l = beta - 1.96*se
    ci_u = beta + 1.96*se
    # Standardised beta
    sdY = y.std(ddof=1)
    sdX = X[xname].std(ddof=1)
    beta_std = beta * (sdX / sdY) if (sdY>0 and sdX>0) else np.nan
    pval = float(m.pvalues.get(xname, np.nan))
    return {
        "base": base,
        "predictor": xname,
        "beta": beta,
        "se": se,
        "ci_lo": ci_l,
        "ci_hi": ci_u,
        "beta_std": beta_std,
        "p_raw": pval,
        "n": int(m.nobs),
        "r2": float(m.rsquared),
    }

mosleh_rows_block = []
mosleh_rows_stack = []
for DATASET in DATASETS:
    print(f"\n[ Mosleh-only ] Processing {DATASET.upper()} dataset")
    df_local = load_person_df(DATASET)
    if df_local is None:
        continue
    # Which outcomes exist in this dataset
    outs = available_outcomes(df_local, MOSLEH_TARGETS)
    outs = [o for o in outs if any(o.startswith(t) for t in MOSLEH_TARGETS)]
    if not outs:
        print(f"[Mosleh] No targeted LIWC columns found for {DATASET}")
        continue
    # Per-block (Q1/Q2) HC3 OLS, CRT2-only
    for y in outs:
        for x in MOSLEH_PREDICTORS:
            if x in df_local.columns and df_local[x].std(skipna=True) > 0:
                rows = run_ols_basic(df_local, y, [x])
                for r in rows:
                    r.update({"dataset": DATASET, "family": "MoslehTargets", "set": "crt2_only"})
                    mosleh_rows_block.append(r)
    # Stacked cluster-robust model per base target (if Q1/Q2 present)
    for base in MOSLEH_TARGETS:
        has_q = [f"{base}_Q1" in df_local.columns, f"{base}_Q2" in df_local.columns]
        if not any(has_q):
            continue
        for x in MOSLEH_PREDICTORS:
            if x in df_local.columns and df_local[x].std(skipna=True) > 0:
                res = run_stacked_cluster(df_local, base, x)
                if res is not None:
                    res.update({"dataset": DATASET, "family": "MoslehTargets", "set": "crt2_only_stacked"})
                    mosleh_rows_stack.append(res)

# Assemble DataFrames
mos_block_df = pd.DataFrame(mosleh_rows_block)
mos_stack_df = pd.DataFrame(mosleh_rows_stack)

# BH–FDR within the focused family, by dataset and model set
def apply_fdr(df_in: pd.DataFrame):
    if df_in.empty:
        return df_in
    df_in = df_in.copy()
    df_in["p_adj"] = np.nan
    df_in["sig_raw"] = df_in["p_raw"] < 0.05
    df_in["sig_adj"] = False
    for ds in df_in["dataset"].dropna().unique():
        for s in df_in["set"].dropna().unique():
            mask = (df_in["dataset"] == ds) & (df_in["set"] == s)
            if mask.any():
                adj, sig = fdr_bh(df_in.loc[mask, "p_raw"].values, alpha=0.05)
                df_in.loc[mask, "p_adj"] = adj
                df_in.loc[mask, "sig_adj"] = sig
    return df_in

mos_block_df = apply_fdr(mos_block_df)
mos_stack_df = apply_fdr(mos_stack_df)

# Persist focused outputs
if not mos_block_df.empty:
    mos_block_df.to_csv(TABLES_MOS / "mosleh_liwc_byblock_combined.csv", index=False)
    for ds in mos_block_df["dataset"].unique():
        mos_block_df[mos_block_df["dataset"] == ds].to_csv(TABLES_MOS / f"mosleh_liwc_byblock_{ds}.csv", index=False)
    print("Wrote Mosleh by-block CSVs ->", TABLES_MOS)
else:
    print("Mosleh by-block: no rows written (no targets present)")

# --- Mosleh-only: Counts -> (beta-)binomial mixed models (random intercept pid) ---
mos_glmm_rows = []
for DATASET in DATASETS:
    dff = load_person_df(DATASET)
    if dff is None or "pid" not in dff.columns:
        print(f"[Mosleh GLMM] no data/pid in {DATASET}")
        continue
    # Which base targets exist (Q1 and/or Q2)
    bases = []
    for t in MOSLEH_TARGETS:
        if f"{t}_Q1" in dff.columns or f"{t}_Q2" in dff.columns or t in dff.columns:
            bases.append(t)
    if not bases:
        print(f"[Mosleh GLMM] no MOSLEH_TARGETS present in {DATASET}")
        continue
    for set_name, Xnames in PRED_SETS.items():
        xcols = [x for x in Xnames if x in dff.columns and dff[x].std(skipna=True)>0]
        if not xcols:
            continue
        for base in bases:
            # Use multi-predictor version for consistency with family GLMM
            long = make_long_counts_multi(dff, base, xcols)
            if long is None or long.empty:
                continue
            # try GLMM via R/glmmTMB (multi-predictor)
            res, err = fit_binom_mixed_R_multi(long, xcols)
            if res is None:
                # fallback: binomial GLM with pid-clustered SEs
                res, err2 = fit_binom_cluster_multi(long, xcols)
                if res is None:
                    print(f"[Mosleh GLMM] {DATASET}/{base}/{set_name} failed: {err} | {err2}")
                    continue
            for xn, est in res.items():
                beta = est["beta"]; se = est["se"]
                ci_l = beta - 1.96*se; ci_u = beta + 1.96*se
                mos_glmm_rows.append({
                    "dataset": DATASET, "base": base, "set": set_name, "predictor": xn,
                    "beta": beta, "se": se, "ci_lo": ci_l, "ci_hi": ci_u,
                    "p_raw": np.nan, "model": est["family"], "n": int(len(long))
                })

mos_glmm_df = pd.DataFrame(mos_glmm_rows)
if not mos_glmm_df.empty:
    # Wald z = beta/se
    mos_glmm_df["z"] = mos_glmm_df["beta"] / mos_glmm_df["se"]
    mos_glmm_df["p_raw"] = two_sided_p_from_z(mos_glmm_df["z"])
    # FDR per (dataset, set, predictor): across MOSLEH base targets only
    mos_glmm_df["p_adj"], mos_glmm_df["sig_adj"] = np.nan, False
    for ds in mos_glmm_df["dataset"].dropna().unique():
        for s in mos_glmm_df["set"].dropna().unique():
            for pred in mos_glmm_df["predictor"].dropna().unique():
                mask = (mos_glmm_df["dataset"]==ds) & (mos_glmm_df["set"]==s) & (mos_glmm_df["predictor"]==pred)
                if mask.any():
                    adj, sig = fdr_bh(mos_glmm_df.loc[mask, "p_raw"].values, alpha=0.05)
                    mos_glmm_df.loc[mask, "p_adj"] = adj
                    mos_glmm_df.loc[mask, "sig_adj"] = sig
    # Write combined and per-dataset files
    mos_glmm_df.to_csv(TABLES_MOS/"mosleh_liwc_glmm_combined.csv", index=False)
    for ds in mos_glmm_df["dataset"].unique():
        mos_glmm_df[mos_glmm_df["dataset"]==ds].to_csv(TABLES_MOS/f"mosleh_liwc_glmm_{ds}.csv", index=False)
    print("Wrote Mosleh GLMM CSVs ->", TABLES_MOS)
else:
    print("Mosleh GLMM: no rows written (no targets present)")
# --- Figures for Mosleh-only GLMMs ---
try:
    if not mos_glmm_df.empty:
        for ds in mos_glmm_df["dataset"].dropna().unique():
            ds_dir = FIGS_MOS_GLMM / ds
            _ensure_dir(ds_dir)
            sub_all = mos_glmm_df[mos_glmm_df["dataset"]==ds].copy()
            if sub_all.empty:
                continue
            # Forest per predictor set (kept for clarity)
            for s in sub_all["set"].dropna().unique():
                sub = sub_all[sub_all["set"]==s].copy()
                if sub.empty:
                    continue
                title = f"Mosleh targets — {s} ({ds})"
                outp = ds_dir / f"mosleh_{s}_forest.png"
                forest_plot(sub.rename(columns={"base":"base"}), title, outp)
            # Combined heatmap across all sets/predictor-terms
            combined_heatmap(sub_all, dataset_label=ds, family_label="mosleh", out_dir=ds_dir, mosleh=True)
except Exception as e:
    print(f"Could not render Mosleh GLMM figures: {e}")
# --- end Mosleh-only section ---

# --- NFC correlations with CRT totals ---
print("\n=== Computing NFC-CRT correlations ===")
corr_rows = []
predictors = ["crt2_total", "bcrt_core3_total", "bcrt_core4_total"]

for DATASET in DATASETS:
    # Load the processed data for this dataset
    if DATASET == "naive":
        liwc_path = PROCESSED/"person_naive_with_liwc.csv"
    else:
        liwc_path = PROCESSED/"person_with_liwc.csv"
    
    if not liwc_path.exists():
        continue
        
    df_nfc = pd.read_csv(liwc_path)
    
    # Apply NFC correction
    nfc_items = [f"nfc_{i}" for i in range(1,19)]
    if any(col in df_nfc.columns for col in nfc_items):
        df_nfc["NFC_total"] = correct_nfc_scoring(df_nfc)
        df_nfc["nfc_total"] = df_nfc["NFC_total"]
    
    # Add core totals if needed
    df_nfc = ensure_core_totals(df_nfc)
    
    if "nfc_total" in df_nfc.columns:
        for pred in predictors:
            if pred in df_nfc.columns:
                x = df_nfc[pred].astype(float)
                y = df_nfc["nfc_total"].astype(float)
                # Handle missing values
                valid_mask = np.isfinite(x) & np.isfinite(y)
                if valid_mask.sum() > 10:  # Need at least 10 valid pairs
                    r = np.corrcoef(x[valid_mask], y[valid_mask])[0,1]
                    corr_rows.append({
                        "dataset": DATASET,
                        "x": pred, 
                        "y": "nfc_total", 
                        "r": r, 
                        "n": int(valid_mask.sum())
                    })

if corr_rows:
    nfc_corr_df = pd.DataFrame(corr_rows)
    nfc_corr_df.to_csv(TABLES / "nfc_correlations.csv", index=False)
    print("Wrote", TABLES / "nfc_correlations.csv")
else:
    print("No NFC correlations computed (missing data)")

# Update person files with corrected NFC scores
print("\n=== Updating person files with corrected NFC scores ===")
for file_name in ["person_full.csv", "person_with_liwc.csv", "person_naive_with_liwc.csv"]:
    file_path = PROCESSED / file_name
    if file_path.exists():
        try:
            pf = pd.read_csv(file_path)
            nfc_items = [f"nfc_{i}" for i in range(1,19)]
            if any(col in pf.columns for col in nfc_items):
                if "NFC_total" in pf.columns and "NFC_total_raw" not in pf.columns:
                    pf["NFC_total_raw"] = pf["NFC_total"]
                pf["NFC_total"] = correct_nfc_scoring(pf)
                pf.to_csv(file_path, index=False)
                print(f"Updated {file_name} with corrected NFC scores")
        except Exception as e:
            print(f"Could not update {file_name}: {e}")

# Refresh correlation table used by notebooks
try:
    pf = pd.read_csv(PROCESSED / "person_full.csv")
    cols = ["crt2_total","bcrt6_total","combined10_total","NFC_total"]
    available_cols = [c for c in cols if c in pf.columns]
    if len(available_cols) > 1:
        pf[available_cols].corr().to_csv(TABLES / "correlations_totals.csv")
        print("Updated correlations_totals.csv")
except Exception as e:
    print(f"Could not update correlations table: {e}")

print("\n=== LIWC/NFC convergent validity analysis complete ===")
print("Key outputs:")
print("- LIWC families analysis: liwc_predict_families_*.csv")
print("- NFC correlations: nfc_correlations.csv")
print("- All person files updated with corrected NFC scoring")