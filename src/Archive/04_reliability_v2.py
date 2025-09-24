# src/04_reliability.py  (updated: writes NAIVE + FULL; includes core3/core4/combined sets)
import re, yaml, pandas as pd, numpy as np
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))
P = Path(cfg["paths"]["processed"])
T = Path("reports")/"tables"; T.mkdir(parents=True, exist_ok=True)

def ensure_totals(df):
    crt2_items = [c for c in df.columns if re.match(r"^crt2_q\d+", c)]
    bcrt_items = [c for c in df.columns if re.match(r"^bcrt_q\d+", c)]
    if "crt2_total" not in df.columns and crt2_items:
        df["crt2_total"] = df[crt2_items].sum(axis=1, min_count=1)
    if "bcrt6_total" not in df.columns and bcrt_items:
        df["bcrt6_total"] = df[bcrt_items].sum(axis=1, min_count=1)
    core3 = [c for c in ["bcrt_q3","bcrt_q6","bcrt_q2"] if c in df.columns]
    core4 = core3 + [c for c in ["bcrt_q4"] if c in df.columns]
    if core3: df["bcrt_core3_total"] = df[core3].sum(axis=1, min_count=1)
    if core4: df["bcrt_core4_total"] = df[core4].sum(axis=1, min_count=1)
    if "crt2_total" in df.columns and "bcrt_core3_total" in df.columns:
        df["combined7_total"] = df["crt2_total"] + df["bcrt_core3_total"]
    if "crt2_total" in df.columns and "bcrt_core4_total" in df.columns:
        df["combined8_total"] = df["crt2_total"] + df["bcrt_core4_total"]
    if "crt2_total" in df.columns and "bcrt6_total" in df.columns:
        df["combined10_total"] = df["crt2_total"] + df["bcrt6_total"]
    return df

def kr20(items_df):
    items = items_df.copy()
    for c in items.columns:
        items[c] = (pd.to_numeric(items[c], errors="coerce").fillna(0) > 0).astype(int)
    k = items.shape[1]
    if k < 2: return np.nan
    p = items.mean(axis=0); q = 1 - p
    var_total = items.sum(axis=1).var(ddof=1)
    if var_total <= 0: return np.nan
    return (k/(k-1)) * (1 - (p.mul(q).sum() / var_total))

def reliability_rows(df):
    rows = []
    crt2_items = [c for c in df.columns if re.match(r"^crt2_q\d+", c)]
    bcrt_items = [c for c in df.columns if re.match(r"^bcrt_q\d+", c)]
    sets = {
        "CRT-2": crt2_items,
        "BCRT-6": bcrt_items,
        "BCRT-4": [c for c in ["bcrt_q3","bcrt_q6","bcrt_q2","bcrt_q4"] if c in df.columns],
        "Combined-8": crt2_items + [c for c in ["bcrt_q3","bcrt_q6","bcrt_q2","bcrt_q4"] if c in df.columns],
        "Combined-10": crt2_items + bcrt_items,
    }
    for label, cols in sets.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            rows.append(dict(label=label, k=len(cols), n=np.nan, KR20=np.nan))
            continue
        items = df[cols].dropna(how="all")
        n = len(items)
        rows.append(dict(label=label, k=len(cols), n=n, KR20=kr20(items)))
    return pd.DataFrame(rows)

def write_corr(df, outpath):
    cols = [c for c in [
        "crt2_total","bcrt6_total","bcrt_core3_total","bcrt_core4_total",
        "combined7_total","combined8_total","combined10_total","NFC_total"
    ] if c in df.columns]
    corr = df[cols].corr()
    corr.to_csv(outpath)

full  = pd.read_csv(P/"person_wide.csv")
naive = pd.read_csv(P/"person_naive.csv")
full  = ensure_totals(full.copy())
naive = ensure_totals(naive.copy())

reliab_full  = reliability_rows(full)
reliab_naive = reliability_rows(naive)
reliab_full.to_csv(T/"reliability_full.csv", index=False)
reliab_naive.to_csv(T/"reliability_naive.csv", index=False)

write_corr(full,  T/"correlations_totals_full.csv")
write_corr(naive, T/"correlations_totals_naive.csv")

print("Wrote reliability_{full,naive}.csv and correlations_totals_{full,naive}.csv")
