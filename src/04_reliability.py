# src/04_reliability.py  (dynamic: supports pre-specified + selected sets; writes NAIVE + FULL)
import re, yaml, pandas as pd, numpy as np
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))
P = Path(cfg["paths"]["processed"])
T = Path("reports")/"tables"; T.mkdir(parents=True, exist_ok=True)
SEL = T/"selection_decision.csv"

def correct_nfc_scoring(df: pd.DataFrame) -> pd.Series:
    """Apply correct NFC scoring (9 forward, 9 reverse) on raw `nfc_*` columns.
    Returns a Series (NFC_total). Falls back to existing NFC_total if items absent.
    """
    forward = [1,2,6,10,11,13,14,15,18]
    reverse = [3,4,5,7,8,9,12,16,17]
    items = [f"nfc_{i}" for i in range(1,19)]
    present = [c for c in items if c in df.columns]
    if not present:
        return df.get("NFC_total", pd.Series(np.nan, index=df.index))
    label_to_num = {
        '1 = Extremely Uncharacteristic': 1,
        '2 = Somewhat Uncharacteristic': 2,
        '3 = Uncertain': 3,
        '4 = Somewhat Characteristic': 4,
        '5 = Extremely Characteristic': 5,
    }
    num = {}
    for c in items:
        if c in df.columns:
            s = df[c]
            # accept either labeled strings or already-numeric
            if s.dtype.kind in "if":
                num[c] = pd.to_numeric(s, errors="coerce")
            else:
                num[c] = s.map(label_to_num)
    num_df = pd.DataFrame(num)
    # reverse-key where needed
    for i in reverse:
        col = f"nfc_{i}"
        if col in num_df.columns:
            num_df[col] = 6 - pd.to_numeric(num_df[col], errors="coerce")
    return num_df.sum(axis=1, min_count=1)

def load_core_item_sets() -> dict:
    """Read `selection_decision.csv` and return dict with keys 'core3' and 'core4'.
    If unavailable, return empty dict to signal fallback.
    """
    if not SEL.exists():
        return {}
    try:
        sel = pd.read_csv(SEL)
    except Exception:
        return {}
    req = set(["label","keep_items"]) - set(sel.columns)
    if req:
        return {}
    out = {"core3": [], "core4": []}
    for _, r in sel.iterrows():
        lab = str(r.get("label",""))
        items_raw = str(r.get("keep_items",""))
        items = [x.strip() for x in items_raw.split(',') if x.strip()]
        if "3" in lab:
            out["core3"] = items
        elif "4" in lab:
            out["core4"] = items
    return out

def ensure_totals(df: pd.DataFrame) -> pd.DataFrame:
    crt2_items = [c for c in df.columns if re.match(r"^crt2_q\d+", c)]
    bcrt_items = [c for c in df.columns if re.match(r"^bcrt_q\d+", c)]
    if "crt2_total" not in df.columns and crt2_items:
        df["crt2_total"] = df[crt2_items].sum(axis=1, min_count=1)
    if "bcrt6_total" not in df.columns and bcrt_items:
        df["bcrt6_total"] = df[bcrt_items].sum(axis=1, min_count=1)
    # Prefer selection_decision.csv if available; else canonical fallback
    sets = load_core_item_sets()
    if sets.get("core3"):
        core3 = [c for c in sets["core3"] if c in df.columns]
    else:
        core3 = [c for c in ["bcrt_q3","bcrt_q6","bcrt_q2"] if c in df.columns]
    if sets.get("core4"):
        core4 = [c for c in sets["core4"] if c in df.columns]
    else:
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

def parse_selection() -> list:
    sets = load_core_item_sets()
    # union of selected items for optional "Selected" summaries
    items = list(dict.fromkeys((sets.get("core3", []) + sets.get("core4", []))))
    return items

def as_dichotomous(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = (pd.to_numeric(out[c], errors="coerce").fillna(0) > 0).astype(int)
    return out

def kr20(items_df: pd.DataFrame) -> float:
    items = as_dichotomous(items_df)
    k = items.shape[1]
    if k < 2: return np.nan
    p = items.mean(axis=0); q = 1 - p
    var_total = items.sum(axis=1).var(ddof=1)
    if var_total <= 0: return np.nan
    return (k/(k-1)) * (1 - (p.mul(q).sum() / var_total))

def reliability_rows(df: pd.DataFrame, selected_items: list) -> pd.DataFrame:
    rows = []
    crt2_items = [c for c in df.columns if re.match(r"^crt2_q\d+", c)]
    bcrt_items = [c for c in df.columns if re.match(r"^bcrt_q\d+", c)]
    core4 = [c for c in ["bcrt_q3","bcrt_q6","bcrt_q2","bcrt_q4"] if c in df.columns]
    sets = {
        "CRT-2": crt2_items,
        "BCRT-6": bcrt_items,
        "BCRT-4": core4,
        "Combined-8": crt2_items + core4,
        "Combined-10": crt2_items + bcrt_items,
    }
    # Add selected set if provided
    sel_items = [c for c in selected_items if c in df.columns]
    if len(sel_items) >= 2:
        sets[f"Selected ({len(sel_items)})"] = sel_items
        sets[f"Combined (CRT2+Selected)"] = crt2_items + sel_items

    for label, cols in sets.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            rows.append(dict(label=label, k=len(cols), n=np.nan, KR20=np.nan))
            continue
        items = df[cols].dropna(how="all")
        rows.append(dict(label=label, k=len(cols), n=len(items), KR20=kr20(items)))
    return pd.DataFrame(rows)

def add_selected_totals(df: pd.DataFrame, selected_items: list) -> pd.DataFrame:
    sel = [c for c in selected_items if c in df.columns]
    if len(sel) >= 1:
        df["bcrt_selected_total"] = df[sel].sum(axis=1, min_count=1)
        if "crt2_total" in df.columns:
            df["combined_selected_total"] = df["crt2_total"] + df["bcrt_selected_total"]
    return df

def _pearson_sig(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = int(len(x))
    if n < 4:
        return dict(n=n, r=np.nan, p=np.nan, lo=np.nan, hi=np.nan)
    r = float(np.corrcoef(x, y)[0,1])
    r = np.clip(r, -0.999999, 0.999999)
    # p-value via t distribution
    try:
        from scipy.stats import t as tdist
        tval = r * np.sqrt((n-2) / max(1e-12, 1 - r*r))
        p = 2 * (1 - tdist.cdf(abs(tval), df=n-2))
    except Exception:
        # fallback using normal approx
        z = r * np.sqrt(n-3)
        from math import erf, sqrt
        p = 2 * (1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    # Fisher z CI
    from math import atanh, tanh
    z = atanh(r)
    se = 1/np.sqrt(max(1, n-3))
    z_lo, z_hi = z - 1.96*se, z + 1.96*se
    lo, hi = tanh(z_lo), tanh(z_hi)
    return dict(n=n, r=r, p=float(p), lo=float(lo), hi=float(hi))

def write_corr(df: pd.DataFrame, outpath: Path) -> None:
    cols = [c for c in [
        "crt2_total","bcrt6_total","bcrt_core3_total","bcrt_core4_total",
        "combined7_total","combined8_total","combined10_total","NFC_total",
        "bcrt_selected_total","combined_selected_total"
    ] if c in df.columns]
    if not cols:
        pd.DataFrame().to_csv(outpath)
        # also produce an empty sig file
        Path(str(outpath).replace('.csv','_sig.csv')).write_text("")
        return
    corr = df[cols].corr()
    corr.to_csv(outpath)
    # significance table (long format)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            res = _pearson_sig(df[a], df[b])
            rows.append({"var_x": a, "var_y": b, **res})
    sig = pd.DataFrame(rows)
    sig_out = Path(str(outpath).replace('.csv','_sig.csv'))
    sig.to_csv(sig_out, index=False)

def main():
    # Load person data
    full  = pd.read_csv(P/"person_wide.csv")
    naive = pd.read_csv(P/"person_naive.csv")
    
    # Load item-level data and create wide format like 03_item_stats.py does
    I = Path("data/interim")
    item_long = pd.read_csv(I / "scored_long.csv")
    
    # Create wide format pivot table with individual items
    pivot = (item_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first")
             .fillna(0).astype(int))
    
    # Merge individual items with person data
    full = full.merge(pivot, left_on="pid", right_index=True, how="left")
    naive = naive.merge(pivot, left_on="pid", right_index=True, how="left")
    
    selected = parse_selection()

    # Ensure totals and correct NFC scoring for both
    full  = ensure_totals(full.copy());
    naive = ensure_totals(naive.copy());
    full["NFC_total"]  = correct_nfc_scoring(full)
    naive["NFC_total"] = correct_nfc_scoring(naive)
    full  = add_selected_totals(full, selected)
    naive = add_selected_totals(naive, selected)

    T.mkdir(parents=True, exist_ok=True)
    reliability_rows(full, selected).to_csv(T/"reliability_full.csv", index=False)
    reliability_rows(naive, selected).to_csv(T/"reliability_naive.csv", index=False)

    write_corr(full,  T/"correlations_totals_full.csv")
    write_corr(naive, T/"correlations_totals_naive.csv")

    msg = "with" if selected else "without"
    print(f"Wrote reliability/correlations for full+naive {msg} selected-set support.")

if __name__ == "__main__":
    main()
