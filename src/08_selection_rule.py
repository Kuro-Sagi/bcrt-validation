# src/08_selection_rule.py
# Select BCRT items (core-3 and core-4) using naïve data only,
# prioritising psychometric quality (KR-20, discrimination, TIF coverage)
# and *secondary* convergent validity (LIWC summary + NFC).
#
# Outputs:
#   reports/tables/selection_decision.csv   # two rows: Core3, Core4
#   reports/tables/selection_diagnostics.csv# per-candidate subset metrics
#   reports/tables/selection_summary.md     # human-readable summary
#
# Notes:
# - Uses only BCRT items as candidates (CRT-2 is NOT considered for selection).
# - Requires: data/interim/scored_long.csv (with columns pid, item_id, family, correct)
#             data/processed/person_naive.csv
#   Optional: reports/tables/naive/irt2pl_bcrt.csv  (a_hat,b_hat per item)
#             reports/tables/liwc_predict_families_naive.csv (LIWC convergent validity)
#             reports/tables/nfc_correlations.csv (NFC convergent validity)
# - Convergent validity now uses pre-computed results from 07_liwc_families.py
# - Keeps method simple and transparent; avoids over-engineering.

import itertools, re, yaml, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm

D = 1.7

def iic(theta, a, b):
    p = 1.0 / (1.0 + np.exp(-D * a * (theta - b)))
    q = 1.0 - p
    return (D**2) * (a**2) * p * q

def kr20(items_df: pd.DataFrame) -> float:
    x = items_df.apply(pd.to_numeric, errors='coerce').fillna(0).clip(0,1)
    k = x.shape[1]
    if k < 2: return np.nan
    p = x.mean(axis=0); q = 1 - p
    var_total = x.sum(axis=1).var(ddof=1)
    if var_total <= 0: return np.nan
    return (k/(k-1)) * (1 - (p.mul(q).sum() / var_total))

def citc(items_df: pd.DataFrame) -> float:
    x = items_df.apply(pd.to_numeric, errors='coerce').fillna(0).clip(0,1)
    if x.shape[1] < 2: return np.nan
    vals = []
    for c in x.columns:
        rest = x.drop(columns=c).sum(axis=1)
        corr = pd.Series(x[c]).corr(rest)
        if pd.notna(corr): vals.append(corr)
    return float(np.nanmean(vals)) if vals else np.nan

def std_beta(y, x):
    yv = pd.to_numeric(y, errors='coerce')
    xv = pd.to_numeric(x, errors='coerce')
    df = pd.DataFrame({'y': yv, 'x': xv}).dropna()
    if len(df) < 25 or df['x'].std() == 0 or df['y'].std() == 0:
        return dict(beta=np.nan, p=np.nan, beta_std=np.nan, r2=np.nan, n=len(df))
    import statsmodels.api as sm
    X = sm.add_constant(df['x'].astype(float))
    m = sm.OLS(df['y'].astype(float), X).fit(cov_type='HC3')
    b = m.params.get('x', np.nan)
    sdY = df['y'].std(ddof=0); sdX = df['x'].std(ddof=0)
    bstd = b * (sdX / sdY) if sdY>0 and sdX>0 else np.nan
    return dict(beta=b, p=m.pvalues.get('x', np.nan), beta_std=bstd, r2=m.rsquared, n=int(m.nobs))

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    P = Path(cfg["paths"]["processed"])
    I = Path(cfg["paths"]["interim"])
    T = Path("reports") / "tables"
    T.mkdir(parents=True, exist_ok=True)
    TN = T / "naive"; TN.mkdir(parents=True, exist_ok=True)  # mirrored structure

    # Load naïve persons (selection must be naïve-only)
    person = pd.read_csv(P / "person_naive.csv")
    if "pid" not in person.columns:
        raise RuntimeError("person_naive.csv missing 'pid'")
    person['pid'] = person['pid'].astype(str)

    # Load item-level correctness (filter to BCRT)
    long = pd.read_csv(I / "scored_long.csv")
    long = long[long['family'].str.upper() == "BCRT"].copy()
    long['pid'] = long['pid'].astype(str)
    long = long[long['pid'].isin(set(person['pid']))]  # restrict to naïve

    # Pivot to item matrix
    wide = long.pivot_table(index='pid', columns='item_id', values='correct', aggfunc='first').reset_index()
    item_cols = [c for c in wide.columns if c != 'pid']
    if len(item_cols) < 3:
        raise RuntimeError("Fewer than 3 BCRT items found in scored_long for naïve sample.")

    # Optional 2PL parameters (naïve)
    irt_path = T / "naive" / "irt2pl_bcrt.csv"
    irt = None
    if irt_path.exists():
        irt = pd.read_csv(irt_path)[['item_id','a_hat','b_hat']].dropna(subset=['item_id'])

    # Load pre-computed convergent validity results
    liwc_families_path = T / "liwc_predict_families_naive.csv"
    nfc_corr_path = T / "nfc_correlations.csv"
    liwc_families = None
    nfc_corr = None
    
    if liwc_families_path.exists():
        liwc_families = pd.read_csv(liwc_families_path)
        liwc_families = liwc_families[liwc_families["dataset"] == "naive"]  # Ensure naive only
        
    if nfc_corr_path.exists():
        nfc_corr = pd.read_csv(nfc_corr_path)
        nfc_corr = nfc_corr[nfc_corr["dataset"] == "naive"]  # Ensure naive only

    # Merge
    df = person.merge(wide, on='pid', how='left')

    items_sorted = sorted(item_cols)
    from itertools import combinations
    combos3 = list(combinations(items_sorted, 3))
    combos4 = list(combinations(items_sorted, 4))

    theta = np.linspace(-2, 2, 161)

    def tif_area_for(items):
        if irt is None: return np.nan
        sub = irt[irt['item_id'].isin(items)].copy()
        if sub.empty or sub['a_hat'].isna().all() or sub['b_hat'].isna().all():
            return np.nan
        tif = np.zeros_like(theta, dtype=float)
        for _, r in sub.iterrows():
            a = float(r['a_hat']); b = float(r['b_hat'])
            if not np.isfinite(a) or not np.isfinite(b): continue
            tif += iic(theta, a, b)
        return float(np.trapz(tif, theta))

    def liwc_validity(items):
        """Get convergent validity metrics from pre-computed results"""
        out = dict(liwc_sig_cnt=0, liwc_mean_abs_bstd=np.nan, nfc_beta_std=np.nan, n_obs=np.nan, deltaR2_mean=np.nan)
        
        # Map items to core totals for lookup
        if len(items) == 3:
            predictor = "bcrt_core3_total"
        elif len(items) == 4:
            predictor = "bcrt_core4_total"
        else:
            return out  # Only handle core3/core4
        
        # Get LIWC convergent validity from pre-computed results
        if liwc_families is not None:
            # Filter to Summary family (main convergent validity indicators)
            summary_results = liwc_families[
                (liwc_families["family"] == "Summary") & 
                (liwc_families["predictor"] == predictor)
            ]
            
            if not summary_results.empty:
                # Count significant results and get mean absolute standardized beta
                sig_results = summary_results[summary_results["sig_adj"] == True]
                out['liwc_sig_cnt'] = len(sig_results)
                
                # Mean absolute standardized beta across all Summary variables
                valid_betas = summary_results["beta_std"].dropna()
                if not valid_betas.empty:
                    out['liwc_mean_abs_bstd'] = float(np.mean(np.abs(valid_betas)))
                
                # Use first valid n_obs as representative
                valid_n = summary_results["n"].dropna()
                if not valid_n.empty:
                    out['n_obs'] = int(valid_n.iloc[0])
                
                # Delta R² approximation from R² values (simplified)
                valid_r2 = summary_results["r2"].dropna()
                if not valid_r2.empty:
                    out['deltaR2_mean'] = float(np.mean(valid_r2))
        
        # Get NFC convergent validity from pre-computed correlations
        if nfc_corr is not None:
            nfc_result = nfc_corr[nfc_corr["x"] == predictor]
            if not nfc_result.empty:
                # Convert correlation to standardized beta (approximate)
                r = nfc_result["r"].iloc[0]
                if pd.notna(r):
                    out['nfc_beta_std'] = float(r)  # For standardized variables, r ≈ β_std
        
        return out

    rows = []
    for k, combos in [(3, combos3), (4, combos4)]:
        for items in combos:
            cols = list(items)
            items_df = df[cols].copy()
            metrics = {}
            metrics['kr20'] = kr20(items_df)
            metrics['citc_mean'] = citc(items_df)
            metrics['tif_area'] = tif_area_for(cols)

            lv = liwc_validity(cols)
            metrics.update(lv)

            rows.append({'k': k, 'items': ",".join(cols), **metrics})

    diag = pd.DataFrame(rows)

    def z(s):
        v = pd.to_numeric(s, errors='coerce'); 
        sd = v.std(ddof=0)
        if not np.isfinite(sd) or sd==0: return (v*0)
        return (v - v.mean()) / sd

    diag['z_kr20'] = z(diag['kr20'])
    diag['z_citc'] = z(diag['citc_mean'])
    diag['z_tif']  = z(diag['tif_area'])
    diag['z_liwc'] = z(diag['liwc_mean_abs_bstd'])
    diag['z_liwc_sig'] = z(diag['liwc_sig_cnt'])
    diag['nfc_pref'] = diag['nfc_beta_std'].apply(lambda b: np.nan if pd.isna(b) else (b if b>0 else -abs(b)))
    diag['z_nfc']  = z(diag['nfc_pref'])
    diag['z_deltaR2'] = z(diag['deltaR2_mean'])

    w = {'z_kr20':0.35, 'z_citc':0.10, 'z_tif':0.25, 'z_liwc':0.10, 'z_liwc_sig':0.05, 'z_nfc':0.10, 'z_deltaR2':0.05}
    diag['score'] = sum(diag[k]*w[k] for k in w if k in diag.columns)

    winners = []
    for k in (3,4):
        sub = diag[diag['k']==k].copy().sort_values('score', ascending=False)
        if not sub.empty:
            winners.append(sub.iloc[0])

    T.mkdir(parents=True, exist_ok=True)
    (T / "selection_diagnostics.csv").write_text(diag.to_csv(index=False))

    out_rows = []
    md_lines = ["# BCRT selection (naïve data only)\n"]
    for wrow in winners:
        label = f"Core-{int(wrow['k'])}"
        out_rows.append({
            'label': label,
            'k': int(wrow['k']),
            'keep_items': wrow['items'],
            'score': round(float(wrow['score']),4),
            'kr20': round(float(wrow['kr20']),4) if pd.notna(wrow['kr20']) else np.nan,
            'citc_mean': round(float(wrow['citc_mean']),4) if pd.notna(wrow['citc_mean']) else np.nan,
            'tif_area': round(float(wrow['tif_area']),4) if pd.notna(wrow['tif_area']) else np.nan,
            'liwc_sig_cnt': int(wrow['liwc_sig_cnt']) if pd.notna(wrow['liwc_sig_cnt']) else 0,
            'liwc_mean_abs_beta': round(float(wrow['liwc_mean_abs_bstd']),4) if pd.notna(wrow['liwc_mean_abs_bstd']) else np.nan,
            'nfc_beta_std': round(float(wrow['nfc_beta_std']),4) if pd.notna(wrow['nfc_beta_std']) else np.nan,
            'deltaR2_mean': round(float(wrow['deltaR2_mean']),4) if pd.notna(wrow['deltaR2_mean']) else np.nan,
            'n_obs_liwc': int(wrow['n_obs']) if pd.notna(wrow['n_obs']) else np.nan
        })
        md_lines.append(f"## {label}\n")
        md_lines.append(f"- Items: **{wrow['items']}**\n"
                        f"- Score: **{wrow['score']:.3f}**  \n"
                        f"- KR-20: {wrow['kr20']:.3f} | CITC(mean): {wrow['citc_mean']:.3f}  \n"
                        f"- TIF area [-2,2]: {wrow['tif_area'] if pd.notna(wrow['tif_area']) else 'NA'}  \n"
                        f"- LIWC summary: sig={int(wrow['liwc_sig_cnt'])}, mean |β_std|={wrow['liwc_mean_abs_bstd']:.3f}  \n"
                        f"- NFC β_std: {wrow['nfc_beta_std'] if pd.notna(wrow['nfc_beta_std']) else 'NA'}  \n"
                        f"- Mean ΔR² over CRT-2 (LIWC summary): {wrow['deltaR2_mean'] if pd.notna(wrow['deltaR2_mean']) else 'NA'}  \n")

    sel = pd.DataFrame(out_rows)
    (T / "selection_decision.csv").write_text(sel.to_csv(index=False))
    (T / "selection_summary.md").write_text("\n".join(md_lines))

    print("Wrote:", T / "selection_decision.csv")
    print("Also:", T / "selection_diagnostics.csv", "and", T / "selection_summary.md")

if __name__ == "__main__":
    main()
