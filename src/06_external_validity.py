
import yaml, pandas as pd, numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr

cfg = yaml.safe_load(open("config/config.yaml"))
processed_dir = Path(cfg["paths"]["processed"])
reports_tables = Path(cfg["paths"]["reports"]) / "tables"
reports_tables.mkdir(parents=True, exist_ok=True)

def robust_ols(df, y, X_cols, add_cov=None):
    X = df[X_cols].copy()
    if add_cov:
        X = pd.concat([X, df[add_cov]], axis=1)
    X = sm.add_constant(X, has_constant="add")
    yv = df[y]
    model = sm.OLS(yv, X, missing="drop").fit(cov_type="HC3")
    return model

# Load with LIWC (if present)
path_naive = processed_dir / "person_naive_liwc.csv"
path_full  = processed_dir / "person_full_liwc.csv"
if not path_naive.exists():
    # fallback to without LIWC (will skip analyses)
    print("Warning: LIWC merge not found; run 05_liwc_io.py after exporting LIWC outputs.")
    exit()

naive = pd.read_csv(path_naive)
full = pd.read_csv(path_full)

# NFC correlations
corrs = []
for dname, d in [("naive", naive), ("full", full)]:
    for s in ["bcrt6_total","combined10_total","crt2_total"]:
        r, p = spearmanr(d[s], d["NFC_total"], nan_policy="omit")
        corrs.append({"dataset": dname, "x": "NFC_total", "y": s, "rho": r, "p": p})
corrs = pd.DataFrame(corrs)
corrs.to_csv(reports_tables / "nfc_correlations.csv", index=False)

# LIWC regressions: DV = LIWC vars, IVs = CRT totals, covariates = WC_avg (implicitly via RLI_resid) and seen_count for full
liwc_vars = [c for c in naive.columns if c.endswith("_avg") and c.startswith(("analytic","cogproc","insight","cause","tentat","certain","number","quant","sixltr","tone"))]
liwc_vars = sorted(set(liwc_vars + ["RLI","RLI_resid"]))
results = []

for dname, d in [("naive", naive), ("full", full)]:
    covs = []
    if dname == "full" and "seen_count" in d.columns:
        covs = ["seen_count"]
    for dv in liwc_vars:
        for iv in ["crt2_total","bcrt6_total","combined10_total"]:
            try:
                m = robust_ols(d, y=dv, X_cols=[iv], add_cov=covs)
                coef = m.params.get(iv, np.nan); se = m.bse.get(iv, np.nan); p = m.pvalues.get(iv, np.nan)
                results.append({"dataset": dname, "dv": dv, "iv": iv, "beta": coef, "se": se, "p": p, "n": int(m.nobs)})
            except Exception as e:
                results.append({"dataset": dname, "dv": dv, "iv": iv, "beta": np.nan, "se": np.nan, "p": np.nan, "n": int(d[[dv,iv]].dropna().shape[0])})

res = pd.DataFrame(results)
# p-adjust within each dataset × dv (Holm)
res["p_adj"] = np.nan
for key, sub in res.groupby(["dataset","dv"]):
    res.loc[sub.index, "p_adj"] = multipletests(sub["p"].values, method="holm")[1]
res.to_csv(reports_tables / "liwc_regressions.csv", index=False)
print("Saved liwc_regressions.csv")
