import yaml, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
D = 1.7

def iic(theta,a,b):
    p = 1/(1+np.exp(-D*a*(theta-b))); q = 1-p
    return (D**2)*(a**2)*p*q

def fit_family(fam, item_long, person_wide, pid_keep, suffix):
    dat = item_long[item_long['family'].str.upper()==fam].copy()
    dat = dat[dat['pid'].isin(pid_keep)]
    # ability proxy
    if "combined10_total" not in person_wide.columns:
        person_wide["combined10_total"] = person_wide["crt2_total"] + person_wide["bcrt6_total"]
    z = (person_wide["combined10_total"] - person_wide["combined10_total"].mean())/person_wide["combined10_total"].std(ddof=0)
    theta = person_wide[["pid"]].copy(); theta["theta_z"] = z
    dat = dat.merge(theta, on="pid", how="left")

    th = np.linspace(-3,3,121); tif = np.zeros_like(th); rows=[]
    for item, g in dat.groupby("item_id", sort=True):
        y = g["correct"].astype(float).values
        x = g["theta_z"].astype(float).values
        if y.mean()<=.02 or y.mean()>=.98 or np.all(y==y[0]): 
            rows.append(dict(item_id=item, a_hat=np.nan, b_hat=np.nan)); continue
        res = sm.Logit(y, sm.add_constant(x), missing='drop').fit(disp=False)
        b0, b1 = res.params; a = b1/D; b = -b0/b1
        tif += iic(th,a,b); rows.append(dict(item_id=item, a_hat=a, b_hat=b))
    Path("reports/figures/irt2pl").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path("reports/tables")/f"irt2pl_{suffix}.csv", index=False)
    plt.figure(figsize=(7,4.4)); plt.plot(th,tif)
    plt.xlabel("Ability (z)"); plt.ylabel("Test information"); plt.title(f"TIF — {suffix}")
    plt.tight_layout(); plt.savefig(Path("reports/figures/irt2pl")/f"tif_{suffix}.png", dpi=200); plt.close()

if __name__=="__main__":
    cfg = yaml.safe_load(open("config/config.yaml"))
    P = Path(cfg["paths"]["processed"]); I = Path(cfg["paths"]["interim"])
    item_long = pd.read_csv(I/"scored_long.csv")
    person_wide = pd.read_csv(P/"person_wide.csv")
    naive = pd.read_csv(P/"person_naive.csv")["pid"].tolist()
    fit_family("CRT2", item_long, person_wide, naive, "crt2_naive")
    fit_family("BCRT", item_long, person_wide, naive, "bcrt_naive")