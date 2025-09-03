# src/08_irt_2pl.py
# Pragmatic 2PL bolt-on using logistic regressions on a z-scored rested ability proxy.
# Outputs:
# - reports/tables/irt2pl_<sample>.csv  (a_hat, b_hat, slopes/intercepts, HC3 SEs)
# - reports/figures/irt2pl/icc_<item>_<sample>.png
# - reports/figures/irt2pl/iic_<item>_<sample>.png
# - reports/figures/irt2pl/tif_<sample>.png

import argparse, yaml, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt

D = 1.7  # logistic scaling to approximate normal-ogive (common convention)

def load_data(cfg):
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    crt2 = cfg["columns"]["crt2_items"]
    bcrt = cfg["columns"]["bcrt_items"]
    all_items = crt2 + bcrt

    item_long = pd.read_csv(interim / "scored_long.csv")
    person_naive = pd.read_csv(processed / "person_naive.csv")
    person_full  = pd.read_csv(processed / "person_full.csv")

    # pivot to wide correct
    pivot = (item_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first")
             .reindex(columns=all_items).fillna(0).astype(int))
    # totals
    tot_crt2 = pivot[crt2].sum(axis=1)
    tot_bcrt = pivot[bcrt].sum(axis=1)
    tot_comb = tot_crt2 + tot_bcrt

    return crt2, bcrt, all_items, item_long, person_naive, person_full, pivot, tot_comb

def ability_proxy_rest_comb_z(pids, pivot, tot_comb, item):
    # rested combined total for the chosen pids, then z-score
    rest = (tot_comb - pivot[item]).loc[pids].astype(float)
    s = rest.std(ddof=1)
    if s == 0 or np.isnan(s):  # guard
        z = (rest - rest.mean())
    else:
        z = (rest - rest.mean()) / s
    return z

def fit_logit(y, x):
    X = sm.add_constant(x)
    model = sm.Logit(y, X, missing='drop')
    res = model.fit(disp=False, method="lbfgs", maxiter=200)
    # robust SE (HC3)
    robust = res.get_robustcov_results(cov_type="HC3")
    return res, robust

def icc_p(theta, a, b):
    # 2PL logistic with D scaling
    return 1.0 / (1.0 + np.exp(-D * a * (theta - b)))

def iic(theta, a, b):
    p = icc_p(theta, a, b)
    return (a**2) * p * (1 - p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["naive","unseen"], default="naive",
                    help="naive = fully naïve participants; unseen = all respondents who have NOT seen the item")
    ap.add_argument("--grid_min", type=float, default=-3.0)
    ap.add_argument("--grid_max", type=float, default=3.0)
    ap.add_argument("--grid_step", type=float, default=0.05)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml"))
    reports = Path(cfg["paths"]["reports"])
    figs_dir = reports / "figures" / "irt2pl"
    tables_dir = reports / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    crt2, bcrt, all_items, item_long, person_naive, person_full, pivot, tot_comb = load_data(cfg)

    if args.sample == "naive":
        pids = person_naive["pid"].tolist()
        sample_tag = "naive"
    else:
        # unseen-only: for each item, we will filter pids who have seen==0 for that item
        pids = person_full["pid"].tolist()
        sample_tag = "unseen"

    theta_grid = np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step)

    rows = []
    tif = np.zeros_like(theta_grid)

    for item in all_items:
        # choose sample
        if args.sample == "naive":
            use_pids = pids
        else:
            unseen_mask = (item_long["item_id"] == item) & (item_long["seen"] == 0)
            use_pids = item_long.loc[unseen_mask, "pid"].unique().tolist()

        if len(use_pids) < 15:
            # too few to fit anything sensible
            continue

        # ability proxy (rested combined z)
        theta = ability_proxy_rest_comb_z(use_pids, pivot, tot_comb, item)
        y = pivot.loc[use_pids, item].reindex(theta.index).astype(int)

        # fit logistic: y ~ const + theta
        try:
            res, robust = fit_logit(y, theta)
        except Exception:
            continue

        beta0, beta1 = robust.params  # intercept, slope
        se0, se1 = np.sqrt(np.diag(robust.cov_params()))
        # Map to 2PL-like params on this theta scale:
        # P = 1 / (1 + exp( - (beta0 + beta1 * theta) ))
        # 2PL form uses D*a*(theta - b):  beta1 ≈ D*a  and  -beta0/beta1 ≈ b
        a_hat = beta1 / D
        b_hat = -beta0 / beta1 if beta1 != 0 else np.nan

        # Curves
        p_curve = icc_p(theta_grid, a_hat, b_hat)
        i_curve = iic(theta_grid, a_hat, b_hat)
        tif += i_curve

        # Save per-item plots
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        ax.plot(theta_grid, p_curve, label=f"ICC {item}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Ability (rested Combined10, z)")
        ax.set_ylabel("P(correct)")
        ax.set_title(f"2PL-approx ICC: {item} ({sample_tag})")
        plt.tight_layout()
        plt.savefig(figs_dir / f"icc_{item}_{sample_tag}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        ax.plot(theta_grid, i_curve, label=f"IIC {item}")
        ax.set_xlabel("Ability (rested Combined10, z)")
        ax.set_ylabel("Information")
        ax.set_title(f"2PL-approx IIC: {item} ({sample_tag})")
        plt.tight_layout()
        plt.savefig(figs_dir / f"iic_{item}_{sample_tag}.png", dpi=200)
        plt.close(fig)

        rows.append({
            "item": item,
            "beta0": beta0, "beta1": beta1, "se_beta0_HC3": se0, "se_beta1_HC3": se1,
            "a_hat": a_hat, "b_hat": b_hat, "n_used": int(len(use_pids))
        })

    # Save parameter table
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / f"irt2pl_{sample_tag}.csv", index=False)

    # TIF
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(theta_grid, tif)
    ax.set_xlabel("Ability (rested Combined10, z)")
    ax.set_ylabel("Test information (sum of IICs)")
    ax.set_title(f"TIF ({sample_tag})")
    plt.tight_layout()
    plt.savefig(figs_dir / f"tif_{sample_tag}.png", dpi=200)
    plt.close(fig)

    print(f"Wrote {len(out)} item rows to reports/tables/irt2pl_{sample_tag}.csv and figures to {figs_dir}")

if __name__ == "__main__":
    main()