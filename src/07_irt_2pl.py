# src/08_irt_2pl.py  (consolidated; runs full & naive with dataset-tagged outputs)
import argparse, yaml, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt

D = 1.7

def icc(theta, a, b):
    z = D * a * (theta - b)
    return 1.0 / (1.0 + np.exp(-z))

def iic(theta, a, b):
    p = icc(theta, a, b); q = 1.0 - p
    return (D**2) * (a**2) * p * q

def fit_logit(x, y):
    X = sm.add_constant(x)
    model = sm.Logit(y, X, missing="drop")
    res = model.fit(disp=False)
    robust = res.get_robustcov_results(cov_type="HC3")
    return res, robust

def run_family(family, item_long, person_df, figs_dir: Path, tables_dir: Path, out_prefix: str):
    dat = item_long[item_long["family"].str.upper() == family].copy()
    if dat.empty:
        return {"ok": False, "params": pd.DataFrame(), "theta_grid": np.linspace(-3,3,121)}

    # Ability proxy: Combined10 z-score (same for both)
    if "combined10_total" not in person_df.columns:
        person_df["combined10_total"] = person_df["crt2_total"] + person_df.get("bcrt6_total", 0)
    comb = person_df["combined10_total"]
    person_df["theta_z"] = (comb - comb.mean()) / comb.std(ddof=0)

    dat = dat.merge(person_df[["pid","theta_z","combined10_total"]], on="pid", how="inner")
    dat["correct"] = pd.to_numeric(dat["correct"], errors="coerce").fillna(0.0)

    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    theta_grid = np.linspace(-3,3,121)
    tif = np.zeros_like(theta_grid)
    rows = []

    for item, g in dat.groupby("item_id", sort=True):
        df = g.copy()
        y = df["correct"].astype(float).values
        x = df["theta_z"].astype(float).values

        pbar = y.mean()
        if pbar <= 0.02 or pbar >= 0.98 or np.all(y == y[0]):
            rows.append(dict(item_id=item, n=len(y), p_correct=pbar,
                             beta0=np.nan, beta1=np.nan, se_beta0=np.nan, se_beta1=np.nan,
                             a_hat=np.nan, b_hat=np.nan))
            continue

        try:
            res, robust = fit_logit(x, y)
            beta0 = float(res.params[0]); beta1 = float(res.params[1])
            se_beta0 = float(robust.bse[0]); se_beta1 = float(robust.bse[1])
        except Exception:
            try:
                model = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial())
                res = model.fit()
                beta0 = float(res.params[0]); beta1 = float(res.params[1])
                se_beta0 = float(res.bse[0]); se_beta1 = float(res.bse[1])
            except Exception:
                beta0 = np.nan; beta1 = np.nan; se_beta0 = np.nan; se_beta1 = np.nan

        if not np.isfinite(beta1) or beta1 == 0:
            a_hat = np.nan; b_hat = np.nan
            p_curve = np.zeros_like(theta_grid)
            i_curve = np.zeros_like(theta_grid)
        else:
            a_hat = beta1 / D
            b_hat = -beta0 / beta1
            p_curve = 1.0 / (1.0 + np.exp(-(beta0 + beta1 * theta_grid)))
            i_curve = iic(theta_grid, a_hat, b_hat)

        tif += i_curve

        # Save per-item curves (dataset-specific directory avoids overwrites)
        fig, ax = plt.subplots(figsize=(6.0,4.3))
        ax.plot(theta_grid, p_curve, color='#2E86AB', linewidth=2.0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Ability (Combined10, z)")
        ax.set_ylabel("P(correct)")
        ax.set_title(f"ICC: {item} ({family})")
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.tight_layout()
        # Remove redundant family prefix in filename; item already encodes family
        plt.savefig(figs_dir / f"icc_{item}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0,4.3))
        ax.plot(theta_grid, i_curve, color='#A23B72', linewidth=2.0)
        ax.set_xlabel("Ability (Combined10, z)")
        ax.set_ylabel("Item information")
        ax.set_title(f"IIC: {item} ({family})")
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.tight_layout()
        # Remove redundant family prefix in filename; item already encodes family
        plt.savefig(figs_dir / f"iic_{item}.png", dpi=200)
        plt.close(fig)

        rows.append(dict(item_id=item, n=len(y), p_correct=pbar, beta0=beta0, beta1=beta1,
                         se_beta0=se_beta0, se_beta1=se_beta1, a_hat=a_hat, b_hat=b_hat))

    out = pd.DataFrame(rows).sort_values("item_id")
    out.to_csv(tables_dir / f"irt2pl_{out_prefix}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8,4.8))
    ax.plot(theta_grid, tif, color='#F18F01', linewidth=2.5)
    ax.set_xlabel("Ability (Combined10, z)")
    ax.set_ylabel("Test information")
    ax.set_title(f"TIF ({family})")
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figs_dir / f"tif_{out_prefix}.png", dpi=200)
    plt.close(fig)

    return {"ok": True, "params": out, "theta_grid": theta_grid}

def run_for_dataset(dataset_tag: str, item_long: pd.DataFrame, person_wide: pd.DataFrame, person_naive: pd.DataFrame):
    # build dataset-specific person df and output dirs
    if dataset_tag == "naive":
        keep = set(person_naive["pid"].astype(str))
        person_df = person_wide[person_wide["pid"].astype(str).isin(keep)].copy()
    else:
        person_df = person_wide.copy()

    base_fig = Path("reports")/"figures"/"irt2pl"/dataset_tag
    base_tab = Path("reports")/"tables"/dataset_tag

    crt2_res = run_family("CRT2", item_long, person_df, base_fig, base_tab, out_prefix="crt2")
    bcrt_res = run_family("BCRT", item_long, person_df, base_fig, base_tab, out_prefix="bcrt")

    # After per-family outputs, try to compute subset TIFs based on selection_decision.csv
    try:
        sel_path = Path("reports")/"tables"/"selection_decision.csv"
        core3_items = []
        core4_items = []
        if sel_path.exists():
            sel_df = pd.read_csv(sel_path)
            if "label" in sel_df.columns and "keep_items" in sel_df.columns:
                for _, r in sel_df.iterrows():
                    label = str(r["label"]).strip().lower()
                    items = [s.strip() for s in str(r["keep_items"]).split(",") if str(s).strip()]
                    if label.startswith("core-3"):
                        core3_items = items
                    elif label.startswith("core-4"):
                        core4_items = items
                if core3_items or core4_items:
                    print(f"[{dataset_tag}] Detected selection: Core-3 items={core3_items if core3_items else 'NA'}, Core-4 items={core4_items if core4_items else 'NA'}")
            else:
                print(f"[{dataset_tag}] selection_decision.csv found but missing required columns; skipping subset TIFs.")
        else:
            print(f"[{dataset_tag}] selection_decision.csv not found; skipping Core-3/Core-4 subset TIFs.")

        # Helper to compute TIF given params df and item list
        def compute_tif(theta_grid, params_df, items):
            tif = np.zeros_like(theta_grid, dtype=float)
            used = 0
            if params_df is None or params_df.empty or not items:
                return tif, used
            sub = params_df[params_df["item_id"].isin(items)].copy()
            for _, rr in sub.iterrows():
                a = float(rr.get("a_hat", np.nan)); b = float(rr.get("b_hat", np.nan))
                if np.isfinite(a) and np.isfinite(b):
                    tif += iic(theta_grid, a, b)
                    used += 1
            return tif, used

        # Common theta grid
        theta = bcrt_res["theta_grid"] if bcrt_res["ok"] else (crt2_res["theta_grid"] if crt2_res["ok"] else np.linspace(-3,3,121))

        subset_rows = []

        # BCRT Core-3 only
        if core3_items:
            tif, used = compute_tif(theta, bcrt_res["params"] if bcrt_res["ok"] else pd.DataFrame(), core3_items)
            if used > 0:
                fig, ax = plt.subplots(figsize=(6.8,4.8))
                ax.plot(theta, tif, color='#F18F01', linewidth=2.5)
                ax.set_xlabel("Ability (Combined10, z)")
                ax.set_ylabel("Test information")
                ax.set_title("TIF (BCRT Core-3)")
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                plt.tight_layout()
                out_path = base_fig / "tif_bcrt_core3.png"
                plt.savefig(out_path, dpi=200)
                plt.close(fig)
                area = float(np.trapz(tif, theta))
                subset_rows.append({"subset": "bcrt_core3", "items": ",".join(core3_items), "n_items": len(core3_items), "n_used": used, "tif_area": area})
                print(f"[{dataset_tag}] Wrote TIF for BCRT Core-3 to {out_path}")
            else:
                print(f"[{dataset_tag}] Core-3 items found but no valid IRT params; skipping BCRT Core-3 TIF.")

        # BCRT Core-4 only
        if core4_items:
            tif, used = compute_tif(theta, bcrt_res["params"] if bcrt_res["ok"] else pd.DataFrame(), core4_items)
            if used > 0:
                fig, ax = plt.subplots(figsize=(6.8,4.8))
                ax.plot(theta, tif, color='#F18F01', linewidth=2.5)
                ax.set_xlabel("Ability (Combined10, z)")
                ax.set_ylabel("Test information")
                ax.set_title("TIF (BCRT Core-4)")
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                plt.tight_layout()
                out_path = base_fig / "tif_bcrt_core4.png"
                plt.savefig(out_path, dpi=200)
                plt.close(fig)
                area = float(np.trapz(tif, theta))
                subset_rows.append({"subset": "bcrt_core4", "items": ",".join(core4_items), "n_items": len(core4_items), "n_used": used, "tif_area": area})
                print(f"[{dataset_tag}] Wrote TIF for BCRT Core-4 to {out_path}")
            else:
                print(f"[{dataset_tag}] Core-4 items found but no valid IRT params; skipping BCRT Core-4 TIF.")

        # CRT2 + Core-3
        if core3_items:
            all_items = list(core3_items)
            tif_core3, used_c3 = compute_tif(theta, bcrt_res["params"] if bcrt_res["ok"] else pd.DataFrame(), core3_items)
            tif_crt2, used_c2 = compute_tif(theta, crt2_res["params"] if crt2_res["ok"] else pd.DataFrame(), list(crt2_res["params"]["item_id"]) if crt2_res["ok"] else [])
            tif = tif_core3 + tif_crt2
            used = used_c3 + used_c2
            if used > 0 and used_c3 > 0 and used_c2 > 0:
                fig, ax = plt.subplots(figsize=(6.8,4.8))
                ax.plot(theta, tif, color='#F18F01', linewidth=2.5)
                ax.set_xlabel("Ability (Combined10, z)")
                ax.set_ylabel("Test information")
                ax.set_title("TIF (CRT-2 + BCRT Core-3)")
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                plt.tight_layout()
                out_path = base_fig / "tif_crt2_plus_core3.png"
                plt.savefig(out_path, dpi=200)
                plt.close(fig)
                area = float(np.trapz(tif, theta))
                subset_rows.append({"subset": "crt2_plus_core3", "items": ",".join([*all_items, "(all CRT2)"]), "n_items": len(all_items), "n_used": used, "tif_area": area})
                print(f"[{dataset_tag}] Wrote TIF for CRT-2 + BCRT Core-3 to {out_path}")
            else:
                print(f"[{dataset_tag}] Missing valid params for CRT-2 and/or Core-3; skipping CRT-2 + Core-3 TIF.")

        # CRT2 + Core-4
        if core4_items:
            all_items = list(core4_items)
            tif_core4, used_c4 = compute_tif(theta, bcrt_res["params"] if bcrt_res["ok"] else pd.DataFrame(), core4_items)
            tif_crt2, used_c2 = compute_tif(theta, crt2_res["params"] if crt2_res["ok"] else pd.DataFrame(), list(crt2_res["params"]["item_id"]) if crt2_res["ok"] else [])
            tif = tif_core4 + tif_crt2
            used = used_c4 + used_c2
            if used > 0 and used_c4 > 0 and used_c2 > 0:
                fig, ax = plt.subplots(figsize=(6.8,4.8))
                ax.plot(theta, tif, color='#F18F01', linewidth=2.5)
                ax.set_xlabel("Ability (Combined10, z)")
                ax.set_ylabel("Test information")
                ax.set_title("TIF (CRT-2 + BCRT Core-4)")
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                plt.tight_layout()
                out_path = base_fig / "tif_crt2_plus_core4.png"
                plt.savefig(out_path, dpi=200)
                plt.close(fig)
                area = float(np.trapz(tif, theta))
                subset_rows.append({"subset": "crt2_plus_core4", "items": ",".join([*all_items, "(all CRT2)"]), "n_items": len(all_items), "n_used": used, "tif_area": area})
                print(f"[{dataset_tag}] Wrote TIF for CRT-2 + BCRT Core-4 to {out_path}")
            else:
                print(f"[{dataset_tag}] Missing valid params for CRT-2 and/or Core-4; skipping CRT-2 + Core-4 TIF.")

        # CRT2 + all BCRT items available
        if bcrt_res["ok"] and crt2_res["ok"]:
            all_bcrt_items = list(bcrt_res["params"]["item_id"]) if not bcrt_res["params"].empty else []
            tif_bcrt_all, used_b = compute_tif(theta, bcrt_res["params"], all_bcrt_items)
            tif_crt2, used_c2 = compute_tif(theta, crt2_res["params"], list(crt2_res["params"]["item_id"]))
            tif = tif_bcrt_all + tif_crt2
            used = used_b + used_c2
            if used > 0 and used_b > 0 and used_c2 > 0:
                fig, ax = plt.subplots(figsize=(6.8,4.8))
                ax.plot(theta, tif, color='#F18F01', linewidth=2.5)
                ax.set_xlabel("Ability (Combined10, z)")
                ax.set_ylabel("Test information")
                ax.set_title("TIF (CRT-2 + all BCRT)")
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                plt.tight_layout()
                out_path = base_fig / "tif_crt2_plus_bcrt_all.png"
                plt.savefig(out_path, dpi=200)
                plt.close(fig)
                area = float(np.trapz(tif, theta))
                subset_rows.append({"subset": "crt2_plus_bcrt_all", "items": "(all BCRT) + (all CRT2)", "n_items": len(all_bcrt_items), "n_used": used, "tif_area": area})
                print(f"[{dataset_tag}] Wrote TIF for CRT-2 + all BCRT to {out_path}")
            else:
                print(f"[{dataset_tag}] Missing valid params for CRT-2 and/or BCRT; skipping CRT-2 + all BCRT TIF.")

        # Write subset summary table if any
        if subset_rows:
            sub_df = pd.DataFrame(subset_rows)
            sub_out = base_tab / "irt2pl_subsets.csv"
            sub_df.to_csv(sub_out, index=False)
            print(f"[{dataset_tag}] Wrote subset TIF summary table to {sub_out}")
    except Exception as e:
        print(f"[{dataset_tag}] Subset TIF generation encountered an error: {e}")

    return bool(crt2_res.get("ok", False)) and bool(bcrt_res.get("ok", False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["full","naive","both"], default="both",
                        help="Which dataset to run: full, naive, or both (default).")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml"))
    interim = Path(cfg["paths"]["interim"])  # scored_long.csv lives here
    processed = Path(cfg["paths"]["processed"])  # person_wide / person_naive live here

    item_long = pd.read_csv(interim / "scored_long.csv")
    person_wide = pd.read_csv(processed / "person_wide.csv")
    person_naive = pd.read_csv(processed / "person_naive.csv")

    if args.dataset in ("full","both"):
        run_for_dataset("full", item_long, person_wide, person_naive)
    if args.dataset in ("naive","both"):
        run_for_dataset("naive", item_long, person_wide, person_naive)

    print("Done: 2PL outputs written under reports/figures/irt2pl/{full,naive} and reports/tables/{full,naive}")

if __name__ == "__main__":
    main()
