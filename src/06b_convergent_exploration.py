"""
LIWC–CRT exploratory correlations (FULL and NAIVE datasets)

Why this is exploratory
- The goal is to surface patterns rather than produce inferentially adjusted,
  model-based estimates. We compute unadjusted Pearson correlations across a
  broad set of LIWC child variables and multiple predictors, apply simple
  p<.05 flags (no multiplicity correction), and visualise the landscape. This
  helps generate hypotheses and guide deeper analyses performed in
  06_convergent_validity.py (which includes HC3/clustered SEs and BH–FDR).

What this script does
- Reads the LIWC variable manifest for each dataset to identify granular LIWC
  child variables (excludes summary/parent totals and operational columns).
- Loads person files (`person_with_liwc.csv` for full, `person_naive_with_liwc.csv`
  for naive), ensures Core3/Core4 totals and composite predictors exist, and
  computes corrected `nfc_total` when NFC items are present.
- Collapses LIWC Q1/Q2 into a single value per participant using a WC-weighted
  average when possible (falls back to simple mean or single available block).
- Computes Pearson correlations (r, two-sided p) between LIWC children and the
  following predictors (as available with variance): CRT2, Core3, Core4,
  CRT2+Core3, CRT2+Core4, CRT2+BCRT(all), NFC.
- Writes long- and wide-form correlation tables and a heatmap of signed r with
  simple p<.05 overlays, plus a human-readable summary grouped by LIWC category.

Outputs
- reports/tables/liwc_exploration/<dataset>/correlations_long.csv
- reports/tables/liwc_exploration/<dataset>/correlations_wide_r.csv
- reports/tables/liwc_exploration/<dataset>/summary_by_predictor.md
- reports/figures/liwc_exploration/<dataset>/heatmap_correlations.png (.pdf)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Reuse helpers to ensure totals/composites exist
import re


BASE = Path(".")
PROCESSED = BASE / "data" / "processed"
TABLES = BASE / "reports" / "tables"
FIGS = BASE / "reports" / "figures"

TABLES_OUT = TABLES / "liwc_exploration"
FIGS_OUT = FIGS / "liwc_exploration"
TABLES_OUT.mkdir(parents=True, exist_ok=True)
FIGS_OUT.mkdir(parents=True, exist_ok=True)


# --- Minimal helpers replicated from 06_convergent_validity.py ---
def correct_nfc_scoring(df: pd.DataFrame) -> pd.Series:
    """Apply correct NFC scoring (9 forward, 9 reverse) to Qualtrics-labeled columns."""
    reverse = [3, 4, 5, 7, 8, 9, 12, 16, 17]
    items = [f"nfc_{i}" for i in range(1, 19)]
    label_to_num = {
        "1 = Extremely Uncharacteristic": 1,
        "2 = Somewhat Uncharacteristic": 2,
        "3 = Uncertain": 3,
        "4 = Somewhat Characteristic": 4,
        "5 = Extremely Characteristic": 5,
    }
    num = pd.DataFrame({c: df[c].map(label_to_num) for c in items if c in df.columns})
    for i in reverse:
        col = f"nfc_{i}"
        if col in num.columns:
            num[col] = 6 - num[col]
    return num.sum(axis=1)


def _norm_item_name(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    if s.startswith("bcrt_q"):
        return s
    m = re.search(r"(\d+)", s)
    return f"bcrt_q{int(m.group(1))}" if m else None


def load_core_item_sets() -> dict:
    sel_candidates = [
        TABLES / "selection_decision.csv",
        TABLES / "selection_rule" / "selection_decision.csv",
        BASE / "selection_decision.csv",
    ]
    sel_path = None
    for p in sel_candidates:
        if p.exists():
            sel_path = p
            break
    if sel_path is None:
        try:
            sel_path = next(BASE.rglob("selection_decision.csv"))
        except StopIteration:
            print("[selection] selection_decision.csv not found; Core3/Core4 unavailable.")
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
    return out


def ensure_core_totals(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    need3 = ("bcrt_core3_total" not in df.columns)
    need4 = ("bcrt_core4_total" not in df.columns)
    if need3 or need4:
        sets = load_core_item_sets()
        if sets:
            need_items = set(sets.get("core3", []) + sets.get("core4", []))
            missing_items = [c for c in need_items if c not in df.columns]
            if missing_items:
                try:
                    scored_long = pd.read_csv(BASE / "data" / "interim" / "scored_long.csv")
                    wide = (
                        scored_long.pivot_table(
                            index="pid", columns="item_id", values="correct", aggfunc="first"
                        ).reset_index()
                    )
                    df = df.merge(wide, on="pid", how="left")
                except Exception as e:
                    print(f"[selection] Could not merge scored_long for core totals: {e}")
            if need3 and sets.get("core3"):
                have = [c for c in sets["core3"] if c in df.columns]
                if have:
                    df["bcrt_core3_total"] = df[have].sum(axis=1)
            if need4 and sets.get("core4"):
                have = [c for c in sets["core4"] if c in df.columns]
                if have:
                    df["bcrt_core4_total"] = df[have].sum(axis=1)
    # Composite totals
    try:
        if (
            ("crt2_total" in df.columns)
            and ("bcrt_core3_total" in df.columns)
            and ("crt2_plus_core3_total" not in df.columns)
        ):
            df["crt2_plus_core3_total"] = df["crt2_total"].astype(float) + df["bcrt_core3_total"].astype(float)
        if (
            ("crt2_total" in df.columns)
            and ("bcrt_core4_total" in df.columns)
            and ("crt2_plus_core4_total" not in df.columns)
        ):
            df["crt2_plus_core4_total"] = df["crt2_total"].astype(float) + df["bcrt_core4_total"].astype(float)
    except Exception as e:
        print(f"[selection] Could not compute composite totals: {e}")
    return df

def load_manifest_children(manifest_csv: Path) -> list[str]:
    """Return list of LIWC base variable names that are considered "children".

    Heuristic, using naming present in the manifest:
    - Exclude known summary/macros and operational columns
    - Include granular child variables, typically lowercase categories
    - Exclude mid-level rollups that we do not want (e.g., function, pronoun)
    """
    dfm = pd.read_csv(manifest_csv)
    bases = dfm["base"].dropna().astype(str).unique().tolist()

    # Exclusions: upper-level or operational variables
    exclude_exact = set(
        [
            # Operational/length metrics
            "WC",
            "WPS",
            "BigWords",
            "Dic",
            "Segment",
            # Summary rollups
            "Analytic",
            "Clout",
            "Authentic",
            "Tone",
            "Affect",
            "Cognition",
            "Linguistic",
            "Drives",
            "Social",
            "Perception",
            "Physical",
            # NFC scaffolding
            "NFC_total_raw",
        ]
    )
    # Exclude some lowercase rollups that are still parents in LIWC2015
    exclude_lower_rollups = {"function", "pronoun", "ppron", "ipron", "cogproc"}

    children = []
    for b in bases:
        if b in exclude_exact:
            continue
        if b in exclude_lower_rollups:
            continue
        # keep typical children (mostly lowercase); allow some lowercase that are fine
        children.append(b)
    children = sorted(set(children))
    return children


def wc_weighted_average(df: pd.DataFrame, base: str) -> pd.Series:
    """Return a per-participant single value combining Q1/Q2 for a LIWC base.
    Uses WC-weighted average if both blocks present, else mean/single.
    Values are in percentages in LIWC; weighting uses word counts.
    """
    y1, y2 = f"{base}_Q1", f"{base}_Q2"
    has1, has2 = (y1 in df.columns), (y2 in df.columns)
    wc1 = df.get("WC_Q1")
    wc2 = df.get("WC_Q2")
    if has1 and has2 and (wc1 is not None) and (wc2 is not None):
        num = df[y1].astype(float) * wc1.astype(float) + df[y2].astype(float) * wc2.astype(float)
        den = wc1.astype(float) + wc2.astype(float)
        out = np.divide(num, np.where(den > 0, den, np.nan))
        return pd.Series(out)
    if has1 and has2:
        return df[[y1, y2]].astype(float).mean(axis=1)
    if has1:
        return df[y1].astype(float)
    if has2:
        return df[y2].astype(float)
    # fall back to non-split column if present
    if base in df.columns:
        return df[base].astype(float)
    return pd.Series(np.nan, index=df.index)


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Return (r, p, n) for finite-pair Pearson correlation, robust to SciPy absence."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    xv, yv = x[mask], y[mask]
    r = float(np.corrcoef(xv, yv)[0, 1])
    # convert r to t and p (two-sided)
    df = max(n - 2, 1)
    if not np.isfinite(r):
        return np.nan, np.nan, n
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    try:
        from scipy.stats import t as sp_t  # type: ignore

        p = 2 * (1 - sp_t.cdf(abs(t), df))
    except Exception:
        # fallback via survival of standard normal approximation
        from math import erf, sqrt

        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return r, float(p), n


def categorize_base(base: str) -> str:
    """Map LIWC base to a coarse category for summary reporting."""
    b = base.lower()
    # Function words / analytic
    if b in {
        "article",
        "prep",
        "i",
        "we",
        "you",
        "shehe",
        "they",
        "ipron",
        "ppron",
        "auxverb",
        "adverb",
        "conj",
        "negate",
        "verb",
        "adj",
        "det",
    }:
        return "FunctionAnalytic"
    # Cognitive
    if b in {"insight", "cause", "discrep", "tentat", "differ", "certitude", "cogproc"}:
        return "CognitiveComplexity"
    # Affect/Tone
    if b.startswith("emo_") or b in {"emotion", "tone_pos", "tone_neg"}:
        return "ToneAffect"
    # Quantity/Length
    if b in {"number", "quantity"}:
        return "QuantityLength"
    # Social
    if b in {"social", "family", "friend", "female", "male", "prosocial", "comm", "conversation"}:
        return "Social"
    # Perception
    if b in {"perception", "visual", "auditory", "see", "hear", "feel"}:
        return "Perception"
    # Biological / Health
    if b in {"bio", "body", "health", "illness", "physical", "wellness", "sexual", "substances"}:
        return "BiologicalHealth"
    # Drives / Motivation
    if b in {"affiliation", "achieve", "power", "reward", "risk", "need", "want", "curiosity", "allure"}:
        return "DrivesMotivation"
    # Time / Relativity
    if b in {"focuspast", "focuspresent", "focusfuture", "time"}:
        return "TimeFocus"
    if b in {"space", "motion"}:
        return "Relativity"
    # Style / Other
    if b in {"polite", "swear", "netspeak", "tech"}:
        return "Style"
    return "Other"


def process_dataset(dataset: str):
    # Paths per dataset
    manifest_csv = TABLES / f"liwc_variables_manifest_{dataset}.csv"
    person_csv = (
        PROCESSED / ("person_naive_with_liwc.csv" if dataset == "naive" else "person_with_liwc.csv")
    )
    if not manifest_csv.exists() or not person_csv.exists():
        print(f"[{dataset}] Missing manifest or person file. Run 06_convergent_validity first.")
        return

    liwc_children = load_manifest_children(manifest_csv)
    print(f"[{dataset}] Using {len(liwc_children)} LIWC child bases")

    df = pd.read_csv(person_csv)
    # Ensure Core totals / composites
    df = ensure_core_totals(df)
    # NFC correction
    nfc_items = [f"nfc_{i}" for i in range(1, 19)]
    if any(col in df.columns for col in nfc_items):
        df["NFC_total"] = correct_nfc_scoring(df)
        df["nfc_total"] = df["NFC_total"]

    # Build a per-person LIWC child matrix (weighted collapse of Q1/Q2)
    liwc_mat = {}
    for b in liwc_children:
        s = wc_weighted_average(df, b)
        if s.notna().sum() > 0:
            liwc_mat[b] = s
    liwc_df = pd.DataFrame(liwc_mat)
    liwc_df.index = df.index

    # Predictors
    predictors = {
        "CRT2": "crt2_total",
        "Core3": "bcrt_core3_total",
        "Core4": "bcrt_core4_total",
        "CRT2+Core3": "crt2_plus_core3_total",
        "CRT2+Core4": "crt2_plus_core4_total",
        "CRT2+BCRT(all)": "combined10_total",
        "NFC": "nfc_total",
    }
    pred_cols = {k: v for k, v in predictors.items() if v in df.columns and df[v].std(skipna=True) > 0}
    if not pred_cols:
        print(f"[{dataset}] No predictors available with variance; skipping")
        return

    # Compute correlations long-form
    rows = []
    for base in liwc_df.columns:
        y = liwc_df[base].to_numpy()
        for label, col in pred_cols.items():
            x = df[col].to_numpy()
            r, p, n = pearson_r_p(x, y)
            rows.append({"base": base, "predictor": label, "r": r, "p": p, "n": n})
    corr_long = pd.DataFrame(rows)
    corr_long["sig"] = corr_long["p"] < 0.05
    # Dataset-specific outputs
    ds_tables = TABLES_OUT / dataset
    ds_figs = FIGS_OUT / dataset
    ds_tables.mkdir(parents=True, exist_ok=True)
    ds_figs.mkdir(parents=True, exist_ok=True)
    corr_long.to_csv(ds_tables / "correlations_long.csv", index=False)
    print("Wrote", ds_tables / "correlations_long.csv")

    # Wide r-matrix for heatmap
    r_wide = corr_long.pivot(index="base", columns="predictor", values="r").fillna(0.0)
    p_wide = corr_long.pivot(index="base", columns="predictor", values="p")

    # Ordering rules
    row_score = r_wide.abs().mean(axis=1)
    col_sig_counts = (p_wide < 0.05).sum(axis=0)
    col_strength = r_wide.abs().mean(axis=0)
    col_order = (
        pd.DataFrame({"sig_count": col_sig_counts, "strength": col_strength})
        .sort_values(["sig_count", "strength"], ascending=[False, False])
        .index.tolist()
    )
    row_order = row_score.sort_values(ascending=True).index.tolist()

    r_ord = r_wide.loc[row_order, col_order]
    p_ord = p_wide.loc[row_order, col_order]
    r_ord.to_csv(ds_tables / "correlations_wide_r.csv")
    print("Wrote", ds_tables / "correlations_wide_r.csv")

    # Heatmap with significance overlay
    fig_h = max(4.0, 0.25 * len(row_order) + 2.0)
    fig_w = max(6.0, 0.65 * len(col_order) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    im = ax.imshow(r_ord.values, aspect="auto", interpolation="none", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_title(f"LIWC children vs CRT predictors — correlations ({dataset})")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Pearson r")
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            if pd.notna(p_ord.iat[i, j]) and p_ord.iat[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8, color="black")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    out_png = ds_figs / "heatmap_correlations.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_png)

    # Summary by predictor: significant variables grouped by category, ordered by |r|
    lines = [f"# Significant LIWC correlations by predictor ({dataset})\n"]
    for label in col_order:
        sub = corr_long[(corr_long["predictor"] == label) & (corr_long["p"] < 0.05)].copy()
        if sub.empty:
            lines.append(f"\n## {label}\n(No significant correlations at p<.05)\n")
            continue
        sub["abs_r"] = sub["r"].abs()
        sub["category"] = sub["base"].map(categorize_base)
        sub = sub.sort_values("abs_r", ascending=False)
        lines.append(f"\n## {label}\n")
        for cat, grp in sub.groupby("category", sort=False):
            lines.append(f"### {cat}\n")
            for _, r in grp.iterrows():
                star = "*" if r["p"] < 0.05 else ""
                lines.append(f"- {r['base']}: r={r['r']:.3f}, p={r['p']:.3g} {star}")
            lines.append("")
    (ds_tables / "summary_by_predictor.md").write_text("\n".join(lines))
    print("Wrote", ds_tables / "summary_by_predictor.md")


def main():
    for dataset in ("full", "naive"):
        process_dataset(dataset)


if __name__ == "__main__":
    main()


