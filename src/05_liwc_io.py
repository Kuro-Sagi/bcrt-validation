# src/05_liwc_io.py
# Export writing texts for LIWC and merge LIWC CSVs back into the analysis dataset.

import argparse, re, yaml, pandas as pd, numpy as np
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))
paths = cfg["paths"]
interim = Path(paths["interim"])
processed = Path(paths["processed"])
raw = Path(paths["raw"])
interim.mkdir(parents=True, exist_ok=True)
processed.mkdir(parents=True, exist_ok=True)

Q1 = cfg["columns"]["writing_q1"]
Q2 = cfg["columns"]["writing_q2"]

def safe_text(x):
    if not isinstance(x, str): return ""
    # Strip Qualtrics artefacts, normalise whitespace
    x = re.sub(r"\s+", " ", x).strip()
    return x

def export_texts():
    # Use the latest person-wide if available, else fall back to raw-clean (interim)
    candidates = [
        processed / "person_wide.csv",
        interim / "resp_raw_clean.csv"
    ]
    for f in candidates:
        if f.exists():
            df = pd.read_csv(f)
            break
    else:
        raise FileNotFoundError("No person_wide.csv or resp_raw_clean.csv found")

    if "pid" not in df.columns:
        raise RuntimeError(f"No 'pid' column found in {f.name}. Ensure 00_load_and_qc.py has been run.")
    df["pid"] = df["pid"].astype(str).str.strip()
    for col in [Q1, Q2]:
        if col not in df.columns:
            df[col] = ""

    base = interim / "liwc_texts"
    (base / "WritingTask_Q1").mkdir(parents=True, exist_ok=True)
    (base / "WritingTask_Q2").mkdir(parents=True, exist_ok=True)

    out_rows_q1, out_rows_q2 = [], []
    for _, r in df.iterrows():
        pid = str(r["pid"])
        t1 = safe_text(r.get(Q1, ""))
        t2 = safe_text(r.get(Q2, ""))
        # one file per pid per question (LIWC likes per-file units)
        (base / "WritingTask_Q1" / f"{pid}.txt").write_text(t1)
        (base / "WritingTask_Q2" / f"{pid}.txt").write_text(t2)
        out_rows_q1.append({"pid": pid, "text": t1})
        out_rows_q2.append({"pid": pid, "text": t2})

    pd.DataFrame(out_rows_q1).to_csv(base / "q1.csv", index=False)
    pd.DataFrame(out_rows_q2).to_csv(base / "q2.csv", index=False)
    print(f"Exported texts to {base}/WritingTask_Q1 and WritingTask_Q2, with q1.csv/q2.csv")

def _read_liwc_csv(path):
    df = pd.read_csv(path)
    # Typical LIWC columns: Filename, Segment, WC, Analytic, Clout, Authentic, Tone, ...
    # Map pid from Filename (strip extension)
    name_col = None
    for c in df.columns:
        if str(c).lower() in ("filename","file","doc","document"): 
            name_col = c; break
    if name_col is None:
        raise RuntimeError(f"Could not find a filename column in {path}")
    df["pid"] = df[name_col].astype(str).str.replace(r"\\.txt$","",regex=True)
    return df

REFLECTIVE_CANDIDATES = [
    "Analytic","Clout","Authentic","Tone",
    "CogProc","CognitiveProcesses","Cognition",
    "Insight","Cause","Causal","Tentat","Differ","Quant","Number",
    "Work","Achieve","WC","WPS"
]

def merge_liwc(q1_csv, q2_csv):
    liwc1 = _read_liwc_csv(q1_csv); liwc1["writing_task"] = "Q1"
    liwc2 = _read_liwc_csv(q2_csv); liwc2["writing_task"] = "Q2"
    liwc = pd.concat([liwc1, liwc2], ignore_index=True, sort=False)

    # If multiple segments per pid, average weighted by WC
    if "WC" in liwc.columns:
        w = liwc["WC"].replace(0, np.nan)
        grouped = []
        for (pid, task), g in liwc.groupby(["pid","writing_task"]):
            weights = g["WC"].replace(0, np.nan)
            cols = [c for c in g.columns if c not in ["pid","writing_task"]]
            num_cols = g[cols].select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols)==0: 
                agg = g.iloc[[0]].copy()
            else:
                agg_vals = g[num_cols].multiply(weights, axis=0).sum() / weights.sum()
                agg = pd.DataFrame([agg_vals])
            agg["pid"] = pid; agg["writing_task"] = task
            grouped.append(agg)
        liwc_agg = pd.concat(grouped, ignore_index=True, sort=False)
    else:
        liwc_agg = (liwc.groupby(["pid","writing_task"]).mean(numeric_only=True).reset_index())

    # wide by task suffix
    wide = liwc_agg.pivot(index="pid", columns="writing_task")
    wide.columns = ["{}_{}".format(a, b) for a,b in wide.columns]
    wide = wide.reset_index()

    # join onto person_wide
    pw = pd.read_csv(processed / "person_wide.csv")
    if "pid" not in pw.columns:
        raise RuntimeError("person_wide.csv has no 'pid' column. Ensure 00_load_and_qc.py has been run.")
    pw["pid"] = pw["pid"].astype(str).str.strip()
    # Clean PIDs in wide format to remove .txt extension if present
    wide["pid"] = wide["pid"].astype(str).str.replace(r"\.txt$", "", regex=True).str.strip()
    out = pw.merge(wide, on="pid", how="left")
    out.to_csv(processed / "person_with_liwc.csv", index=False)
    wide.to_csv(processed / "liwc_merged.csv", index=False)

    # save a list of the LIWC variables we'll consider
    liwc_vars = [c for c in out.columns for key in REFLECTIVE_CANDIDATES if key.lower() in c.lower()]
    pd.DataFrame({"liwc_var": sorted(set(liwc_vars))}).to_csv(processed / "liwc_var_list.csv", index=False)

    print("Merged LIWC -> data/processed/person_with_liwc.csv and liwc_merged.csv")
    print(f"Candidate LIWC variables list: data/processed/liwc_var_list.csv (n={len(set(liwc_vars))})")
    
    # Also create naive subset if person_naive.csv exists
    naive_path = processed / "person_naive.csv"
    if naive_path.exists():
        naive = pd.read_csv(naive_path)[["pid"]]
        naive["pid"] = naive["pid"].astype(str).str.strip()
        
        naive_out = naive.merge(out, on="pid", how="inner")
        naive_out.to_csv(processed / "person_naive_with_liwc.csv", index=False)
        print(f"Also created naive subset -> person_naive_with_liwc.csv (n={len(naive_out)})")
    else:
        print("Note: person_naive.csv not found, skipping naive subset creation. Run 02_make_splits.py first if needed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["export","merge"], required=True, 
                   help="export: Export texts for LIWC analysis; merge: Merge LIWC results (creates both full and naive datasets)")
    ap.add_argument("--q1", default=str(raw / "LIWC_Q1.csv"))
    ap.add_argument("--q2", default=str(raw / "LIWC_Q2.csv"))
    args = ap.parse_args()

    if args.mode == "export":
        export_texts()
    else:  # merge
        merge_liwc(args.q1, args.q2)