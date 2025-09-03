
import os, re, yaml, pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

cfg = yaml.safe_load(open("config/config.yaml"))
processed_dir = Path(cfg["paths"]["processed"])
interim_dir = Path(cfg["paths"]["interim"])
raw_dir = Path(cfg["paths"]["raw"])
reports_tables = Path(cfg["paths"]["reports"]) / "tables"
reports_tables.mkdir(parents=True, exist_ok=True)

person = pd.read_csv(processed_dir / "person_wide.csv")
q1_col = cfg["columns"]["writing_q1"]
q2_col = cfg["columns"]["writing_q2"]

# === A) Export texts for LIWC ===
q1_dir = interim_dir / "liwc_texts" / "q1"
q2_dir = interim_dir / "liwc_texts" / "q2"
for d in (q1_dir, q2_dir):
    d.mkdir(parents=True, exist_ok=True)

def write_txts(series, outdir, suffix):
    manifest = []
    for pid, text in series.fillna("").items():
        filename = f"{pid}_{suffix}.txt"
        (outdir / filename).write_text(str(text), encoding="utf-8")
        wc = len(str(text).split())
        manifest.append({"pid": pid, "filename": filename, "question": suffix, "wordcount": wc})
    return pd.DataFrame(manifest)

m1 = write_txts(person.set_index("pid")[q1_col], q1_dir, "Q1")
m2 = write_txts(person.set_index("pid")[q2_col], q2_dir, "Q2")
m = pd.concat([m1, m2], ignore_index=True)
m.to_csv(interim_dir / "liwc_manifest.csv", index=False)
print("Exported LIWC texts to:", q1_dir, "and", q2_dir)

# === B) Merge LIWC outputs (after you run LIWC GUI) ===
# Expect files: LIWC2022_Writing_Q1.csv and LIWC2022_Writing_Q2.csv in data/raw/
q1_file = raw_dir / "LIWC2022_Writing_Q1.csv"
q2_file = raw_dir / "LIWC2022_Writing_Q2.csv"

def load_liwc_csv(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Expect a 'Filename' column that contains the txt file names
    # Standard columns include: WC, Analytic, Clout, Authentic, Tone, and category columns like 'cogproc','insight','cause','tentat','certain','number','quant','sixltr'
    # Normalise column names to lower-case
    df.columns = [c.strip().lower() for c in df.columns]
    if "filename" not in df.columns:
        # try alternative
        cand = [c for c in df.columns if "file" in c]
        if cand:
            df.rename(columns={cand[0]: "filename"}, inplace=True)
    return df

liwc_q1 = load_liwc_csv(q1_file)
liwc_q2 = load_liwc_csv(q2_file)

if liwc_q1 is None or liwc_q2 is None:
    print(">>> Place LIWC outputs in data/raw/ as LIWC2022_Writing_Q1.csv and LIWC2022_Writing_Q2.csv, then re-run this script to merge.")
else:
    # parse pid back from filename "<pid>_Q1.txt"
    def parse_pid(s):
        m = re.match(r"^(.+?)_(Q1|Q2)\.txt$", s)
        return m.group(1) if m else None
    liwc_q1["pid"] = liwc_q1["filename"].apply(parse_pid)
    liwc_q2["pid"] = liwc_q2["filename"].apply(parse_pid)

    # Select fields
    keep = ["pid","filename","wc","analytic","clout","authentic","tone",
            "cogproc","insight","cause","tentat","certain","number","quant","sixltr"]
    for df in (liwc_q1, liwc_q2):
        for c in keep:
            if c not in df.columns:
                df[c] = np.nan

    liwc_q1 = liwc_q1[keep].copy(); liwc_q1["question"]="Q1"
    liwc_q2 = liwc_q2[keep].copy(); liwc_q2["question"]="Q2"
    liwc_all = pd.concat([liwc_q1, liwc_q2], ignore_index=True)

    # Pivot to wide per PID with Q1/Q2 suffixes
    def pivot_suffix(df, var):
        wide = df.pivot_table(index="pid", columns="question", values=var, aggfunc="first")
        wide.columns = [f"{var}_{c}" for c in wide.columns]
        return wide

    vars_out = ["wc","analytic","clout","authentic","tone","cogproc","insight","cause","tentat","certain","number","quant","sixltr"]
    parts = [pivot_suffix(liwc_all, v) for v in vars_out]
    liwc_wide = pd.concat(parts, axis=1)

    # Aggregate averages across Q1/Q2
    avg = {}
    for v in vars_out:
        cols = [f"{v}_Q1", f"{v}_Q2"]
        present = [c for c in cols if c in liwc_wide.columns]
        if present:
            avg[f"{v}_avg"] = liwc_wide[present].mean(axis=1, skipna=True)
    liwc_wide = pd.concat([liwc_wide, pd.DataFrame(avg)], axis=1)

    # Build RLI and residualise by WC_avg
    def zscore(s):
        return (s - s.mean())/s.std(ddof=0) if s.std(ddof=0) not in (0, np.nan) else s*0

    # Choose number or quant (use whichever exists)
    num_series = liwc_wide.get("number_avg", liwc_wide.get("quant_avg", pd.Series(index=liwc_wide.index, dtype=float)))

    rli = (
        zscore(liwc_wide.get("analytic_avg")) +
        zscore(liwc_wide.get("cogproc_avg")) +
        0.5*zscore(liwc_wide.get("insight_avg")) +
        0.5*zscore(liwc_wide.get("cause_avg")) +
        0.5*zscore(liwc_wide.get("tentat_avg")) +
        0.5*zscore(num_series) +
        0.5*zscore(liwc_wide.get("sixltr_avg")) -
        0.5*zscore(liwc_wide.get("certain_avg"))
    )
    liwc_wide["RLI"] = rli

    wc_avg = liwc_wide.get("wc_avg", liwc_wide.filter(regex=r"^wc_").mean(axis=1))
    wc_avg = wc_avg.fillna(0).values.reshape(-1,1)
    rli_vals = liwc_wide["RLI"].fillna(0).values.reshape(-1,1)
    # Residualise via OLS
    reg = LinearRegression().fit(wc_avg, rli_vals)
    rli_resid = rli_vals - reg.predict(wc_avg)
    liwc_wide["RLI_resid"] = rli_resid

    # Merge into person files
    person_naive = pd.read_csv(processed_dir / "person_naive.csv")
    person_full = pd.read_csv(processed_dir / "person_full.csv")
    person_naive = person_naive.merge(liwc_wide, left_on="pid", right_index=True, how="left")
    person_full = person_full.merge(liwc_wide, left_on="pid", right_index=True, how="left")

    person_naive.to_csv(processed_dir / "person_naive_liwc.csv", index=False)
    person_full.to_csv(processed_dir / "person_full_liwc.csv", index=False)
    liwc_wide.to_csv(processed_dir / "liwc_wide.csv", index=False)
    print("Merged LIWC into person files.")
