
import os, yaml, pandas as pd, numpy as np, re, json
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))

raw_dir = Path(cfg["paths"]["raw"])
interim_dir = Path(cfg["paths"]["interim"])
interim_dir.mkdir(parents=True, exist_ok=True)

csv_path = raw_dir / cfg["input"]["qualtrics_csv"]

# Read Qualtrics with two-level headers, then drop first 2 meta rows
df = pd.read_csv(csv_path, header=[0,1], dtype=str)
df = df.iloc[2:].reset_index(drop=True)

# flatten to level0 variable names
df.columns = [c[0] for c in df.columns]

# Basic ID column
pid = None
for cand in cfg["columns"]["pid_candidates"]:
    if cand in df.columns:
        pid = cand; break
if pid is None:
    pid = "ResponseId"
    df[pid] = np.arange(len(df)).astype(str)

df.rename(columns={pid: "pid"}, inplace=True)

# cast familiarity to 0/1
fam_cols = cfg["columns"]["familiarity"]
for c in fam_cols:
    if c in df.columns:
        df[c] = df[c].fillna("").str.lower().str.strip()
        df[c] = df[c].replace({"yes":"1","no":"0"})
        df[c] = df[c].replace({"true":"1","false":"0"})
        df[c] = df[c].replace({"1":"1","0":"0"})
        df[c] = df[c].apply(lambda x: "1" if x in ("1","yes","y","true") else ("0" if x in ("0","no","n","false","") else "0"))
    else:
        df[c] = "0"

# seen_count
df["seen_count"] = df[fam_cols].astype(int).sum(axis=1)

# Save cleaned
out_path = interim_dir / "resp_raw_clean.parquet"
df.to_csv(interim_dir / 'resp_raw_clean.csv', index=False)
print(f"Saved: data/interim/resp_raw_clean.csv with shape {df.shape}")
