# src/01_score_items.py
import os, yaml, pandas as pd, numpy as np, re
from pathlib import Path
from utils import (
    normalise_answer, match_any, match_regex, match_any_fuzzy,
    extract_first_number, contains_any
)
from tqdm import tqdm
import time

cfg = yaml.safe_load(open("config/config.yaml"))
interim_dir = Path(cfg["paths"]["interim"])
processed_dir = Path(cfg["paths"]["processed"])
reports_dir = Path(cfg["paths"]["reports"])
processed_dir.mkdir(parents=True, exist_ok=True)
(reports_dir / "tables").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(interim_dir / "resp_raw_clean.csv")

ckpt_path = interim_dir / "resp_scoring_progress.csv"
if ckpt_path.exists():
    df = pd.read_csv(ckpt_path)
    print(f"Resuming from checkpoint: {ckpt_path} (shape={df.shape})")
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "200"))

FORCE_RECLASS = os.environ.get("RECLASSIFY_ALL", "0") == "1"
if FORCE_RECLASS:
    print("RECLASSIFY_ALL=1 → existing routes will be ignored and recomputed.")

crt2 = cfg["columns"]["crt2_items"]
bcrt = cfg["columns"]["bcrt_items"]
writing_q1 = cfg["columns"]["writing_q1"]
writing_q2 = cfg["columns"]["writing_q2"]
scoring = cfg["scoring"]

# ---- LLM (forced) -----------------------------------------------------------
try:
    from llm_classify import classify_with_llm
except Exception:
    classify_with_llm = lambda *args, **kwargs: None

use_llm = bool(os.environ.get("OPENAI_API_KEY"))
model_name = os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL_SNAPSHOT", "gpt-5-nano"))

# ---- Rules fallback (only when no API) --------------------------------------
def score_item_rules(answer: str, key: dict, item_id: str):
    """Rules-only scoring used when LLM is unavailable. Returns (correct, resp_type, route)."""
    a = normalise_answer(answer)
    if a == "": return 0, 0, "empty"
    if key is None: return 0, 0, "no_key"

    corr_list = key.get("correct", [])
    intu_list = key.get("intuitive", [])
    corr_phr  = key.get("correct_phrases", [])
    intu_phr  = key.get("intuitive_phrases", [])
    intu_regex = key.get("intuitive_pattern")

    # exact
    if corr_list and match_any(a, corr_list): return 1, 2, "rule_exact_correct"
    if intu_list and match_any(a, intu_list): return 0, 1, "rule_exact_intuitive"
    # regex / numeric
    if intu_regex and match_regex(a, intu_regex):
        if a.strip() in ("0","0.0","zero","none","no dirt"): return 1, 2, "rule_regex_zero_correct"
        return 0, 1, "rule_regex_intuitive"
    want_num = key.get("correct_number", None)
    if want_num is not None:
        got_num = extract_first_number(a)
        if got_num is not None and abs(float(want_num) - got_num) < 1e-6:
            return 1, 2, "rule_numeric_correct"
    # fuzzy
    if corr_list and match_any_fuzzy(a, corr_list, threshold=90): return 1, 2, "rule_fuzzy_correct"
    if intu_list and match_any_fuzzy(a, intu_list, threshold=90): return 0, 1, "rule_fuzzy_intuitive"
    # phrases
    if corr_phr and contains_any(a, corr_phr): return 1, 2, "rule_phrase_correct"
    if intu_phr and contains_any(a, intu_phr): return 0, 1, "rule_phrase_intuitive"

    return 0, 0, "rule_other"

# ---- LLM-first scorer -------------------------------------------------------
def score_item(answer: str, key: dict, item_id: str):
    """
    Always try LLM first for any non-empty answer (correct/intuitive/other).
    If API is missing or call fails, fall back to rules.
    """
    a = normalise_answer(answer)
    if a == "": return 0, 0, "empty"
    if key is None: return 0, 0, "no_key"

    if use_llm:
        label = classify_with_llm(item_id, item_id, answer, key)
        if label == "correct":   return 1, 2, "llm_correct_forced"
        if label == "intuitive": return 0, 1, "llm_intuitive_forced"
        if label == "other":     return 0, 0, "llm_other_forced"
        # if the LLM call failed, drop to rules:
    # fallback when no API or failure
    return score_item_rules(answer, key, item_id)

# ---- Normalise and score ----------------------------------------------------
for col in crt2 + bcrt:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("").apply(normalise_answer)
    else:
        df[col] = ""

for col in crt2 + bcrt:
    key = scoring.get(col, None)
    correct_col   = f"{col}_correct"
    rtype_col     = f"{col}_resp_type"
    route_col     = f"{col}_route"
    # Ensure columns exist so we can assign in-place
    if correct_col not in df.columns: df[correct_col] = np.nan
    if rtype_col   not in df.columns: df[rtype_col]   = np.nan
    if route_col   not in df.columns: df[route_col]   = ""

    try:
        for i, a in tqdm(list(df[col].astype(str).items()), desc=f"Scoring {col}", unit="resp"):
            # Skip already-scored rows if resuming — unless we are forcing reclassification
            if not FORCE_RECLASS:
                if isinstance(df.at[i, route_col], str) and df.at[i, route_col].startswith((
                    "llm_","rule_","regex_","fuzzy_","exact_","phrase_","numeric_"
                )):
                    continue
            else:
                # clear any prior values so we actually recompute
                df.at[i, correct_col] = np.nan
                df.at[i, rtype_col]   = np.nan
                df.at[i, route_col]   = ""

            c, rt, route = score_item(a, key, col)
            df.at[i, correct_col] = c
            df.at[i, rtype_col]   = rt
            df.at[i, route_col]   = route
            # Periodic checkpoint
            if (i+1) % CKPT_EVERY == 0:
                df.to_csv(ckpt_path, index=False)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected — writing checkpoint and exiting cleanly...")
        df.to_csv(ckpt_path, index=False)
        # also write partial audit so far
        partial_audit_rows = []
        for itm in crt2 + bcrt:
            rcol = f"{itm}_route"
            if rcol in df.columns:
                counts = df[rcol].value_counts(dropna=False).to_dict()
                unresolved = int(counts.get("empty", 0) + counts.get("no_key", 0))
                partial_audit_rows.append({"item": itm, **counts, "unresolved": unresolved})
        pd.DataFrame(partial_audit_rows).fillna(0).to_csv(reports_dir / "tables" / "scoring_audit_partial.csv", index=False)
        raise

    # Save a checkpoint at the end of each item as well
    df.to_csv(ckpt_path, index=False)

# Totals
df["crt2_total"]      = df[[f"{c}_correct" for c in crt2]].sum(axis=1)
df["bcrt6_total"]     = df[[f"{c}_correct" for c in bcrt]].sum(axis=1)
df["combined10_total"]= df["crt2_total"] + df["bcrt6_total"]

# NFC
nfc_cols = [c for c in df.columns if re.fullmatch(r"nfc_\d+", c)]
def parse_nfc(x):
    s = str(x); m = re.match(r"^\s*(\d+)", s)
    return int(m.group(1)) if m else np.nan
df["NFC_total"] = df[nfc_cols].applymap(parse_nfc).sum(axis=1, min_count=1) if nfc_cols else np.nan

# Writing text passthrough
for wt in [writing_q1, writing_q2]:
    if wt not in df.columns: df[wt] = ""

# Save person-wide
person_cols = ["pid","seen_count","crt2_total","bcrt6_total","combined10_total","NFC_total",
               writing_q1, writing_q2] + nfc_cols
df[person_cols].to_csv(processed_dir / "person_wide.csv", index=False)

# Long-form + audit
rows = []
fam_map = cfg["familiarity_map"]
for _, row in df.iterrows():
    for item in crt2 + bcrt:
        seen_col = next((k for k,v in fam_map.items() if v==item), None)
        rows.append({
            "pid": row["pid"],
            "item_id": item,
            "family": "CRT2" if item in crt2 else "BCRT",
            "answer": row[item],
            "correct": row[f"{item}_correct"],
            "resp_type": row[f"{item}_resp_type"],
            "route": row[f"{item}_route"],
            "seen": int(row.get(seen_col, 0)) if seen_col in df.columns else 0
        })
item_long = pd.DataFrame(rows)
item_long.to_csv(interim_dir / "scored_long.csv", index=False)

# Scoring audit (unresolved = only empty + no_key)
audit_rows = []
for item in crt2 + bcrt:
    counts = df[f"{item}_route"].value_counts(dropna=False).to_dict()
    unresolved = int(counts.get("empty", 0) + counts.get("no_key", 0))
    audit_rows.append({"item": item, **counts, "unresolved": unresolved})
pd.DataFrame(audit_rows).fillna(0).sort_values("item").to_csv(
    reports_dir / "tables" / "scoring_audit.csv", index=False
)

print(f"LLM {'ON' if use_llm else 'OFF'} (model={model_name}). Saved person_wide.csv, scored_long.csv, scoring_audit.csv. Force reclass={'ON' if FORCE_RECLASS else 'OFF'}")
try:
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Removed checkpoint {ckpt_path}")
except Exception:
    pass