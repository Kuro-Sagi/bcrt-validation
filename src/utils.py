
import re
import numpy as np
import pandas as pd

def normalise_answer(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # replace fancy quotes/dashes
    s = s.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    s = re.sub(r"\s+", " ", s)
    # map common words to numerals
    word2num = {
        "one": "1", "two": "2", "three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"
    }
    toks = s.split()
    toks = [word2num.get(t, t) for t in toks]
    return " ".join(toks)

def match_any(s: str, patterns) -> bool:
    s = normalise_answer(s)
    for p in patterns:
        if s == normalise_answer(p):
            return True
    return False

def match_regex(s: str, pattern: str) -> bool:
    s = normalise_answer(s)
    try:
        return bool(re.search(pattern, s))
    except re.error:
        return False

def kr20(scores: pd.DataFrame) -> float:
    """KR-20 for dichotomous items. scores = DataFrame of 0/1 across items."""
    if scores.shape[1] < 2:
        return np.nan
    k = scores.shape[1]
    p = scores.mean(axis=0)
    q = 1 - p
    var_total = scores.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    return (k/(k-1)) * (1 - (p.mul(q)).sum() / var_total)

def point_biserial(item, total_rest):
    return pd.Series(item).corr(pd.Series(total_rest))

def bootstrap_ci(data, func, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n_obs = len(data)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        vals.append(func(data[idx]))
    return np.quantile(vals, [0.025, 0.975])


from rapidfuzz import fuzz

def extract_first_number(s: str):
    s = normalise_answer(s)
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None

def fuzzy_equal(a: str, b: str, threshold: int = 90) -> bool:
    a_n = normalise_answer(a); b_n = normalise_answer(b)
    return fuzz.token_sort_ratio(a_n, b_n) >= threshold

def match_any_fuzzy(s: str, patterns, threshold: int = 90) -> bool:
    return any(fuzzy_equal(s, p, threshold) for p in (patterns or []))

def contains_any(s: str, phrases):
    s = normalise_answer(s)
    return any(normalise_answer(p) in s for p in (phrases or []))
