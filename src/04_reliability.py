
import yaml, pandas as pd, numpy as np
from pathlib import Path
from utils import kr20

cfg = yaml.safe_load(open("config/config.yaml"))
interim_dir = Path(cfg["paths"]["interim"])
processed_dir = Path(cfg["paths"]["processed"])
reports_tables = Path(cfg["paths"]["reports"]) / "tables"
reports_tables.mkdir(parents=True, exist_ok=True)

crt2 = cfg["columns"]["crt2_items"]
bcrt = cfg["columns"]["bcrt_items"]
all_items = crt2 + bcrt

item_long = pd.read_csv(interim_dir / "scored_long.csv")
pivot = item_long.pivot_table(index="pid", columns="item_id", values="correct", aggfunc="first").reindex(columns=all_items).fillna(0).astype(int)

def rel_block(items, label):
    block = pivot[items].copy()
    k = block.shape[1]
    alpha = kr20(block)
    return {"label": label, "k": k, "KR20": alpha}

rows = []
rows.append(rel_block(crt2, "CRT-2"))
rows.append(rel_block(bcrt, "BCRT-6"))
rows.append(rel_block(all_items, "Combined-10"))

# Drop-one deltas for BCRT items on Combined
base = pivot[all_items].copy()
base_alpha = kr20(base)
for item in bcrt:
    alt = base.drop(columns=[item])
    rows.append({"label": f"Combined-10 minus {item}", "k": alt.shape[1], "KR20": kr20(alt), "Delta_from_base": kr20(alt)-base_alpha})

out = pd.DataFrame(rows)
out.to_csv(reports_tables / "reliability.csv", index=False)
print("Saved reliability.csv")
