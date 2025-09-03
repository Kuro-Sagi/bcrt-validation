
import yaml, pandas as pd
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))
processed_dir = Path(cfg["paths"]["processed"])
interim_dir = Path(cfg["paths"]["interim"])

person = pd.read_csv(processed_dir / "person_wide.csv")

cutoff = cfg['analysis']['naive_only_cutoff_seen_count']
naive = person[person['seen_count'] == cutoff].copy()
full = person.copy()

naive.to_csv(processed_dir / "person_naive.csv", index=False)
full.to_csv(processed_dir / "person_full.csv", index=False)

print(f"Naive n={len(naive)}; Full n={len(full)}")
