import os
import json

idx_path = os.path.join("data", "samples", "cs10k", "cs10k_samples_index.json")
with open(idx_path, "r", encoding="utf-8") as f:
    items = json.load(f)

for i, it in enumerate(items):
    viol = it["violations"][0]["name"] if it.get("violations") else "Compliant / Safe Site"
    rule = it.get("rule_tag")
    fn = it.get("filename")
    print(f"[{i:02d}] {rule:30s} | {viol:28s} | {fn}")
