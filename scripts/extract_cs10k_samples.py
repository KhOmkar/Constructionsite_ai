"""
Extracts diverse, high-impact demo scenes from ConstructionSite 10k parquet dataset.
Extracts examples of:
- Rule 1: PPE violation (Hardhat / Vest)
- Rule 2: Working at height / Fall hazard
- Rule 3: Deep excavation / Trench edge hazard
- Rule 4: Worker in excavator blindspot / swing radius
- Safe / Compliant scenes
"""

import os
import json
import pyarrow.parquet as pq
from PIL import Image
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "2_data")
OUT_DIR = os.path.join(BASE_DIR, "data", "samples", "cs10k")
os.makedirs(OUT_DIR, exist_ok=True)

parquet_path = os.path.join(DATA_DIR, "train-00001-of-00002.parquet")

def extract_samples(max_samples=24):
    print(f"Reading parquet from {parquet_path}...")
    table = pq.read_table(parquet_path)
    num_rows = table.num_rows
    print(f"Total rows in partition 1: {num_rows}")

    samples = []
    rule_counts = {1: 0, 2: 0, 3: 0, 4: 0, "safe": 0}

    for i in range(num_rows):
        row = table.slice(i, 1).to_pydict()
        
        img_id = row["image_id"][0]
        caption = row["image_caption"][0]
        r1 = row["rule_1_violation"][0]
        r2 = row["rule_2_violation"][0]
        r3 = row["rule_3_violation"][0]
        r4 = row["rule_4_violation"][0]
        excavator = row["excavator"][0]
        
        has_r1 = r1 is not None and len(r1.get("bounding_box", [])) > 0
        has_r2 = r2 is not None and len(r2.get("bounding_box", [])) > 0
        has_r3 = r3 is not None and len(r3.get("bounding_box", [])) > 0
        has_r4 = r4 is not None and len(r4.get("bounding_box", [])) > 0
        is_safe = not (has_r1 or has_r2 or has_r3 or has_r4)

        target = False
        rule_tag = "safe"
        if has_r4 and rule_counts[4] < 6:
            target = True
            rule_counts[4] += 1
            rule_tag = "Rule_4_Excavator_Blindspot"
        elif has_r1 and rule_counts[1] < 6:
            target = True
            rule_counts[1] += 1
            rule_tag = "Rule_1_PPE_Violation"
        elif has_r2 and rule_counts[2] < 6:
            target = True
            rule_counts[2] += 1
            rule_tag = "Rule_2_Height_Fall_Hazard"
        elif has_r3 and rule_counts[3] < 6:
            target = True
            rule_counts[3] += 1
            rule_tag = "Rule_3_Trench_Edge_Hazard"
        elif is_safe and rule_counts["safe"] < 6:
            target = True
            rule_counts["safe"] += 1
            rule_tag = "Safe_Compliant_Site"

        if target:
            img_bytes = row["image"][0]["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            img_fname = f"cs10k_{img_id}_{rule_tag}.jpg"
            img_fpath = os.path.join(OUT_DIR, img_fname)
            pil_img.save(img_fpath, quality=90)

            # Violation details
            violations = []
            if has_r1:
                violations.append({"rule": 1, "name": "PPE Violation", "reason": r1.get("reason"), "boxes": r1.get("bounding_box")})
            if has_r2:
                violations.append({"rule": 2, "name": "Fall from Height Hazard", "reason": r2.get("reason"), "boxes": r2.get("bounding_box")})
            if has_r3:
                violations.append({"rule": 3, "name": "Trench / Excavation Edge", "reason": r3.get("reason"), "boxes": r3.get("bounding_box")})
            if has_r4:
                violations.append({"rule": 4, "name": "Machinery Swing / Blindspot", "reason": r4.get("reason"), "boxes": r4.get("bounding_box")})

            meta = {
                "image_id": img_id,
                "filename": img_fname,
                "rule_tag": rule_tag,
                "caption": caption,
                "violations": violations,
                "excavator_boxes": excavator,
                "illumination": row.get("illumination", [None])[0],
                "camera_distance": row.get("camera_distance", [None])[0],
                "view": row.get("view", [None])[0]
            }
            samples.append(meta)

            if len(samples) >= max_samples:
                break

    meta_path = os.path.join(OUT_DIR, "cs10k_samples_index.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"\n[SUCCESS] Extracted {len(samples)} diverse scenes to: {OUT_DIR}")
    print(f"Rule distribution: {rule_counts}")
    return samples

if __name__ == "__main__":
    extract_samples()
