"""
Verification script for Vision Transformer Hazard Assessment & Safety Reasoning.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from inference import ModelInferenceManager

def main():
    print("=" * 70)
    print("  VERIFYING VISION TRANSFORMER HAZARD ASSESSMENT ENGINE")
    print("=" * 70)

    mgr = ModelInferenceManager()

    test_imgs = [
        os.path.join(BASE_DIR, "data", "css-data", "test", "images", "004063_jpg.rf.1b7cdc4035bcb24ef69b8798b444053e.jpg"),
        os.path.join(BASE_DIR, "data", "css-data", "test", "images", "002551_jpg.rf.ce4b9f934161faa72c80dc6898d37b2d.jpg"),
        os.path.join(BASE_DIR, "data", "css-data", "test", "images", "004763_jpg.rf.46484e6ca73caeaa9de45822cf1085a9.jpg"),
    ]

    for idx, path in enumerate(test_imgs):
        if not os.path.exists(path):
            print(f"[WARN] Not found: {path}")
            continue
        print(f"\n--- Testing Sample {idx + 1}: {os.path.basename(path)} ---")
        with open(path, "rb") as f:
            img_bytes = f.read()

        res = mgr.assess_transformer_hazard(img_bytes)
        print(f"Status           : {res['status']}")
        print(f"Risk Score       : {res['risk_score']} ({res['risk_percentage']})")
        print(f"Severity         : {res['severity']}")
        print(f"Latency          : {res['latency_ms']} ms")
        print(f"Counts           : Workers={res['counts']['workers']}, Machinery={res['counts']['machinery']}, Violations={res['counts']['violations']}, Compliant={res['counts']['compliant']}")
        print(f"Total Detections : {res['counts']['total']}")
        for d in res['detections'][:5]:
            print(f"  * {d['class']} ({d['confidence']:.2f}) at {[int(c) for c in d['box']]}")
        print(f"Reasoning Preview: {res['reasoning'][:140]}...")

    # Also test GAN integration
    print("\n--- Testing GAN + Transformer Integration ---")
    gan_audit = mgr.audit_gan_transformer()
    print(f"GAN Audit Status : {gan_audit['status']}")
    print(f"Synthetic Source : {gan_audit.get('gan_source', 'N/A')}")
    print(f"Assessed Risk    : {gan_audit['risk_percentage']} ({gan_audit['severity']})")

if __name__ == "__main__":
    main()
