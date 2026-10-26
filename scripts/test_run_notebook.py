import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = os.path.join("notebooks", "05_Transformer_Architecture.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Change cwd to notebooks/ so it runs under the exact notebook working directory
os.chdir(os.path.abspath("notebooks"))
print(f"Testing notebook execution from: {os.getcwd()}")

global_scope = {
    "__name__": "__main__",
    "plt": plt
}

code_cell_num = 0
for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        code_cell_num += 1
        src = "".join(cell.get("source", []))
        print(f"\n--- Running Code Cell {code_cell_num} (JSON cell {idx + 1}) ---")
        try:
            exec(src, global_scope)
            print(f"[OK] Code Cell {code_cell_num} succeeded")
        except Exception as e:
            print(f"[ERROR] in Code Cell {code_cell_num}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

print("\n==============================================")
print("ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY WITH ZERO ERRORS!")
print("==============================================")
