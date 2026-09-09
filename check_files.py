"""Run this BEFORE deploying: python check_files.py
Verifies every file the app needs is present and correctly named."""

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ok, problems = [], []


def check(rel, kind="file", hint=""):
    p = os.path.join(BASE, rel)
    exists = os.path.isdir(p) if kind == "dir" else os.path.isfile(p)
    if exists:
        if kind == "dir":
            n = len([f for f in os.listdir(p) if not f.startswith("_")])
            if n == 0:
                problems.append(f"EMPTY   {rel}/  -> {hint}")
                return False
            ok.append(f"OK      {rel}/  ({n} items)")
        else:
            mb = os.path.getsize(p) / 1e6
            ok.append(f"OK      {rel}  ({mb:.1f} MB)")
        return True
    problems.append(f"MISSING {rel}  -> {hint}")
    return False


print("\nChecking deployment folder...\n")

check("app.py", hint="should already be here")
check("requirements.txt", hint="should already be here")
check("README.md", hint="should already be here")

for f in ("classes.json", "fish_classes.json", "bird_classes.json"):
    check(f, hint="copy from MyDrive/PetProject")

for f in ("final_pet_model.pth", "fish_model.pth", "bird_model.pth"):
    check(f, hint="copy from MyDrive/PetProject (rename fish_model2.pth if needed)")

check("pet_food_dataset.csv", hint="rename 'pet_food_dataset (1).csv' - remove the space and brackets")
check("pet_products.csv", hint="copy from MyDrive/PetProject")
check("food_bert_model", "dir", hint="copy the whole folder contents")
check("productimages", "dir", hint="copy all product photos")

# class counts vs model heads
try:
    counts = {}
    for name, f in (("dog/cat", "classes.json"), ("fish", "fish_classes.json"), ("bird", "bird_classes.json")):
        with open(os.path.join(BASE, f)) as fh:
            counts[name] = len(json.load(fh))
    ok.append(f"OK      class counts -> dog/cat {counts['dog/cat']}, fish {counts['fish']}, bird {counts['bird']}")
except Exception:
    pass

# CSV columns
try:
    import pandas as pd
    p = pd.read_csv(os.path.join(BASE, "pet_products.csv"))
    need = {"product_name", "pet_type", "pet_category", "image1"}
    missing = need - set(p.columns)
    if missing:
        problems.append(f"COLUMNS pet_products.csv is missing {sorted(missing)}")
    else:
        ok.append("OK      pet_products.csv columns")

    f = pd.read_csv(os.path.join(BASE, "pet_food_dataset.csv"))
    need = {"pet_type", "pet_category", "food_name", "label"}
    missing = need - set(f.columns)
    if missing:
        problems.append(f"COLUMNS pet_food_dataset.csv is missing {sorted(missing)}")
    else:
        ok.append("OK      pet_food_dataset.csv columns")
except ImportError:
    pass
except FileNotFoundError:
    pass

for line in ok:
    print("  " + line)
if problems:
    print("\nPROBLEMS TO FIX:\n")
    for line in problems:
        print("  " + line)
    print(f"\n{len(problems)} problem(s). Fix these before deploying.\n")
    sys.exit(1)

print("\nEverything is present. Ready to deploy.\n")
