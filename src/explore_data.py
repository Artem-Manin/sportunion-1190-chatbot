# src/explore_data.py
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import pprint

DATA_DIR = Path("data")  # adjust if running from project root

def sample_structure(path, n=3):
    j = json.load(open(path, "r", encoding="utf-8"))
    print(f"Loaded {path}: top-level keys = {list(j.keys())}")
    # print a small sample of keys for nested objects
    def walk_sample(obj, depth=0, max_depth=2):
        if depth > max_depth or not isinstance(obj, dict): 
            return
        for k,v in list(obj.items())[:10]:
            print("  " * depth + f"- {k}: type={type(v).__name__}")
            if isinstance(v, dict):
                walk_sample(v, depth+1, max_depth)
            elif isinstance(v, list) and v:
                print("  " * (depth+1) + f"list len={len(v)}; first element keys={list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}")
    walk_sample(j)

def collect_field_counts(path, list_key):
    j = json.load(open(path, "r", encoding="utf-8"))
    items = j.get(list_key, [])
    counts = Counter()
    for item in items[:1000]:
        if isinstance(item, dict):
            counts.update(item.keys())
    print("Top fields in", list_key)
    pprint.pprint(counts.most_common(50))

if __name__ == "__main__":
    paths = [DATA_DIR / "season_2425.json", DATA_DIR / "season_2526.json"]
    for p in paths:
        print("\n=== SAMPLE STRUCTURE", p, "===\n")
        sample_structure(p)
        # try common nested lists
        try:
            collect_field_counts(p, "matches")
        except Exception:
            pass
