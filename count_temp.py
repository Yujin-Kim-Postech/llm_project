import json
from collections import Counter
import pandas as pd
from pathlib import Path

jsonl_path = Path("labels") / "review_queue.jsonl"  # <- 여기에 파일 경로 넣기 (예: "/mnt/data/papers.jsonl")

c = Counter()
rows = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        lab = obj.get("final_l0") or "(missing)"
        c[lab] += 1

df = pd.DataFrame(sorted(c.items(), key=lambda x: (-x[1], x[0])),
                    columns=["final_l0", "count"])

print(df.to_string(index=False))
print("TOTAL =", sum(c.values()))