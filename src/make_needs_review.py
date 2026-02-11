import json
from pathlib import Path

src = Path("labels/review_queue.jsonl")
dst = Path("labels/needs_review.jsonl")

rows = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if (o.get("final_l1") or "").strip() == "":
            rows.append(o)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    for o in rows:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

print("Wrote", dst, "n=", len(rows))
