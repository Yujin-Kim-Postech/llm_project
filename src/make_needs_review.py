import json
from pathlib import Path

review_path = Path("labels/review_queue.jsonl")
labels_path = Path("labels/paper_labels.jsonl")
out_path = Path("labels/needs_review.jsonl")

def read_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

paper_labels = read_jsonl(labels_path)
review_rows = read_jsonl(review_path)

# paper_labels에 "이미 분류 완료"로 간주할 paper_id 집합 만들기
# 정책:
# - GENERAL_ECONOMICS: topic_l0만 있으면 OK (L1 없어도 됨)
# - INSURANCE_RISK: topic_l1이 있어야 OK (없으면 아직 미완)
done = set()

for r in paper_labels:
    pid = (r.get("paper_id") or "").strip()
    if not pid:
        continue
    l0 = (r.get("topic_l0") or "").strip()
    l1 = (r.get("topic_l1") or "").strip()

    if l0 == "GENERAL_ECONOMICS":
        done.add(pid)
    elif l0 == "INSURANCE_RISK":
        if l1:  # 보험은 L1까지 있어야 완료
            done.add(pid)

# needs_review: paper_labels에서 아직 완료(done)가 아닌 것만 남기기
needs = []
for r in review_rows:
    pid = (r.get("paper_id") or "").strip()
    if not pid:
        continue
    if pid in done:
        continue
    needs.append(r)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for r in needs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {out_path} n={len(needs)} (done={len(done)} / review_total={len(review_rows)})")
