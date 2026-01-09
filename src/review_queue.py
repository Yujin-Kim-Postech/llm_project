from __future__ import annotations
import json
from pathlib import Path
from src.weak_label import recommend_l1

ROOT = Path(__file__).resolve().parents[1]

def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    papers_path = ROOT / "data" / "papers.jsonl"
    out_path = ROOT / "labels" / "review_queue.jsonl"

    rows = []
    for paper in iter_jsonl(papers_path):
        top3, evidence = recommend_l1(paper, k=3)
        rows.append({
            "paper_id": paper["paper_id"],
            "title": paper.get("metadata", {}).get("title", ""),
            "l1_top3": top3,
            "evidence_l1": evidence,
            "final_l1": "",
            "final_l2": "",
            "tags": []
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
