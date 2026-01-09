# src/commit_labels.py
# Merge reviewed labels (final_l1/final_l2/tags) from review_queue.jsonl
# into paper_labels.jsonl by paper_id.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "labels" / "review_queue.jsonl"
LABELS_PATH = ROOT / "labels" / "paper_labels.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_tags(tags) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    # allow comma-separated string
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def main() -> None:
    review_rows = read_jsonl(REVIEW_PATH)
    existing = read_jsonl(LABELS_PATH)

    # index existing labels by paper_id
    by_id: Dict[str, Dict[str, Any]] = {r.get("paper_id", ""): r for r in existing if r.get("paper_id")}

    updated = 0
    added = 0
    skipped = 0

    for r in review_rows:
        paper_id = r.get("paper_id")
        if not paper_id:
            skipped += 1
            continue

        final_l1 = (r.get("final_l1") or "").strip()
        final_l2 = (r.get("final_l2") or "").strip()
        tags = normalize_tags(r.get("tags"))

        # Only commit if at least L1 is decided.
        if not final_l1:
            skipped += 1
            continue

        out = {
            "paper_id": paper_id,
            "topic_l1": final_l1,
            "topic_l2": final_l2,
            "tags": tags,
        }

        if paper_id in by_id:
            cur = by_id[paper_id]

            # ✅ fill-only: 기존 값이 비어있을 때만 채운다 (사람 수정값 보호)
            if not (cur.get("topic_l1") or "").strip():
                cur["topic_l1"] = final_l1

            # topic_l2도 동일
            if final_l2 and not (cur.get("topic_l2") or "").strip():
                cur["topic_l2"] = final_l2

            # tags는 "합집합(중복제거)"
            if tags:
                cur_tags = normalize_tags(cur.get("tags"))
                cur["tags"] = sorted(set(cur_tags) | set(tags))

            updated += 1
        else:
            by_id[paper_id] = out
            added += 1


    # write back in stable order (sorted by paper_id)
    merged = [by_id[k] for k in sorted(by_id.keys())]
    write_jsonl(LABELS_PATH, merged)

    print(f"Wrote: {LABELS_PATH}")
    print(f"Added: {added}, Updated: {updated}, Skipped (no final_l1): {skipped}")


if __name__ == "__main__":
    main()

