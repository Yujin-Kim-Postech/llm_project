# src/commit_needs_review.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]

NEEDS_PATH = ROOT / "labels" / "needs_review.jsonl"
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
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def main() -> None:
    needs = read_jsonl(NEEDS_PATH)
    existing = read_jsonl(LABELS_PATH)

    by_id: Dict[str, Dict[str, Any]] = {r.get("paper_id", ""): r for r in existing if r.get("paper_id")}

    committed = 0
    still_needs = []

    for r in needs:
        paper_id = (r.get("paper_id") or "").strip()
        if not paper_id:
            still_needs.append(r)
            continue

        l0 = (r.get("final_l0") or "").strip() or "Unlabeled"
        l1 = (r.get("final_l1") or "").strip()  # 사람이 채워야 함 (보험이면 필수)
        l2 = (r.get("final_l2") or "").strip()
        tags = normalize_tags(r.get("tags"))

        # GENERAL_ECONOMICS는 leaf라 L1 없어도 커밋 가능
        if l0 == "GENERAL_ECONOMICS":
            out = {
                "paper_id": paper_id,
                "topic_l0": "GENERAL_ECONOMICS",
                "topic_l1": "",
                "topic_l2": "",
                "tags": tags,
            }
        else:
            # INSURANCE_RISK(및 기타)는 L1이 없으면 아직 미완 → needs_review에 남김
            if not l1:
                still_needs.append(r)
                continue

            out = {
                "paper_id": paper_id,
                "topic_l0": l0,
                "topic_l1": l1,
                "topic_l2": l2,
                "tags": tags,
            }

        # fill-only로 paper_labels 업데이트 (사람이 이미 넣은 값은 덮지 않음)
        if paper_id in by_id:
            cur = by_id[paper_id]
            if not (cur.get("topic_l0") or "").strip():
                cur["topic_l0"] = out["topic_l0"]
            if out["topic_l1"] and not (cur.get("topic_l1") or "").strip():
                cur["topic_l1"] = out["topic_l1"]
            if out["topic_l2"] and not (cur.get("topic_l2") or "").strip():
                cur["topic_l2"] = out["topic_l2"]

            if tags:
                cur_tags = normalize_tags(cur.get("tags"))
                cur["tags"] = sorted(set(cur_tags) | set(tags))
        else:
            by_id[paper_id] = out

        committed += 1

    merged = [by_id[k] for k in sorted(by_id.keys())]
    write_jsonl(LABELS_PATH, merged)

    # 처리된 건 needs_review에서 제거(남은 것만 재저장)
    write_jsonl(NEEDS_PATH, still_needs)

    print(f"Committed to paper_labels: {committed}")
    print(f"Remaining in needs_review: {len(still_needs)}")
    print(f"Wrote: {LABELS_PATH}")
    print(f"Updated: {NEEDS_PATH}")


if __name__ == "__main__":
    main()
