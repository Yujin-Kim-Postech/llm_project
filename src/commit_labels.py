# src/commit_labels.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


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
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def _best_and_gap(top3: Any) -> Tuple[Optional[str], int, int]:
    """
    top3 format example: [[label, score], [label2, score2], ...]
    returns (best_label, best_score, gap_to_2nd)
    """
    if not isinstance(top3, list) or not top3:
        return None, 0, 0
    try:
        best_label, best_score = top3[0][0], int(top3[0][1])
    except Exception:
        return None, 0, 0

    second_score = 0
    if len(top3) >= 2:
        try:
            second_score = int(top3[1][1])
        except Exception:
            second_score = 0

    return str(best_label), best_score, (best_score - second_score)


def _auto_pick_label(r: Dict[str, Any], level: str) -> Tuple[str, str]:
    """
    Try to auto-decide final_l{level} based on policy and top3.
    Returns (picked_label, reason). If cannot, picked_label="".
    """
    auto = r.get("auto_meta") or {}
    policy = auto.get(f"l{level}_policy") or {}
    min_score = int(policy.get("min_score", 999))
    min_gap = int(policy.get("min_gap", 999))

    top3 = r.get(f"l{level}_top3") or []
    best_label, best_score, gap = _best_and_gap(top3)

    if not best_label:
        return "", f"auto_fail(no_l{level}_top3)"

    if best_score >= min_score and gap >= min_gap:
        return best_label, f"auto_ok(score={best_score},gap={gap},min_score={min_score},min_gap={min_gap})"

    return "", f"auto_fail(score={best_score},gap={gap},min_score={min_score},min_gap={min_gap})"


def main() -> None:
    review_rows = read_jsonl(REVIEW_PATH)
    existing = read_jsonl(LABELS_PATH)

    by_id: Dict[str, Dict[str, Any]] = {r.get("paper_id", ""): r for r in existing if r.get("paper_id")}

    added = 0
    updated = 0
    skipped = 0
    auto_committed = 0
    manual_committed = 0

    for r in review_rows:
        paper_id = r.get("paper_id")
        if not paper_id:
            skipped += 1
            continue

        # 1) 사람이 확정했으면 그걸 우선 사용
        final_l1 = (r.get("final_l1") or "").strip()
        final_l2 = (r.get("final_l2") or "").strip()

        # 2) 사람이 안 했으면 정책기반 자동확정 시도
        auto_reason_l1 = ""
        auto_reason_l2 = ""

        if not final_l1:
            final_l1, auto_reason_l1 = _auto_pick_label(r, "1")

        if final_l1 and not final_l2:
            # L2는 L1이 확정된 뒤에만 자동확정 시도
            final_l2, auto_reason_l2 = _auto_pick_label(r, "2")

        tags = normalize_tags(r.get("tags"))

        # commit 조건: L1이 있어야 함 (사람이든 자동이든)
        if not final_l1:
            skipped += 1
            continue

        is_auto = (auto_reason_l1.startswith("auto_ok") and (not (r.get("final_l1") or "").strip()))
        if is_auto:
            auto_committed += 1
        else:
            manual_committed += 1

        out = {
            "paper_id": paper_id,
            "topic_l1": final_l1,
            "topic_l2": final_l2,
            "tags": tags,
        }

        if paper_id in by_id:
            cur = by_id[paper_id]

            # fill-only (사람 수정값 보호)
            if not (cur.get("topic_l1") or "").strip():
                cur["topic_l1"] = final_l1

            if final_l2 and not (cur.get("topic_l2") or "").strip():
                cur["topic_l2"] = final_l2

            if tags:
                cur_tags = normalize_tags(cur.get("tags"))
                cur["tags"] = sorted(set(cur_tags) | set(tags))

            updated += 1
        else:
            by_id[paper_id] = out
            added += 1

    merged = [by_id[k] for k in sorted(by_id.keys())]
    write_jsonl(LABELS_PATH, merged)

    print(f"Wrote: {LABELS_PATH}")
    print(f"Added: {added}, Updated: {updated}, Skipped(no L1): {skipped}")
    print(f"Committed(auto): {auto_committed}, Committed(manual): {manual_committed}")


if __name__ == "__main__":
    main()
