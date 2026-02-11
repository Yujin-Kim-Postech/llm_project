# src/commit_labels.py
# Merge reviewed labels (final_l0/final_l1/final_l2/tags) from review_queue.jsonl
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
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def _norm(s) -> str:
    return (s or "").strip()


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

        # NEW: L0 scope
        final_l0 = _norm(r.get("final_l0"))  # "INSURANCE_RISK" or "GENERAL_ECONOMICS"
        final_l1 = _norm(r.get("final_l1"))
        final_l2 = _norm(r.get("final_l2"))
        tags = normalize_tags(r.get("tags"))

        # ✅ Commit rule:
        # - If final_l0 is decided, commit even if L1/L2 empty (esp. GENERAL_ECONOMICS).
        # - Backward compatibility: if review_queue doesn't have final_l0,
        #   require final_l1 as before.
        if final_l0:
            commit_ok = True
        else:
            commit_ok = bool(final_l1)

        if not commit_ok:
            skipped += 1
            continue

        out = {
            "paper_id": paper_id,
            # store L0 if present
            **({"topic_l0": final_l0} if final_l0 else {}),
            "topic_l1": final_l1,
            "topic_l2": final_l2,
            "tags": tags,
        }

        if paper_id in by_id:
            cur = by_id[paper_id]

            # ✅ fill-only (human edits protection)
            if final_l0 and not _norm(cur.get("topic_l0")):
                cur["topic_l0"] = final_l0

            if final_l1 and not _norm(cur.get("topic_l1")):
                cur["topic_l1"] = final_l1

            if final_l2 and not _norm(cur.get("topic_l2")):
                cur["topic_l2"] = final_l2

            # tags: union
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
    print(f"Added: {added}, Updated: {updated}, Skipped (no decision): {skipped}")


if __name__ == "__main__":
    main()
