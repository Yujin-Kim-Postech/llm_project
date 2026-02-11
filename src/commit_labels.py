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


def _auto_pick_top1(r: Dict[str, Any], level: str, min_score: int = 1) -> Tuple[str, str]:
    """
    Aggressive auto-pick:
      - If l{level}_top3 exists, pick top1 as long as its score >= min_score.
      - If score < min_score, return "" (leave for manual).
    """
    top3 = r.get(f"l{level}_top3") or []
    best_label, best_score, gap = _best_and_gap(top3)
    if not best_label:
        return "", f"auto_fail(no_l{level}_top3)"
    if best_score >= min_score:
        return best_label, f"auto_ok_top1(score={best_score},gap={gap},min_score={min_score})"
    return "", f"auto_fail_low(score={best_score},min_score={min_score})"


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

        # --- L0 is always used for tree root-level ---
        final_l0 = (r.get("final_l0") or "").strip() or "Unlabeled"

        # 사람이 확정했으면 우선
        final_l1 = (r.get("final_l1") or "").strip()
        final_l2 = (r.get("final_l2") or "").strip()
        tags = normalize_tags(r.get("tags"))

        auto_reason_l1 = ""
        auto_reason_l2 = ""

        # --- Policy: GENERAL_ECONOMICS stays at L0 (leaf), so L1/L2 are optional ---
        if final_l0 == "GENERAL_ECONOMICS":
            # We still allow tags (optional). Keep L1/L2 empty.
            out = {
                "paper_id": paper_id,
                "topic_l0": final_l0,
                "topic_l1": "",
                "topic_l2": "",
                "tags": tags,
            }

            if paper_id in by_id:
                cur = by_id[paper_id]
                if not (cur.get("topic_l0") or "").strip():
                    cur["topic_l0"] = final_l0
                # do not fill L1/L2 for GENERAL_ECONOMICS
                if tags:
                    cur_tags = normalize_tags(cur.get("tags"))
                    cur["tags"] = sorted(set(cur_tags) | set(tags))
                updated += 1
            else:
                by_id[paper_id] = out
                added += 1

            # count as committed (manual if user set final_l1; otherwise auto-ish)
            if (r.get("final_l1") or "").strip():
                manual_committed += 1
            else:
                auto_committed += 1
            continue

        # --- INSURANCE_RISK path: must have L1 for tree expansion (but we will auto-pick top1) ---
        if not final_l1:
            # aggressive: pick top1 even if score <4; require score>=1 to avoid pure noise
            final_l1, auto_reason_l1 = _auto_pick_top1(r, "1", min_score=1)

        if final_l1 and not final_l2:
            # L2 auto-pick top1 too (score>=1)
            final_l2, auto_reason_l2 = _auto_pick_top1(r, "2", min_score=1)

        # If still no L1, leave for manual review
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
            "topic_l0": final_l0,
            "topic_l1": final_l1,
            "topic_l2": final_l2,
            "tags": tags,
        }

        if paper_id in by_id:
            cur = by_id[paper_id]

            # fill-only (사람 수정값 보호)
            if not (cur.get("topic_l0") or "").strip():
                cur["topic_l0"] = final_l0

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
    print(f"Added: {added}, Updated: {updated}, Skipped(no INSURANCE L1): {skipped}")
    print(f"Committed(auto): {auto_committed}, Committed(manual): {manual_committed}")


if __name__ == "__main__":
    main()
