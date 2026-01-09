# src/review_queue.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.weak_label import recommend_l1, recommend_l2

ROOT = Path(__file__).resolve().parents[1]
PAPERS_PATH = ROOT / "data" / "papers.jsonl"
OUT_PATH = ROOT / "labels" / "review_queue.jsonl"


# ---------------------------
# Semi-auto policy (tweak!)
# ---------------------------
AUTO_L1_MIN_SCORE = 4
AUTO_L1_MIN_GAP = 2

AUTO_L2_MIN_SCORE = 4
AUTO_L2_MIN_GAP = 2

AUTO_FILL_TAGS = True


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


def safe_tag_heuristics(title: str) -> List[str]:
    t = (title or "").lower()
    tags = []

    # climate / nat-cat
    if "natural disaster" in t or "catastrophe" in t or "hurricane" in t or "flood" in t:
        tags += ["natural-disaster"]
    if "climate" in t or "natural disaster" in t or "catastrophe" in t:
        tags += ["climate-risk"]

    # cyber
    if "cyber" in t:
        tags += ["cyber-risk"]

    # AI / ML
    if "ai" in t or "machine learning" in t or "intelligent" in t or "risk profiling" in t:
        tags += ["ai"]

    # LTC / pension
    if "long-term care" in t or "long‐term care" in t or "ltc" in t:
        tags += ["long-term-care"]
    if "pension" in t or "annuity" in t or "retirement" in t:
        tags += ["pension"]

    out = []
    for x in tags:
        if x not in out:
            out.append(x)
    return out


def decide_auto(topk_list: List[Tuple[str, int]], min_score: int, min_gap: int) -> Tuple[str, str]:
    """
    Return (final_label, reason) where final_label="" if not confident.
    """
    if not topk_list:
        return "", "no_candidates"

    top1_label, top1_score = topk_list[0]
    top2_score = topk_list[1][1] if len(topk_list) > 1 else 0
    gap = top1_score - top2_score

    if top1_score >= min_score and gap >= min_gap:
        return top1_label, f"auto(score={top1_score},gap={gap})"
    return "", f"manual_needed(score={top1_score},gap={gap})"


def main() -> None:
    papers = read_jsonl(PAPERS_PATH)
    out_rows = []

    for p in papers:
        paper_id = p.get("paper_id", "")
        title = (p.get("metadata", {}) or {}).get("title", "")

        # L1
        l1_top3, evidence_l1 = recommend_l1(p)
        final_l1, l1_reason = decide_auto(l1_top3, AUTO_L1_MIN_SCORE, AUTO_L1_MIN_GAP)

        # L2 (only if L1 is confident)
        l2_top3 = []
        evidence_l2 = {}
        final_l2 = ""
        l2_reason = "skipped(no_final_l1)"

        if final_l1:
            l2_top3, evidence_l2 = recommend_l2(p, final_l1)
            final_l2, l2_reason = decide_auto(l2_top3, AUTO_L2_MIN_SCORE, AUTO_L2_MIN_GAP)

        tags = safe_tag_heuristics(title) if AUTO_FILL_TAGS else []

        out_rows.append({
            "paper_id": paper_id,
            "title": title,

            "l1_top3": l1_top3,
            "evidence_l1": evidence_l1,

            "l2_top3": l2_top3,
            "evidence_l2": evidence_l2,

            # semi-auto outputs
            "final_l1": final_l1,
            "final_l2": final_l2,
            "tags": tags,

            "auto_meta": {
                "l1_policy": {"min_score": AUTO_L1_MIN_SCORE, "min_gap": AUTO_L1_MIN_GAP},
                "l2_policy": {"min_score": AUTO_L2_MIN_SCORE, "min_gap": AUTO_L2_MIN_GAP},
                "l1_reason": l1_reason,
                "l2_reason": l2_reason
            }
        })

    write_jsonl(OUT_PATH, out_rows)
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
