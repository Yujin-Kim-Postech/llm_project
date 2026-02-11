# src/review_queue.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from src.weak_label import recommend_l0, recommend_l1, recommend_l2
from src.ontology import POLICIES

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_POLICY = {
    "l0_policy": {"min_score": 1, "min_gap": 1},  # ✅ 여기서 L0 min_score=1 적용
    "l1_policy": {"min_score": 4, "min_gap": 2},
    "l2_policy": {"min_score": 4, "min_gap": 2},
}

# Allow ontology.yaml to override policy if present
POLICY = {
    "l0_policy": dict(DEFAULT_POLICY["l0_policy"]),
    "l1_policy": dict(DEFAULT_POLICY["l1_policy"]),
    "l2_policy": dict(DEFAULT_POLICY["l2_policy"]),
}
if isinstance(POLICIES, dict):
    for k in ("l0_policy", "l1_policy", "l2_policy"):
        if isinstance(POLICIES.get(k), dict):
            POLICY[k].update(POLICIES[k])


AI_TAG_RE = re.compile(
    r"(?is)\bai\b|artificial intelligence|machine learning|deep learning|neural network|large language model|llm"
)


def safe_tag_heuristics(p: Mapping[str, Any]) -> List[str]:
    """
    Conservative tagger. Avoids false positives from 'ai' appearing inside words.
    """
    tags: List[str] = []
    title = str(p.get("title") or "")
    text = title

    raw = p.get("raw_text") or {}
    if isinstance(raw, dict):
        text += "\n" + str(raw.get("abstract") or "")
        text += "\n" + str(raw.get("keywords_text") or "")

    t = text.strip()
    if AI_TAG_RE.search(t):
        tags.append("ai")

    # keep your existing custom tags if you had any (example placeholders)
    # if re.search(r"(?is)\bcyber\b|\bsecurity\b", t):
    #     tags.append("cyber-risk")

    return tags


def _decision(top3: List[List[Any]] | List[tuple], min_score: int, min_gap: int) -> tuple[bool, int, int]:
    """
    Return (auto_ok, best_score, gap). Assumes top3 sorted.
    """
    if not top3:
        return (False, 0, 0)
    best = int(top3[0][1])
    second = int(top3[1][1]) if len(top3) > 1 else 0
    gap = best - second
    auto_ok = (best >= min_score) and (gap >= min_gap)
    return (auto_ok, best, gap)


def process_paper(p: Dict[str, Any]) -> Dict[str, Any]:
    # 1) L0
    r0 = recommend_l0(p)
    p["l0_top3"] = r0.top3
    p["evidence_l0"] = r0.evidence

    l0_ok, l0_best, l0_gap = _decision(p["l0_top3"], **POLICY["l0_policy"])
    if l0_ok:
        p["final_l0"] = p["l0_top3"][0][0]
        p["auto_meta"] = p.get("auto_meta") or {}
        p["auto_meta"]["l0_policy"] = POLICY["l0_policy"]
        p["auto_meta"]["l0_reason"] = f"auto(score={l0_best},gap={l0_gap})"
    else:
        p["final_l0"] = ""
        p["auto_meta"] = p.get("auto_meta") or {}
        p["auto_meta"]["l0_policy"] = POLICY["l0_policy"]
        p["auto_meta"]["l0_reason"] = f"manual_needed(score={l0_best},gap={l0_gap})"
        # still attach tags
        p["tags"] = list(dict.fromkeys((p.get("tags") or []) + safe_tag_heuristics(p)))
        # do not proceed to L1/L2 if L0 not fixed
        p["l1_top3"], p["evidence_l1"] = [], {}
        p["l2_top3"], p["evidence_l2"] = [], {}
        p["final_l1"], p["final_l2"] = "", ""
        p["auto_meta"]["l1_reason"] = "skipped(manual_l0_needed)"
        p["auto_meta"]["l2_reason"] = "skipped(no_final_l1)"
        return p

    # 2) L1
    r1 = recommend_l1(p, final_l0=p["final_l0"])
    p["l1_top3"] = r1.top3
    p["evidence_l1"] = r1.evidence

    l1_ok, l1_best, l1_gap = _decision(p["l1_top3"], **POLICY["l1_policy"])
    if l1_ok:
        p["final_l1"] = p["l1_top3"][0][0]
        p["auto_meta"]["l1_policy"] = POLICY["l1_policy"]
        p["auto_meta"]["l1_reason"] = f"auto(score={l1_best},gap={l1_gap})"
    else:
        p["final_l1"] = ""
        p["auto_meta"]["l1_policy"] = POLICY["l1_policy"]
        p["auto_meta"]["l1_reason"] = f"manual_needed(score={l1_best},gap={l1_gap})"

    # 3) L2 (only if final_l1 exists)
    if p["final_l1"]:
        r2 = recommend_l2(p, final_l1=p["final_l1"])
        p["l2_top3"] = r2.top3
        p["evidence_l2"] = r2.evidence

        l2_ok, l2_best, l2_gap = _decision(p["l2_top3"], **POLICY["l2_policy"])
        if l2_ok:
            p["final_l2"] = p["l2_top3"][0][0]
            p["auto_meta"]["l2_policy"] = POLICY["l2_policy"]
            p["auto_meta"]["l2_reason"] = f"auto(score={l2_best},gap={l2_gap})"
        else:
            p["final_l2"] = ""
            p["auto_meta"]["l2_policy"] = POLICY["l2_policy"]
            p["auto_meta"]["l2_reason"] = f"manual_needed(score={l2_best},gap={l2_gap})"
    else:
        p["l2_top3"], p["evidence_l2"] = [], {}
        p["final_l2"] = ""
        p["auto_meta"]["l2_reason"] = "skipped(no_final_l1)"

    # tags
    p["tags"] = list(dict.fromkeys((p.get("tags") or []) + safe_tag_heuristics(p)))
    return p


def run(in_jsonl: Path, out_jsonl: Path) -> None:
    out_lines: List[str] = []
    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            p2 = process_paper(p)
            out_lines.append(json.dumps(p2, ensure_ascii=False))
    out_jsonl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
