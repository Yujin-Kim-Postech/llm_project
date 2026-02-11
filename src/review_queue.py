# src/review_queue.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from src.weak_label import recommend_l0, recommend_l1, recommend_l2
from src.ontology import POLICIES

ROOT = Path(__file__).resolve().parents[1]

# Default policy (authoritative here)
DEFAULT_POLICY = {
    "l0_policy": {"min_score": 1, "min_gap": 1},  # ✅ L0 min_score=1
    "l1_policy": {"min_score": 4, "min_gap": 2},
    "l2_policy": {"min_score": 4, "min_gap": 2},
}

# Policy merge behavior:
# - DEFAULT_POLICY wins over ontology.yaml (POLICIES), so your local tweak works immediately.
# - If you want ontology.yaml to win, swap the merge order below.
POLICY = {
    "l0_policy": dict(DEFAULT_POLICY["l0_policy"]),
    "l1_policy": dict(DEFAULT_POLICY["l1_policy"]),
    "l2_policy": dict(DEFAULT_POLICY["l2_policy"]),
}

if isinstance(POLICIES, dict):
    for k in ("l0_policy", "l1_policy", "l2_policy"):
        if isinstance(POLICIES.get(k), dict):
            # Only fill missing keys from ontology (DEFAULT has priority)
            for kk, vv in POLICIES[k].items():
                POLICY[k].setdefault(kk, vv)

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
    auto_ok = (best >= int(min_score)) and (gap >= int(min_gap))
    return (auto_ok, best, gap)


def process_paper(p: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure auto_meta exists early
    p["auto_meta"] = p.get("auto_meta") or {}

    # 1) L0 추천
    r0 = recommend_l0(p)
    p["l0_top3"] = r0.top3
    p["evidence_l0"] = r0.evidence

    # Store applied policy in output
    p["auto_meta"]["l0_policy"] = POLICY["l0_policy"]

    # ---- L0 fallback rule (INSURANCE_RISK only when strong; else GENERAL) ----
    if p["l0_top3"]:
        best_label = str(p["l0_top3"][0][0])
        best_score = int(p["l0_top3"][0][1])
        second_score = int(p["l0_top3"][1][1]) if len(p["l0_top3"]) > 1 else 0
        gap = best_score - second_score
    else:
        best_label, best_score, gap = "", 0, 0

    # Only enforce thresholding for INSURANCE_RISK.
    if best_label == "INSURANCE_RISK":
        min_score = int(POLICY["l0_policy"]["min_score"])
        min_gap = int(POLICY["l0_policy"]["min_gap"])
        if best_score >= min_score and gap >= min_gap:
            p["final_l0"] = "INSURANCE_RISK"
            p["auto_meta"]["l0_reason"] = f"auto_insurance(score={best_score},gap={gap})"
        else:
            p["final_l0"] = "GENERAL_ECONOMICS"
            p["auto_meta"]["l0_reason"] = f"fallback_general(weak_insurance score={best_score},gap={gap})"
    else:
        # Non-insurance -> default GENERAL
        p["final_l0"] = "GENERAL_ECONOMICS"
        p["auto_meta"]["l0_reason"] = f"fallback_general(best={best_label},score={best_score},gap={gap})"

    # tags always
    p["tags"] = list(dict.fromkeys((p.get("tags") or []) + safe_tag_heuristics(p)))

    # If L0 is GENERAL, skip L1/L2 entirely (insurance ontology tree not applicable)
    if p["final_l0"] == "GENERAL_ECONOMICS":
        p["l1_top3"], p["evidence_l1"] = [], {}
        p["l2_top3"], p["evidence_l2"] = [], {}
        p["final_l1"], p["final_l2"] = "", ""
        p["auto_meta"]["l1_policy"] = POLICY["l1_policy"]
        p["auto_meta"]["l2_policy"] = POLICY["l2_policy"]
        p["auto_meta"]["l1_reason"] = "skipped(non_insurance_l0)"
        p["auto_meta"]["l2_reason"] = "skipped(non_insurance_l0)"
        return p

    # 2) L1 (only for INSURANCE_RISK)
    p["auto_meta"]["l1_policy"] = POLICY["l1_policy"]
    r1 = recommend_l1(p, final_l0=p["final_l0"])
    p["l1_top3"] = r1.top3
    p["evidence_l1"] = r1.evidence

    l1_ok, l1_best, l1_gap = _decision(p["l1_top3"], **POLICY["l1_policy"])
    if l1_ok:
        p["final_l1"] = p["l1_top3"][0][0]
        p["auto_meta"]["l1_reason"] = f"auto(score={l1_best},gap={l1_gap})"
    else:
        p["final_l1"] = ""
        p["auto_meta"]["l1_reason"] = f"manual_needed(score={l1_best},gap={l1_gap})"

    # 3) L2 (only if final_l1 exists)
    p["auto_meta"]["l2_policy"] = POLICY["l2_policy"]
    if p["final_l1"]:
        r2 = recommend_l2(p, final_l1=p["final_l1"])
        p["l2_top3"] = r2.top3
        p["evidence_l2"] = r2.evidence

        l2_ok, l2_best, l2_gap = _decision(p["l2_top3"], **POLICY["l2_policy"])
        if l2_ok:
            p["final_l2"] = p["l2_top3"][0][0]
            p["auto_meta"]["l2_reason"] = f"auto(score={l2_best},gap={l2_gap})"
        else:
            p["final_l2"] = ""
            p["auto_meta"]["l2_reason"] = f"manual_needed(score={l2_best},gap={l2_gap})"
    else:
        p["l2_top3"], p["evidence_l2"] = [], {}
        p["final_l2"] = ""
        p["auto_meta"]["l2_reason"] = "skipped(no_final_l1)"

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
