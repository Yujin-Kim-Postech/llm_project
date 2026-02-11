# src/weak_label.py
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------
# Robust imports (works for `python -m src.review_queue` and script runs)
# ---------------------------
try:
    from src.ontology import (
        paper_text,
        L0_SCOPE,
        L0_RULES,
        L0_TO_L1,
        L1_LIST,
        L2_MAP,
        L2_RULES,
    )
except Exception:
    # Script execution fallback: ensure repo root is on sys.path then retry
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.ontology import (  # type: ignore
        paper_text,
        L0_SCOPE,
        L0_RULES,
        L0_TO_L1,
        L1_LIST,
        L2_MAP,
        L2_RULES,
    )


# ---------------------------
# Optional gates (reduce false positives)
# - Keep this minimal; scoring is still 100% based on ontology rules.
# ---------------------------
# label -> (anchor_regex, blocked_regex_list_when_no_anchor)
GATES: Dict[str, Tuple[str, List[str]]] = {
    # Example: C3 should not fire on "spreads/issuance" unless ILS anchor exists
    "C3. CAT bonds / ILS: issuance, spreads, triggers, basis risk": (
        r"\b(catastrophe bond(s)?|cat bond(s)?|ils\b|insurance[- ]linked securit(y|ies)|insurance[- ]linked securities)\b",
        [
            r"\b(spread(s)?|issuance|trigger(s)?)\b",
        ],
    ),
}


def _compile_safe(pat: str):
    try:
        return re.compile(pat, flags=re.IGNORECASE)
    except re.error:
        return None


def score_rules(
    text: str, rules_for_scoring: Dict[str, List[str]]
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Score each label by counting matched regex patterns (weight=1 per pattern).
    Evidence records which regex fired.
    """
    scores: Dict[str, int] = {}
    evidence: Dict[str, List[str]] = {}

    for label, pats in (rules_for_scoring or {}).items():
        scores[label] = 0
        evidence[label] = []
        if not pats:
            continue

        gate = GATES.get(label)
        has_anchor = True
        blocked_list: List[str] = []

        if gate is not None:
            anchor_pat, blocked_list = gate
            anchor_re = _compile_safe(anchor_pat)
            has_anchor = bool(anchor_re and anchor_re.search(text))

        for pat in pats:
            # Gate: if no anchor, skip blocked patterns
            if gate is not None and not has_anchor:
                if any(pat == b for b in blocked_list):
                    continue

            cre = _compile_safe(pat)
            if not cre:
                continue
            if cre.search(text):
                scores[label] += 1
                evidence[label].append(pat)

    return scores, evidence


def topk(scores: Dict[str, int], k: int = 3) -> List[Tuple[str, int]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ---------------------------
# Public API
# ---------------------------
def recommend_l0(paper: Dict[str, Any], k: int = 2):
    """
    Recommend L0 by L0_RULES only.
    If L0_RULES is empty, this returns all zeros (manual step required).
    """
    text = paper_text(paper)
    rules_for_scoring: Dict[str, List[str]] = {}

    for l0 in (L0_SCOPE or list(L0_RULES.keys())):
        rules_for_scoring[l0] = L0_RULES.get(l0, []) or []

    scores, evidence = score_rules(text, rules_for_scoring)
    return topk(scores, k), evidence


def recommend_l1(paper: Dict[str, Any], l0: str | None = None, k: int = 3):
    """
    Recommend L1 by pooling (max over) its L2 scores.
    Candidate L1 list can be restricted by L0_TO_L1 if l0 is given.
    """
    text = paper_text(paper)

    # Candidate L1s
    if l0 and L0_TO_L1:
        candidates_l1 = L0_TO_L1.get(l0, []) or []
    else:
        candidates_l1 = L1_LIST or list(L2_RULES.keys())

    scores_l1: Dict[str, int] = {}
    evidence_l1: Dict[str, List[str]] = {}

    for l1 in candidates_l1:
        l1_block = L2_RULES.get(l1, {}) or {}  # {l2: [regex...]}
        l2_scores, l2_evidence = score_rules(text, l1_block)

        # L1 score = max L2 score within that L1
        best = max(l2_scores.values()) if l2_scores else 0
        scores_l1[l1] = best

        # Evidence = union of all fired patterns under this L1
        fired: List[str] = []
        for _l2, ev in (l2_evidence or {}).items():
            fired.extend(ev)
        evidence_l1[l1] = fired

    return topk(scores_l1, k), evidence_l1


def recommend_l2(
    paper: Dict[str, Any], l1: str, k: int = 3
) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Recommend L2 labels under a given L1.
    Candidates are from L2_MAP[l1]; rules are from L2_RULES[l1][l2].
    """
    text = paper_text(paper)

    candidates = L2_MAP.get(l1, []) or []
    l1_block = L2_RULES.get(l1, {}) or {}

    rules_for_scoring: Dict[str, List[str]] = {}
    for l2 in candidates:
        rules_for_scoring[l2] = l1_block.get(l2, []) or []

    scores, evidence = score_rules(text, rules_for_scoring)
    return topk(scores, k), evidence
