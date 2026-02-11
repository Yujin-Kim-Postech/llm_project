# src/weak_label.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

from src import ontology as ont  # ✅ 단일 진입점: fallback 제거


# ---------------------------
# Paper → text (abstract + keywords only)
# ---------------------------
def paper_text(p: Dict[str, Any]) -> str:
    abstract = (p.get("abstract") or "")
    keywords = " ".join(p.get("keywords") or [])
    return f"{abstract}\n{keywords}".strip()


# ---------------------------
# Regex scoring
# ---------------------------
def _search(pat: str, text: str) -> bool:
    try:
        return re.search(pat, text, flags=re.IGNORECASE) is not None
    except re.error:
        return False


def score_rules(
    text: str,
    rules: Dict[str, List[Any]],
    gates: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    rules: {label: [regex | (regex, weight), ...]}
    gates: {label: {"anchor": regex, "blocked": [regex, ...]}}
           If anchor does NOT match, any pattern exactly equal to one of "blocked" is skipped.
    """
    scores: Dict[str, int] = {}
    evidence: Dict[str, List[str]] = {}

    gates = gates or {}

    for label, pats in (rules or {}).items():
        scores[label] = 0
        evidence[label] = []
        if not pats:
            continue

        gate = gates.get(label)
        has_anchor = True
        blocked_list: List[str] = []

        if gate:
            anchor_pat = str(gate.get("anchor", "") or "")
            blocked_list = list(gate.get("blocked", []) or [])
            has_anchor = _search(anchor_pat, text) if anchor_pat else True

        for item in pats:
            if isinstance(item, str):
                pat, w = item, 1
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                pat = str(item[0])
                w = item[1] if len(item) >= 2 and isinstance(item[1], (int, float)) else 1
            else:
                continue

            # gate: anchor 없으면 blocked 패턴은 스킵
            if gate and (not has_anchor) and any(pat == b for b in blocked_list):
                continue

            if _search(pat, text):
                scores[label] += int(w)
                evidence[label].append(pat)

    return scores, evidence


def topk(scores: Dict[str, int], k: int = 3) -> List[Tuple[str, int]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ---------------------------
# Public API
# ---------------------------
def recommend_l0(paper: Dict[str, Any], k: int = 2) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Recommend L0 labels using ontology.yaml (L0_RULES). If not defined, returns empty.
    """
    text = paper_text(paper)
    rules: Dict[str, List[Any]] = getattr(ont, "L0_RULES", {}) or {}
    if not rules:
        return [], {}
    scores, evidence = score_rules(text, rules, gates=None)
    return topk(scores, k), evidence


def recommend_l1(
    paper: Dict[str, Any],
    k: int = 3,
    l0: Optional[str] = None,
) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Recommend L1 labels.
    If l0 is provided and L0_TO_L1 exists, restrict L1 candidates to that mapping.
    """
    text = paper_text(paper)

    L1_LIST = getattr(ont, "L1_LIST", []) or []
    L2_RULES = getattr(ont, "L2_RULES", {}) or {}
    L0_TO_L1 = getattr(ont, "L0_TO_L1", {}) or {}
    GATE_RULES = getattr(ont, "GATE_RULES", {}) or {}

    if l0 and L0_TO_L1.get(l0):
        l1_candidates = L0_TO_L1.get(l0) or []
    else:
        l1_candidates = L1_LIST or list(L2_RULES.keys())

    scores: Dict[str, int] = {}
    evidence_out: Dict[str, List[str]] = {}

    for l1 in l1_candidates:
        l1_rules_block = (L2_RULES.get(l1, {}) or {})  # {l2: [regex...]}
        l2_scores, l2_evidence = score_rules(text, l1_rules_block, gates=GATE_RULES)

        best_l2 = max(l2_scores.values()) if l2_scores else 0
        scores[l1] = best_l2

        fired: List[str] = []
        for _l2, ev in (l2_evidence or {}).items():
            fired.extend(ev)
        evidence_out[l1] = fired

    return topk(scores, k), evidence_out


def recommend_l2(
    paper: Dict[str, Any],
    l1: str,
    k: int = 3,
) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Recommend L2 labels within a given L1.
    """
    text = paper_text(paper)

    L2_MAP = getattr(ont, "L2_MAP", {}) or {}
    L2_RULES = getattr(ont, "L2_RULES", {}) or {}
    GATE_RULES = getattr(ont, "GATE_RULES", {}) or {}

    candidates = (L2_MAP.get(l1, []) or [])
    l1_rules_block = (L2_RULES.get(l1, {}) or {})

    rules_for_scoring: Dict[str, List[Any]] = {}
    for l2 in candidates:
        rules_for_scoring[l2] = l1_rules_block.get(l2, []) or []

    scores, evidence = score_rules(text, rules_for_scoring, gates=GATE_RULES)
    return topk(scores, k), evidence
