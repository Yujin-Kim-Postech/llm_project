# src/weak_label.py
from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ---------------------------
# Ontology loading (runtime)
# ---------------------------
# Supports both "from src.ontology" and "from ontology" depending on execution context.
try:
    from src.ontology import L1_LIST, L2_MAP, L2_RULES
except Exception:  # pragma: no cover
    from ontology import L1_LIST, L2_MAP, L2_RULES


# ---------------------------
# Paper → text
# ---------------------------
def paper_text(paper: dict) -> str:
    """
    Build a single text blob for weak labeling.
    Compatible with both older (JORI) and newer (QJE) schema variants.
    """
    md = paper.get("metadata", {}) or {}
    raw = paper.get("raw_text", {}) or {}
    emp = paper.get("empirical_analysis") or {}
    theo = paper.get("theoretical_summary") or ""
    study_type = paper.get("study_type") or ""

    # keywords: old/new compatible
    kw_author = md.get("keywords_author") or md.get("keywords") or []
    if isinstance(kw_author, str):
        kw_author = [kw_author]
    kw_text = raw.get("keywords_text") or md.get("keywords_text") or ""

    # empirical fields (new schema)
    emp_subject = emp.get("subject") or []
    emp_keywords = emp.get("keywords") or []
    if isinstance(emp_subject, str):
        emp_subject = [emp_subject]
    if isinstance(emp_keywords, str):
        emp_keywords = [emp_keywords]

    parts = [
        md.get("title", ""),
        raw.get("abstract", ""),
        " ".join([str(x) for x in kw_author if str(x).strip()]),
        kw_text,
        study_type,
        theo,
        " ".join([str(x) for x in emp_subject if str(x).strip()]),
        " ".join([str(x) for x in emp_keywords if str(x).strip()]),
    ]
    return "\n".join([p for p in parts if p])


# ---------------------------
# Scoring utilities
# ---------------------------
def score_rules(text: str, rules: dict) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    rules: {label: [pattern OR [pattern, weight] OR (pattern, weight), ...]}

    Returns:
        scores: dict[label] = int
        evidence: dict[label] = list of regex patterns fired (string)
    """
    scores: Dict[str, int] = {}
    evidence: Dict[str, List[str]] = {}

    for label, pats in (rules or {}).items():
        scores[label] = 0
        evidence[label] = []

        if not pats:
            continue

        for item in pats:
            # item can be:
            # - "regex"
            # - ["regex", weight]
            # - ("regex", weight)
            if isinstance(item, str):
                pat, w = item, 1
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                pat = item[0]
                w = item[1] if len(item) >= 2 and isinstance(item[1], (int, float)) else 1
            else:
                continue

            try:
                if re.search(pat, text, flags=re.IGNORECASE):
                    scores[label] += int(w)
                    evidence[label].append(str(pat))
            except re.error:
                # bad regex should not crash the pipeline
                continue

    return scores, evidence


def topk(scores: Dict[str, int], k: int = 3) -> List[Tuple[str, int]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ---------------------------
# L1 rules derived from ontology
# ---------------------------
def build_l1_rules_from_l2_rules(l2_rules: dict) -> dict:
    """
    Build L1 scoring rules automatically from ontology's L2_RULES.

    Why: you want 'ontology.yaml 수정 → weaklabel 자동 반영'.
    We simply union all L2 regex under each L1 into that L1's rule list.

    Note: This is intentionally simple/robust for an empirical-only 2-depth tree.
    If L1 separation becomes weak, strengthen by adding more distinctive patterns
    inside ontology.yaml's L2_RULES for each L1.
    """
    out = {}
    for l1, l2_block in (l2_rules or {}).items():
        pats = []
        if isinstance(l2_block, dict):
            for _l2, regex_list in l2_block.items():
                for r in (regex_list or []):
                    # weight=1 default
                    pats.append((r, 1))
        out[l1] = pats
    return out


RULES_L1 = build_l1_rules_from_l2_rules(L2_RULES)


# ---------------------------
# Public API (used by review_queue.py)
# ---------------------------
def recommend_l1(paper: dict, k: int = 3) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Returns:
        topk_list: [(l1_label, score), ...]
        evidence:  {l1_label: [regex_fired, ...], ...}
    """
    text = paper_text(paper)

    # Ensure we only score labels present in ontology L1_LIST (order/whitelist)
    rules = {l1: RULES_L1.get(l1, []) for l1 in (L1_LIST or list(RULES_L1.keys()))}
    scores, evidence = score_rules(text, rules)
    return topk(scores, k), evidence


def recommend_l2(paper: dict, l1: str, k: int = 3) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Recommend L2 labels within a given L1 based on ontology.yaml.

    Returns:
        l2_topk: [(l2_label, score), ...]
        evidence_l2: {l2_label: [regex_fired, ...], ...}
    """
    text = paper_text(paper)

    # Candidate L2 list is defined by ontology structure
    candidates = L2_MAP.get(l1, []) or []

    # Scoring rules for those candidates
    l1_rules_block = L2_RULES.get(l1, {}) or {}

    # Build {l2: rules} dict limited to candidates
    rules_for_scoring = {}
    for l2 in candidates:
        rules_for_scoring[l2] = l1_rules_block.get(l2, []) or []

    scores, evidence = score_rules(text, rules_for_scoring)
    return topk(scores, k), evidence
