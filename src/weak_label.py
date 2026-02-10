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
def _flatten_text(x):
    """
    Recursively extract all string-like content from dict/list structures.
    """
    out = []
    if x is None:
        return out
    if isinstance(x, str):
        s = x.strip()
        if s:
            out.append(s)
        return out
    if isinstance(x, (int, float, bool)):
        out.append(str(x))
        return out
    if isinstance(x, list):
        for it in x:
            out.extend(_flatten_text(it))
        return out
    if isinstance(x, dict):
        for v in x.values():
            out.extend(_flatten_text(v))
        return out
    out.append(str(x))
    return out


def paper_text(paper: dict) -> str:
    """
    Build a single text blob for weak labeling.
    Now matches against the *entire* empirical/theoretical summary,
    not only title/abstract.
    """
    md = paper.get("metadata", {}) or {}
    raw = paper.get("raw_text", {}) or {}
    emp = paper.get("empirical_analysis") or {}
    theo = paper.get("theoretical_summary") or ""
    study_type = paper.get("study_type") or ""

    kw_author = md.get("keywords_author") or md.get("keywords") or []
    if isinstance(kw_author, str):
        kw_author = [kw_author]

    kw_text = raw.get("keywords_text") or md.get("keywords_text") or ""

    parts = []

    # ---- bibliographic ----
    parts.append(md.get("title", ""))
    parts.append(raw.get("abstract", ""))

    # ---- keywords ----
    parts.append(" ".join(str(x) for x in kw_author if str(x).strip()))
    parts.append(str(kw_text) if kw_text else "")

    # ---- study info ----
    parts.append(str(study_type))
    parts.append(str(theo) if theo else "")

    # ---- 핵심: empirical_analysis 전체 flatten ----
    if isinstance(emp, dict):
        parts.append(" ".join(_flatten_text(emp)))
    else:
        parts.append(str(emp))

    # normalize
    return "\n".join(p for p in parts if p).lower()



# ---------------------------
# Scoring utilities
# ---------------------------
# label -> (anchor_regex, blocked_regex_when_no_anchor)
GATES = {
    # C3: spread/issuance/trigger는 ILS/CAT bond 앵커 없으면 무시
    "C3. CAT bonds / ILS: issuance, spreads, triggers, basis risk": (
        r"\b(catastrophe bond(s)?|cat bond(s)?|ils\b|insurance[- ]linked securit(y|ies))\b",
        r"\b(spread(s)?|issuance|trigger(s)?)\b",
    ),
    # D1: capital 같은 단독 패턴은 원천 차단(ontology에서 지우는 게 1순위지만 방어로)
    "D1. Operational risk (loss events & loss data empirics)": (
        r"\b(operational risk|op risk|internal loss data|loss distribution approach|lda\b)\b",
        r"\b(capital)\b",
    ),
}

def score_rules(text: str, rules: dict) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    scores: Dict[str, int] = {}
    evidence: Dict[str, List[str]] = {}

    for label, pats in (rules or {}).items():
        scores[label] = 0
        evidence[label] = []

        if not pats:
            continue

        # --- Gate 준비: label에 gate가 있으면 anchor 존재 여부를 미리 계산 ---
        gate = GATES.get(label)
        if gate is not None:
            anchor_pat, blocked_list = gate
            try:
                has_anchor = re.search(anchor_pat, text, flags=re.IGNORECASE) is not None
            except re.error:
                has_anchor = False
        else:
            has_anchor = True
            blocked_list = []

        for item in pats:
            if isinstance(item, str):
                pat, w = item, 1
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                pat = item[0]
                w = item[1] if len(item) >= 2 and isinstance(item[1], (int, float)) else 1
            else:
                continue

            # ✅ Gate: anchor 없으면 "blocked rule"은 스킵
            if gate is not None and not has_anchor:
                # pat이 blocked_list 중 하나와 "동일(또는 포함)"이면 스킵
                # (너의 ontology는 pat 문자열이 그대로 들어오니, 이 방식이 제일 안전)
                if any(pat == b for b in blocked_list):
                    continue

            try:
                if re.search(pat, text, flags=re.IGNORECASE):
                    scores[label] += int(w)
                    evidence[label].append(str(pat))
            except re.error:
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
def recommend_l1(paper: dict, k: int = 3):
    text = paper_text(paper)
    scores = {}
    evidence = {}

    for l1 in (L1_LIST or list(L2_RULES.keys())):
        l1_rules_block = L2_RULES.get(l1, {}) or {}
        # l1_rules_block: {l2: [regex...]}
        l2_scores, l2_evidence = score_rules(text, l1_rules_block)

        # L1 점수: (추천) max, 또는 sum
        best_l2 = max(l2_scores.values()) if l2_scores else 0
        scores[l1] = best_l2

        # evidence는 L1 안에서 터진 모든 regex를 모으기 (혹은 best L2만)
        fired = []
        for l2, ev in l2_evidence.items():
            fired.extend(ev)
        evidence[l1] = fired

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
