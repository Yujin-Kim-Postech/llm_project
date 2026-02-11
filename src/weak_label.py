# src/weak_label.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple
import re

from src.ontology import (
    paper_text,
    L0_RULES,
    L1_RULES,
    L2_RULES,
    L0_TO_L1,
    L1_TO_L2,
)


@dataclass
class RecommendResult:
    top3: List[Tuple[str, int]]
    evidence: Dict[str, List[str]]


def _score_text(text: str, rules: Dict[str, List[str]]) -> RecommendResult:
    """
    Score each label by counting how many unique regex patterns matched (>=1 occurrence).
    Evidence stores the regex patterns that matched.
    """
    t = text or ""
    top: List[Tuple[str, int]] = []
    evidence: Dict[str, List[str]] = {}

    for label, pats in rules.items():
        matched: List[str] = []
        seen = set()
        for pat in pats:
            if not pat or pat in seen:
                continue
            seen.add(pat)
            try:
                if re.search(pat, t):
                    matched.append(pat)
            except re.error:
                # ignore invalid regex to avoid crashing queue
                continue
        score = len(matched)
        top.append((label, score))
        evidence[label] = matched

    top.sort(key=lambda x: (-x[1], x[0]))
    return RecommendResult(top3=top[:3], evidence=evidence)


def recommend_l0(p: Mapping[str, Any]) -> RecommendResult:
    txt = paper_text(p)
    return _score_text(txt, L0_RULES)


def recommend_l1(p: Mapping[str, Any], final_l0: str | None = None) -> RecommendResult:
    """
    If ontology provides L0_TO_L1, optionally restrict candidate labels.
    If L1_RULES is empty, returns empty outputs safely.
    """
    if not L1_RULES:
        return RecommendResult(top3=[], evidence={})

    txt = paper_text(p)
    res = _score_text(txt, L1_RULES)

    if final_l0 and final_l0 in L0_TO_L1 and L0_TO_L1[final_l0]:
        allowed = set(L0_TO_L1[final_l0])
        res = RecommendResult(
            top3=[x for x in res.top3 if x[0] in allowed],
            evidence={k: v for k, v in res.evidence.items() if k in allowed},
        )
    return res


def recommend_l2(p: Mapping[str, Any], final_l1: str | None = None) -> RecommendResult:
    """
    If ontology provides L1_TO_L2, optionally restrict candidate labels.
    If L2_RULES is empty, returns empty outputs safely.
    """
    if not L2_RULES:
        return RecommendResult(top3=[], evidence={})

    txt = paper_text(p)
    res = _score_text(txt, L2_RULES)

    if final_l1 and final_l1 in L1_TO_L2 and L1_TO_L2[final_l1]:
        allowed = set(L1_TO_L2[final_l1])
        res = RecommendResult(
            top3=[x for x in res.top3 if x[0] in allowed],
            evidence={k: v for k, v in res.evidence.items() if k in allowed},
        )
    return res
