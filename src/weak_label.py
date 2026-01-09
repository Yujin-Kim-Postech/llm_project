from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def paper_text(paper: dict) -> str:
    md = paper.get("metadata", {})
    raw = paper.get("raw_text", {})
    parts = [
        md.get("title", ""),
        raw.get("abstract", ""),
        " ".join(md.get("keywords_author", []) or []),
        raw.get("keywords_text", "")
    ]
    return "\n".join([p for p in parts if p])

# --- L1 scoring rules (starter set; tune over time) ---
RULES_L1 = {
  "Household Insurance Demand": [
      (r"\b(willingness to pay|WTP|take[- ]?up|purchase|demand)\b", 2),
      (r"\b(household|individual|consumer|retirement|annuity|pension|long[- ]?term care|LTC)\b", 2),
  ],
  "Corporate Risk Management": [
      (r"\b(firm|corporate|nonfinancial|enterprise)\b", 2),
      (r"\b(hedg(e|ing)|risk management|ERM|insurance use)\b", 2),
  ],
  "Insurer Behavior & Performance": [
      (r"\b(insurer|insurance company|property casualty|P\&C)\b", 2),
      (r"\b(capital|solvency|RBC|reinsurance|underwriting|reserving|portfolio)\b", 2),
  ],
  "Market / Regulation / Policy": [
      (r"\b(regulation|regulatory|policy|reform|mandate|Solvency)\b", 2),
      (r"\b(competition|market structure|entry|premium regulation)\b", 1),
  ],
  "Risk & Loss Modeling": [
      (r"\b(loss distribution|tail risk|extreme value|catastrophe model)\b", 2),
      (r"\b(pricing model|valuation|robust|ambiguity|risk measure|classification|machine learning|AI)\b", 2),
  ],
}

def score_rules(text: str, rules: dict):
    """
    rules: {label: [pattern OR [pattern, weight] OR (pattern, weight), ...]}
    Returns:
      scores: dict[label] = int
      evidence: dict[label] = list of matched patterns (string)
    """
    import re

    scores = {}
    evidence = {}

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
                    evidence[label].append(pat)
            except re.error:
                # bad regex should not crash the pipeline
                continue

    return scores, evidence


def topk(scores: Dict[str,int], k: int = 3):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

def recommend_l1(paper: dict, k: int = 3):
    text = paper_text(paper)
    scores, evidence = score_rules(text, RULES_L1)
    return topk(scores, k), evidence

def recommend_l2(paper: dict, l1: str):
    """
    Return (l2_top3, evidence_l2) given a fixed L1.
    l2_top3: list of [label, score]
    evidence_l2: dict[label] = list of regex patterns fired
    """
    from src.ontology import L2_RULES  # uses rules defined in ontology.py
    text = paper_text(paper)

    rules = L2_RULES.get(l1, {})
    scores, evidence = score_rules(text, rules)
    top3 = topk(scores, k=3)
    return top3, evidence

# add to src/weak_label.py

from src.ontology import L2_RULES

def recommend_l2(paper: dict, l1: str):
    """
    Recommend L2 labels within a given L1.
    Returns: (l2_top3, evidence_l2)
    """
    text = paper_text(paper)  # existing helper in your weak_label.py
    rules = L2_RULES.get(l1, {})
    scores, evidence = score_rules(text, rules)  # existing helper
    top3 = topk(scores, k=3)  # existing helper
    return top3, evidence
