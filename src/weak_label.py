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

def score_rules(text: str, rules: Dict[str, List[Tuple[str,int]]]):
    scores = defaultdict(int)
    evidence = defaultdict(list)
    for label, pats in rules.items():
        for pat, w in pats:
            if re.search(pat, text, flags=re.IGNORECASE):
                scores[label] += w
                evidence[label].append(pat)
    return dict(scores), dict(evidence)

def topk(scores: Dict[str,int], k: int = 3):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

def recommend_l1(paper: dict, k: int = 3):
    text = paper_text(paper)
    scores, evidence = score_rules(text, RULES_L1)
    return topk(scores, k), evidence
