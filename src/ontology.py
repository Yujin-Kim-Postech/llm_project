# src/ontology.py
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "ontology.yaml"

with ONTOLOGY_PATH.open("r", encoding="utf-8") as f:
    ONTOLOGY = yaml.safe_load(f)

# --- raw ---
L0_SCOPE = ONTOLOGY.get("L0_scope", ["INSURANCE_RISK", "GENERAL_ECONOMICS"])
L1_LIST = ONTOLOGY.get("L1", [])
L2_MAP  = ONTOLOGY.get("L2", {})
L2_RULES = ONTOLOGY.get("L2_RULES", {})

# --- optional ---
L0_RULES = ONTOLOGY.get("L0_RULES", {})  # {scope: [regex...]}

# ✅ gate도 ontology.py로 이동 (weak_label에 남기지 않기)
GATES = {
    "C3. CAT bonds / ILS: issuance, spreads, triggers, basis risk": (
        r"\b(catastrophe bond(s)?|cat bond(s)?|ils\b|insurance[- ]linked securit(y|ies))\b",
        [r"\b(spread(s)?|issuance|trigger(s)?)\b"],
    ),
    "D1. Operational risk (loss events & loss data empirics)": (
        r"\b(operational risk|op risk|internal loss data|loss distribution approach|lda\b)\b",
        [r"\b(capital)\b"],
    ),
}
