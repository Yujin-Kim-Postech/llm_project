# src/ontology.py
from __future__ import annotations

from pathlib import Path
import yaml
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "ontology.yaml"

with ONTOLOGY_PATH.open("r", encoding="utf-8") as f:
    ONTOLOGY: Dict[str, Any] = yaml.safe_load(f) or {}

# ---------------------------
# L0 (new; optional)
# ---------------------------
L0_SCOPE: List[str] = ONTOLOGY.get("L0_scope", []) or ONTOLOGY.get("L0_SCOPE", []) or []

# L0_RULES is optional; if you don't define it in ontology.yaml, it stays empty.
L0_RULES: Dict[str, List[Any]] = ONTOLOGY.get("L0_RULES", {}) or {}

# L0_TO_L1 is optional; if not provided, we won't restrict L1 by L0.
# expected: {"INSURANCE_RISK": [...L1 labels...], "GENERAL_ECONOMICS": [...L1 labels...]}
L0_TO_L1: Dict[str, List[str]] = ONTOLOGY.get("L0_TO_L1", {}) or {}

# ---------------------------
# L1/L2 (existing)
# ---------------------------
L1_LIST: List[str] = ONTOLOGY.get("L1", []) or []
L2_MAP: Dict[str, List[str]] = ONTOLOGY.get("L2", {}) or {}
L2_RULES: Dict[str, Dict[str, List[Any]]] = ONTOLOGY.get("L2_RULES", {}) or {}

# ---------------------------
# Optional gates (optional)
# ---------------------------
# expected format:
# GATE_RULES:
#   "C3....":
#     anchor: "..."
#     blocked: ["...", "..."]
GATE_RULES: Dict[str, Dict[str, Any]] = ONTOLOGY.get("GATE_RULES", {}) or {}
