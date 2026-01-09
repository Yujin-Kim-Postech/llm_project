# src/ontology.py
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "ontology.yaml"

with ONTOLOGY_PATH.open("r", encoding="utf-8") as f:
    ONTOLOGY = yaml.safe_load(f)

L1_LIST = ONTOLOGY.get("L1", [])
L2_MAP = ONTOLOGY.get("L2", {})
L2_RULES = ONTOLOGY.get("L2_RULES", {})
