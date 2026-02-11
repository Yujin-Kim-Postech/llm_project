# src/ontology.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import yaml

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "ontology.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


ONTOLOGY: Dict[str, Any] = _load_yaml(ONTOLOGY_PATH)

# ---------------------------
# Ontology fields (safe getters)
# ---------------------------
L0_SCOPE: List[str] = ONTOLOGY.get("L0_scope", []) or []
L1_LIST: List[str] = ONTOLOGY.get("L1", []) or []
L2_MAP: Dict[str, List[str]] = ONTOLOGY.get("L2", {}) or {}
L2_RULES: Dict[str, Dict[str, List[str]]] = ONTOLOGY.get("L2_RULES", {}) or {}

# Optional (if you add these into ontology.yaml)
L0_RULES: Dict[str, List[str]] = ONTOLOGY.get("L0_RULES", {}) or {}
L0_TO_L1: Dict[str, List[str]] = ONTOLOGY.get("L0_TO_L1", {}) or {}

# If L0 exists but L0_TO_L1 is absent, we infer a sane default:
# - put all current L1 into INSURANCE_RISK (because your current L1 are insurance topics A~E)
# - leave GENERAL_ECONOMICS empty (you can later add GENERAL_ECONOMICS L1s)
if L0_SCOPE and not L0_TO_L1:
    inferred = {}
    for l0 in L0_SCOPE:
        if l0 == "INSURANCE_RISK":
            inferred[l0] = list(L1_LIST)
        else:
            inferred[l0] = []
    L0_TO_L1 = inferred


# ---------------------------
# Paper -> text (single source of truth)
# ---------------------------
def paper_text(p: Dict[str, Any]) -> str:
    """
    Return text for weak labeling.

    Priority:
        1) raw_text.abstract
        2) raw_text.keywords_text
        3) metadata.title (fallback)
    """
    raw = p.get("raw_text") or {}
    meta = p.get("metadata") or {}

    abstract = (raw.get("abstract") or "").strip()
    keywords_text = (raw.get("keywords_text") or "").strip()
    title = (meta.get("title") or p.get("title") or "").strip()

    chunks = []
    if abstract:
        chunks.append(abstract)
    if keywords_text:
        chunks.append(keywords_text)

    # Fallback: if both are empty, at least use title
    if not chunks and title:
        chunks.append(title)

    return "\n".join(chunks).strip()
