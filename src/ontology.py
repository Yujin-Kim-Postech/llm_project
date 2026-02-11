# src/ontology.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import re

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_YAML = ROOT / "ontology.yaml"


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def paper_text(p: Mapping[str, Any]) -> str:
    """
    Extract a single text blob for labeling.

    Priority:
      1) p["raw_text"]["abstract"]
      2) p["raw_text"]["keywords_text"]
      3) fallback: title (+ optional fields if present)
    """
    raw = p.get("raw_text") or {}
    abstract = _safe_str(raw.get("abstract")).strip()
    keywords = _safe_str(raw.get("keywords_text")).strip()

    chunks: List[str] = []
    if abstract:
        chunks.append(abstract)
    if keywords:
        chunks.append(keywords)

    if chunks:
        return "\n".join(chunks)

    # Optional fallback (safe, but weaker)
    title = _safe_str(p.get("title")).strip()
    if title:
        chunks.append(title)

    # add lightweight extra metadata only if present
    venue = _safe_str(p.get("venue") or p.get("journal")).strip()
    if venue:
        chunks.append(venue)

    authors = p.get("authors")
    if isinstance(authors, list) and authors:
        # authors might be list[str] or list[dict]
        names: List[str] = []
        for a in authors[:20]:
            if isinstance(a, str):
                names.append(a)
            elif isinstance(a, dict):
                g = _safe_str(a.get("given")).strip()
                f = _safe_str(a.get("family")).strip()
                nm = (g + " " + f).strip()
                if nm:
                    names.append(nm)
        if names:
            chunks.append(", ".join(names))

    return "\n".join([c for c in chunks if c]).strip()


# -------------------------
# Ontology defaults (fallback if ontology.yaml not present)
# -------------------------

DEFAULT_L0_SCOPE: List[str] = [
    "INSURANCE_RISK",
    "GENERAL_ECONOMICS",
]

# Rules are regex strings. Score = number of unique regex patterns that match at least once.
DEFAULT_L0_RULES: Dict[str, List[str]] = {
    "INSURANCE_RISK": [
        r"(?is)\binsurance\b",
        r"(?is)\breinsurance\b",
        r"(?is)\bunderwriting\b",
        r"(?is)\bactuar(ial|y)\b",
        r"(?is)\bclaim(s)?\b",
        r"(?is)\bpremium(s)?\b",
    ],
    "GENERAL_ECONOMICS": [
        r"(?is)\bmacroeconom(ics|y)\b",
        r"(?is)\bmonetary\b",
        r"(?is)\bfiscal\b",
        r"(?is)\bgdp\b",
        r"(?is)\binflation\b",
        r"(?is)\bunemployment\b",
        r"(?is)\btrade\b",
        r"(?is)\bindustrial organization\b",
    ],
}

DEFAULT_L0_TO_L1: Dict[str, List[str]] = {
    # Optional. If you later add L1 rules, you can constrain candidates via L0.
    "INSURANCE_RISK": [],
    "GENERAL_ECONOMICS": [],
}


def _normalize_rules(obj: Any) -> Dict[str, List[str]]:
    """
    Accept YAML forms:
      - {LABEL: [regex, regex, ...]}
      - {LABEL: [{pattern: "...", weight: 1}, ...]}  -> we ignore weight for now but allow it.
    """
    out: Dict[str, List[str]] = {}
    if not isinstance(obj, dict):
        return out
    for label, rules in obj.items():
        if not isinstance(label, str):
            continue
        pats: List[str] = []
        if isinstance(rules, list):
            for r in rules:
                if isinstance(r, str):
                    pats.append(r)
                elif isinstance(r, dict):
                    pat = r.get("pattern")
                    if isinstance(pat, str) and pat.strip():
                        pats.append(pat.strip())
        elif isinstance(rules, str) and rules.strip():
            pats.append(rules.strip())
        out[label] = pats
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


_yaml_data = _load_yaml(ONTOLOGY_YAML)

# Export these (requested)
L0_SCOPE: List[str] = (
    list(_yaml_data.get("l0_scope"))
    if isinstance(_yaml_data.get("l0_scope"), list)
    else DEFAULT_L0_SCOPE
)

_yaml_l0_rules = _normalize_rules(_yaml_data.get("l0_rules"))
L0_RULES: Dict[str, List[str]] = _yaml_l0_rules if _yaml_l0_rules else DEFAULT_L0_RULES

_yaml_l0_to_l1 = _yaml_data.get("l0_to_l1")
if isinstance(_yaml_l0_to_l1, dict):
    L0_TO_L1: Dict[str, List[str]] = {
        k: list(v) if isinstance(v, list) else []
        for k, v in _yaml_l0_to_l1.items()
        if isinstance(k, str)
    }
else:
    L0_TO_L1 = DEFAULT_L0_TO_L1


# Optional future expansions (safe exports)
_yaml_l1_rules = _normalize_rules(_yaml_data.get("l1_rules"))
L1_RULES: Dict[str, List[str]] = _yaml_l1_rules if _yaml_l1_rules else {}

_yaml_l2_rules = _normalize_rules(_yaml_data.get("l2_rules"))
L2_RULES: Dict[str, List[str]] = _yaml_l2_rules if _yaml_l2_rules else {}

_yaml_l1_to_l2 = _yaml_data.get("l1_to_l2")
if isinstance(_yaml_l1_to_l2, dict):
    L1_TO_L2: Dict[str, List[str]] = {
        k: list(v) if isinstance(v, list) else []
        for k, v in _yaml_l1_to_l2.items()
        if isinstance(k, str)
    }
else:
    L1_TO_L2 = {}

# Optional policies (review_queue에서 사용 가능)
POLICIES: Dict[str, Any] = _yaml_data.get("policies") if isinstance(_yaml_data.get("policies"), dict) else {}
