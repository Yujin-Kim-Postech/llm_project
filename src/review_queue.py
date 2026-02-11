# src/review_queue.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]

# ---------- Policy ----------
# 핵심: QJE 같은 "보험 아닌 분야"는 대부분 GENERAL로 가야 하므로
# INSURANCE_RISK만 '충분히 강할 때' 자동 확정하고,
# 그 외는 전부 GENERAL_ECONOMICS로 자동 확정.
DEFAULT_POLICY = {
    "l0_policy": {"min_score": 1, "min_gap": 1},  # ✅ 요청: L0 min_score=1
    "l1_policy": {"min_score": 4, "min_gap": 2},
    "l2_policy": {"min_score": 4, "min_gap": 2},
}

AI_TAG_RE = re.compile(
    r"(?is)\bai\b|artificial intelligence|machine learning|deep learning|neural network|large language model|llm"
)

INSURANCE_RE = re.compile(r"(?is)\binsurance\b|\breinsurance\b|\bunderwriting\b|\bactuar(ial|y)\b|\bclaim(s)?\b|\bpremium(s)?\b")
GENERAL_RE = re.compile(r"(?is)\bmacroeconom(ics|y)\b|\bmonetary\b|\bfiscal\b|\bgdp\b|\binflation\b|\bunemployment\b|\btrade\b|\bindustrial organization\b|\blabor\b|\btax\b|\bpublic finance\b|\bdevelopment\b|\beducation\b|\bhealth\b")


def _paper_text(p: Mapping[str, Any]) -> str:
    title = str(p.get("title") or "")
    raw = p.get("raw_text") or {}
    abstract = ""
    keywords = ""
    if isinstance(raw, dict):
        abstract = str(raw.get("abstract") or "")
        keywords = str(raw.get("keywords_text") or "")
    return (title + "\n" + abstract + "\n" + keywords).strip()


def safe_tag_heuristics(p: Mapping[str, Any]) -> List[str]:
    tags: List[str] = []
    t = _paper_text(p)
    if AI_TAG_RE.search(t):
        tags.append("ai")
    return tags


def _decision(top3: List[List[Any]] | List[Tuple[Any, Any]], min_score: int, min_gap: int) -> Tuple[bool, int, int]:
    """
    Return (auto_ok, best_score, gap). Assumes top3 sorted.
    """
    if not top3:
        return (False, 0, 0)
    best = int(top3[0][1])
    second = int(top3[1][1]) if len(top3) > 1 else 0
    gap = best - second
    auto_ok = (best >= min_score) and (gap >= min_gap)
    return (auto_ok, best, gap)


@dataclass
class RecResult:
    top3: List[List[Any]]
    evidence: Dict[str, List[str]]


def recommend_l0(p: Mapping[str, Any]) -> RecResult:
    """
    매우 보수적으로 INSURANCE만 잡고, 나머지는 GENERAL로 fall back시키기 위한 L0 추천.
    """
    text = _paper_text(p)

    ins_hits = []
    gen_hits = []

    if INSURANCE_RE.search(text):
        ins_hits.append(INSURANCE_RE.pattern)
    if GENERAL_RE.search(text):
        gen_hits.append(GENERAL_RE.pattern)

    ins_score = 1 if ins_hits else 0
    gen_score = 1 if gen_hits else 0

    # top3 형식 유지
    pairs = [["INSURANCE_RISK", ins_score], ["GENERAL_ECONOMICS", gen_score]]
    pairs.sort(key=lambda x: (-int(x[1]), x[0]))

    return RecResult(
        top3=pairs,
        evidence={"INSURANCE_RISK": ins_hits, "GENERAL_ECONOMICS": gen_hits},
    )


def process_paper(p: Dict[str, Any]) -> Dict[str, Any]:
    p["auto_meta"] = p.get("auto_meta") or {}

    # 1) L0 scoring
    r0 = recommend_l0(p)
    p["l0_top3"] = r0.top3
    p["evidence_l0"] = r0.evidence

    # 2) L0 decision rule:
    #    - 보험이 "충분히 강하게" 잡히면 INSURANCE_RISK
    #    - 그 외는 모두 GENERAL_ECONOMICS (manual로 보내지 않음)
    l0_ok, l0_best, l0_gap = _decision(p["l0_top3"], **DEFAULT_POLICY["l0_policy"])
    best_label = p["l0_top3"][0][0] if p["l0_top3"] else ""

    if best_label == "INSURANCE_RISK" and l0_ok:
        p["final_l0"] = "INSURANCE_RISK"
        p["auto_meta"]["l0_reason"] = f"auto_insurance(score={l0_best},gap={l0_gap})"
    else:
        p["final_l0"] = "GENERAL_ECONOMICS"
        p["auto_meta"]["l0_reason"] = f"auto_default_general(best={best_label},score={l0_best},gap={l0_gap})"

    p["auto_meta"]["l0_policy"] = dict(DEFAULT_POLICY["l0_policy"])
    p["auto_meta"]["l1_policy"] = dict(DEFAULT_POLICY["l1_policy"])
    p["auto_meta"]["l2_policy"] = dict(DEFAULT_POLICY["l2_policy"])

    # 3) L1/L2: 지금은 GENERAL이면 보통 세부 분류 스킵(원하면 나중에 확장)
    p["l1_top3"], p["evidence_l1"] = [], {}
    p["l2_top3"], p["evidence_l2"] = [], {}
    p["final_l1"], p["final_l2"] = "", ""
    p["auto_meta"]["l1_reason"] = "skipped(l1_not_configured)"
    p["auto_meta"]["l2_reason"] = "skipped(no_final_l1)"

    # tags
    p["tags"] = list(dict.fromkeys((p.get("tags") or []) + safe_tag_heuristics(p)))
    return p


def _resolve_papers_path() -> Path:
    """
    GitHub Actions에서 papers.jsonl 경로가 흔히 달라져서, 존재하는 쪽을 자동 탐색.
    """
    candidates = [
        ROOT / "papers.jsonl",
        ROOT / "data" / "papers.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing papers.jsonl. Tried: {', '.join(map(str, candidates))}")


def run(in_jsonl: Path, out_jsonl: Path) -> None:
    out_lines: List[str] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            p2 = process_paper(p)
            out_lines.append(json.dumps(p2, ensure_ascii=False))

    out_jsonl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    papers_path = _resolve_papers_path()
    out_path = ROOT / "labels" / "review_queue.jsonl"
    run(papers_path, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
