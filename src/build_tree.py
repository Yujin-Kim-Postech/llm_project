from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _norm(x: Optional[str]) -> str:
    x = (x or "").strip()
    return x if x else "Unlabeled"


TOP_RE = re.compile(r"^([A-F])(\d+)?([a-z])?$")  # e.g., A1a, A1, A, F2b

TOPIC_L0_LABEL = {
    "A": "Insurance Demand · Consumer Choice",
    "B": "Loss Modeling · Claims · Pricing · Operations (incl. fraud, triage)",
    "C": "Catastrophe · Climate · Reinsurance · ILS",
    "D": "Cyber · Technology Risk",
    "E": "Finance & Macro-Finance Links",
    "F": "Regulation · Accounting · Disclosure · Governance",
}

TOPIC_L1_LABEL = {
    "A1": "Retirement, Longevity, LTC",
    "A2": "Health insurance",
    "A3": "Index, agri insurance & consumer information design",
    "B1": "Claim frequency, severity & loss prediction",
    "B2": "Reserving, claims development, IBNR",
    "B3": "Claims operations & fraud, verification",
    "B4": "Underwriting, risk classification & information frictions",
    "C1": "Nat-cat & climate extremes: losses, exposure, insurance outcomes",
    "C2": "Reinsurance: capacity, pricing cycles, supply frictions",
    "C3": "CAT bonds, ILS: issuance, spreads, triggers, basis risk",
    "D1": "Cyber risk",
    "D2": "AI, Model risk & automation in insurance",
    "E1": "Risk premia & asset pricing",
    "E2": "Intermediation, systemic risk & financial stability",
    "E3": "Climate finance",
    "F1": "Solvency, capital regulation & market discipline",
    "F2": "Insurance & pension accounting, valuation",
    "F3": "Risk governance & culture",
    "F4": "Insurance market regulation, competition & availability",
}


def _first(x: Any) -> Optional[str]:
    """list면 첫 원소, str이면 그대로, 그 외 None"""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        return str(x[0])
    return None


def get_topic_fields(r: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """
    Support both schemas:
        - new: topicl1/topicl2 (string)
        - old: topic_l1/topic_l2 (list like ["A"], ["A1"])
    Returns (topicl1, topicl2) as strings or None
    """
    t1 = _first(r.get("topicl1"))
    t2 = _first(r.get("topicl2"))

    if t1 is None and t2 is None:
        # fallback to old schema
        old_l1 = _first(r.get("topic_l1"))  # e.g., "A"
        old_l2 = _first(r.get("topic_l2"))  # e.g., "A1"
        # 보통 old_l2가 더 정보가 많으니 topicl1에 넣고, topicl2는 비워둠
        # (A1a 같은 형태가 오면 split_topics가 알아서 처리 가능)
        if old_l2:
            t1 = old_l2
        elif old_l1:
            t1 = old_l1
        t2 = None

    return t1, t2


def split_topics(topicl1: Optional[str], topicl2: Optional[str]) -> tuple[str, str, str]:
    """
    Returns (L0, L1, L2)
        L0: A/B/C/D/E/F/Unlabeled
        L1: A1, B3, ...
        L2: a/b/c/Unlabeled
    Supports:
        - topicl1="A1", topicl2="a"
        - topicl1="A1a" (topicl2 missing)
        - topicl1 missing => Unlabeled
    """
    l1_raw = _norm(topicl1)
    l2_raw = _norm(topicl2)

    if l1_raw == "Unlabeled":
        return ("Unlabeled", "Unlabeled", l2_raw)

    m = TOP_RE.match(l1_raw)
    if not m:
        # fallback: group by first char if looks like A~F
        if l1_raw and l1_raw[0] in list("ABCDEF"):
            return (l1_raw[0], l1_raw, l2_raw)
        return ("Unlabeled", l1_raw, l2_raw)

    letter = m.group(1)                         # A
    num = m.group(2)                            # 1
    tail = m.group(3)                           # a

    l0 = letter
    l1 = f"{letter}{num}" if num else letter

    # if topicl2 is missing but topicl1 already contains tail (A1a)
    if (l2_raw == "Unlabeled") and tail:
        l2 = tail
    else:
        l2 = l2_raw

    return (l0, l1, l2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="data/papers.jsonl")
    ap.add_argument("--out", default="tree.json")
    ap.add_argument(
    "--include_theory",
    action="store_true",
    help="Include theory papers in the tree (default: exclude theory)",
    )
    args = ap.parse_args()

    papers_path = Path(args.papers)
    out_path = Path(args.out)

    # tree[L0][L1][L2] = list of papers
    tree: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    SELF_KEY = "__SELF__"  # topicl2 없는 paper를 L1에 붙이기 위한 내부 키

    for r in iter_jsonl(papers_path):
        pid = (r.get("paper_id") or "").strip()
        if not pid:
            continue
        
        # ✅ Exclude theory papers by default
        study_type = (r.get("study_type") or "").strip().lower()
        if (study_type == "theory") and (not args.include_theory):
            continue


        t1, t2 = get_topic_fields(r)
        l0, l1, l2 = split_topics(t1, t2)


        meta = r.get("metadata") or {}
        title = meta.get("title") if isinstance(meta, dict) else None
        paper_obj = {"paper_id": pid, "title": title}

        # ✅ topicl2가 Unlabeled면 leaf를 만들지 않고 L1에 직접 붙임
        if l2 == "Unlabeled":
            tree[l0][l1][SELF_KEY].append(paper_obj)
        else:
            tree[l0][l1][l2].append(paper_obj)

    # D3-friendly JSON
    d3: Dict[str, Any] = {"name": "ROOT", "children": []}

    def _sort_key_label(x: str):
        # Order A..F, then Unlabeled last
        if x == "Unlabeled":
            return (99, x)
        if len(x) == 1 and x in "ABCDEF":
            return (0, x)
        # For A1, A2... sort by letter then number
        m = re.match(r"^([A-F])(\d+)$", x)
        if m:
            return (1, m.group(1), int(m.group(2)))
        return (2, x)

    for l0 in sorted(tree.keys(), key=_sort_key_label):
        l0_label = TOPIC_L0_LABEL.get(l0, "")
        node_l0 = {"name": f"{l0}. {l0_label}" if l0_label else l0, "children": [], "topic_code": l0}


        for l1 in sorted(tree[l0].keys(), key=_sort_key_label):
            l2map = tree[l0][l1]

            # ✅ L1에 직접 붙는 paper들(= topicl2 없던 것들)
            self_papers = l2map.get(SELF_KEY, [])
            l1_label = TOPIC_L1_LABEL.get(l1, "")
            node_l1 = {"name": f"{l1}. {l1_label}" if l1_label else l1, "children": [], "topic_code": l1}


            if self_papers:
                node_l1["value"] = len(self_papers)
                node_l1["paper_ids"] = sorted([p["paper_id"] for p in self_papers])
                node_l1["papers"] = sorted(self_papers, key=lambda x: (x["title"] or "", x["paper_id"]))

            # ✅ 실제 L2들만 leaf로 생성 (Unlabeled는 애초에 안 만듦)
            for l2 in sorted([k for k in l2map.keys() if k != SELF_KEY]):
                papers = l2map[l2]
                node_l2 = {
                    "name": l2,
                    "value": len(papers),
                    "paper_ids": sorted([p["paper_id"] for p in papers]),
                    "papers": sorted(papers, key=lambda x: (x["title"] or "", x["paper_id"])),
                }
                node_l1["children"].append(node_l2)

            node_l0["children"].append(node_l1)

        d3["children"].append(node_l0)


    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d3, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
