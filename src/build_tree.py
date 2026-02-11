from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    labels_path = ROOT / "labels" / "paper_labels.jsonl"
    out_path = ROOT / "tree.json"

    # L0 buckets
    # - GENERAL_ECONOMICS: leaf at L0 with paper_ids
    # - INSURANCE_RISK: expand to L1 -> L2
    econ_papers = []
    ins_tree = defaultdict(lambda: defaultdict(list))  # ins_tree[L1][L2] = paper_ids
    unlabeled_papers = []

    for r in iter_jsonl(labels_path):
        l0 = (r.get("topic_l0") or "").strip() or "Unlabeled"
        pid = r.get("paper_id")

        if not pid:
            continue

        if l0 == "GENERAL_ECONOMICS":
            econ_papers.append(pid)
            continue

        if l0 == "INSURANCE_RISK":
            l1 = (r.get("topic_l1") or "").strip() or "Unlabeled"
            l2 = (r.get("topic_l2") or "").strip() or "Unlabeled"
            ins_tree[l1][l2].append(pid)
            continue

        # fallback: anything else goes to Unlabeled bucket
        unlabeled_papers.append(pid)

    # D3-friendly format
    d3 = {"name": "ROOT", "children": []}

    # GENERAL_ECONOMICS leaf
    d3["children"].append(
        {
            "name": "GENERAL_ECONOMICS",
            "value": len(econ_papers),
            "paper_ids": sorted(econ_papers),
        }
    )

    # INSURANCE_RISK subtree
    node_ins = {"name": "INSURANCE_RISK", "children": []}
    for l1, l2map in sorted(ins_tree.items()):
        node_l1 = {"name": l1, "children": []}
        for l2, papers in sorted(l2map.items()):
            node_l2 = {"name": l2, "value": len(papers), "paper_ids": sorted(papers)}
            node_l1["children"].append(node_l2)
        node_ins["children"].append(node_l1)
    d3["children"].append(node_ins)

    # Unlabeled leaf (optional)
    if unlabeled_papers:
        d3["children"].append(
            {
                "name": "Unlabeled",
                "value": len(unlabeled_papers),
                "paper_ids": sorted(unlabeled_papers),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d3, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
