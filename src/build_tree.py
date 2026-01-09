from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    labels_path = ROOT / "labels" / "paper_labels.jsonl"
    out_path = ROOT / "tree.json"

    # tree[L1][L2] = list of paper_ids
    tree = defaultdict(lambda: defaultdict(list))
    for r in iter_jsonl(labels_path):
        l1 = r.get("topic_l1","") or "Unlabeled"
        l2 = r.get("topic_l2","") or "Unlabeled"
        tree[l1][l2].append(r["paper_id"])

    # D3-friendly format
    d3 = {"name":"ROOT","children":[]}
    for l1, l2map in sorted(tree.items()):
        node_l1 = {"name": l1, "children": []}
        for l2, papers in sorted(l2map.items()):
            node_l2 = {"name": l2, "value": len(papers), "paper_ids": papers}
            node_l1["children"].append(node_l2)
        d3["children"].append(node_l1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d3, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
