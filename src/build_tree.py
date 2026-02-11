from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _norm_label(x: Optional[str]) -> str:
    x = (x or "").strip()
    return x if x else "Unlabeled"


def main():
    # ✅ papers.jsonl을 읽어서 topicl1/topicl2 기반 트리를 만든다
    papers_path = ROOT / "papers.jsonl"  # 필요 시 경로만 바꿔주면 됨
    out_path = ROOT / "tree.json"

    # tree[L1][L2] = list of paper nodes
    # paper node에는 paper_id + title 정도를 넣어두면 UI에서 유용함
    tree: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for r in iter_jsonl(papers_path):
        pid = r.get("paper_id")
        if not pid:
            continue

        l1 = _norm_label(r.get("topicl1"))
        l2 = _norm_label(r.get("topicl2"))

        title = None
        meta = r.get("metadata") or {}
        if isinstance(meta, dict):
            title = meta.get("title")

        tree[l1][l2].append(
            {
                "paper_id": pid,
                "title": title,
            }
        )

    # D3-friendly format
    d3: Dict[str, Any] = {"name": "ROOT", "children": []}

    for l1, l2map in sorted(tree.items()):
        node_l1 = {"name": l1, "children": []}

        for l2, papers in sorted(l2map.items()):
            # paper_ids만 필요하면 papers에서 paper_id만 뽑아도 됨
            paper_ids = sorted([p["paper_id"] for p in papers])
            node_l2 = {
                "name": l2,
                "value": len(papers),
                "paper_ids": paper_ids,
                "papers": sorted(papers, key=lambda x: (x["title"] or "", x["paper_id"])),
            }
            node_l1["children"].append(node_l2)

        d3["children"].append(node_l1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d3, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
