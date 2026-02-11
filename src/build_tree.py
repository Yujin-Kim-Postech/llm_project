from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _norm_label(x: Optional[str]) -> str:
    x = (x or "").strip()
    return x if x else "Unlabeled"


def _resolve_from_repo_root(p: str) -> Path:
    """
    Resolve a path relative to the git repo root (best-effort).
    In GitHub Actions, cwd is typically repo root, so this is safe.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp

    # best-effort: walk upwards to find .git (for local runs)
    cur = Path.cwd().resolve()
    for _ in range(6):
        if (cur / ".git").exists():
            return (cur / pp).resolve()
        if cur.parent == cur:
            break
        cur = cur.parent

    # fallback: cwd-based
    return (Path.cwd() / pp).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="data/papers.jsonl", help="Input papers.jsonl path")
    ap.add_argument("--out", default="tree.json", help="Output tree.json path")
    args = ap.parse_args()

    papers_path = _resolve_from_repo_root(args.papers)
    out_path = _resolve_from_repo_root(args.out)

    if not papers_path.exists():
        # help debug in CI
        cwd = Path.cwd().resolve()
        candidates = []
        for cand in [cwd / "papers.jsonl", cwd / "data" / "papers.jsonl", cwd / "data" / "papers.jsonl"]:
            if cand.exists():
                candidates.append(str(cand))
        raise FileNotFoundError(
            f"papers.jsonl not found: {papers_path}\n"
            f"CWD: {cwd}\n"
            f"Try passing correct path via --papers. "
            f"Existing candidates found: {candidates}\n"
            f"Directory listing (cwd): {sorted([p.name for p in cwd.iterdir()])}"
        )

    # tree[L1][L2] = list of paper nodes
    tree: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for r in iter_jsonl(papers_path):
        pid = (r.get("paper_id") or "").strip()
        if not pid:
            continue

        l1 = _norm_label(r.get("topicl1"))
        l2 = _norm_label(r.get("topicl2"))

        meta = r.get("metadata") or {}
        title = meta.get("title") if isinstance(meta, dict) else None

        tree[l1][l2].append({"paper_id": pid, "title": title})

    # D3-friendly format
    d3: Dict[str, Any] = {"name": "ROOT", "children": []}

    for l1, l2map in sorted(tree.items()):
        node_l1 = {"name": l1, "children": []}

        for l2, papers in sorted(l2map.items()):
            paper_ids = sorted([p["paper_id"] for p in papers])
            node_l2 = {
                "name": l2,
                "value": len(papers),
                "paper_ids": paper_ids,
                "papers": sorted(papers, key=lambda x: (x["title"] or "", x["paper_id"])),
            }
            node_l1["children"].append(node_l2)

        d3["children"].append(node_l1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d3, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path} (papers={papers_path})")


if __name__ == "__main__":
    main()
