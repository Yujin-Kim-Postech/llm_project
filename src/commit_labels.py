# src/commit_labels.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def _resolve_papers_path() -> Path:
    candidates = [
        ROOT / "papers.jsonl",
        ROOT / "data" / "papers.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing papers.jsonl. Tried: {', '.join(map(str, candidates))}")


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")


def main() -> None:
    papers_path = _resolve_papers_path()
    review_path = ROOT / "labels" / "review_queue.jsonl"

    papers = _read_jsonl(papers_path)
    review = _read_jsonl(review_path)

    by_id: dict[str, Dict[str, Any]] = {p.get("paper_id", ""): p for p in papers if p.get("paper_id")}

    updated = 0
    for r in review:
        pid = r.get("paper_id")
        if not pid or pid not in by_id:
            continue

        # review_queue에서 확정된 값이 있으면 papers에 반영
        for k in ("final_l0", "final_l1", "final_l2", "tags"):
            if k in r:
                by_id[pid][k] = r[k]

        # meta도 유지하고 싶으면 반영
        if "auto_meta" in r:
            by_id[pid]["auto_meta"] = r["auto_meta"]

        updated += 1

    _write_jsonl(papers_path, list(by_id.values()))
    print(f"Updated {updated} papers -> {papers_path}")


if __name__ == "__main__":
    main()
