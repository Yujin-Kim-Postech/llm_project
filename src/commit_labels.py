# src/commit_labels.py
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
INP = ROOT / "papers.jsonl"
OUT = ROOT / "papers_committed.jsonl"


def merge_labels(p):
    # Keep final_* if present, else fall back to existing.
    # This script should remain schema-stable as long as you keep final_l0/l1/l2 + tags.
    return p


def main():
    if not INP.exists():
        raise FileNotFoundError(f"Missing {INP}")

    out = []
    for line in INP.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        out.append(json.dumps(merge_labels(p), ensure_ascii=False))
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
