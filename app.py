# app.py
import json
import textwrap
from pathlib import Path
from collections import Counter, defaultdict

import streamlit as st
from graphviz import Digraph


# -----------------------------
# IO helpers
# -----------------------------
def load_tree(path="tree.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def norm_pid(pid: str) -> str:
    pid = (pid or "").strip().lower()
    if not pid:
        return ""
    if pid.startswith("doi:"):
        pid = pid[4:]
    return pid

def load_papers_index(papers_jsonl_path: str) -> dict:
    """
    normalized paper_id -> paper dict
    """
    idx = {}
    p = Path(papers_jsonl_path)
    if not p.exists():
        return idx

    for r in iter_jsonl(p):
        raw = (r.get("paper_id") or "").strip()
        pid = norm_pid(raw)
        if pid:
            idx[pid] = r
    return idx


# -----------------------------
# Tree traversal helpers
# -----------------------------
def wrap_label(s: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def build_graphviz(tree: dict, show_paper_ids: bool = False) -> Digraph:
    dot = Digraph("LiteratureTree")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="white", fontname="Arial")
    dot.attr("edge", arrowsize="0.7")

    node_counter = {"i": 0}

    def new_id():
        node_counter["i"] += 1
        return f"n{node_counter['i']}"

    def add_node(parent_id: str | None, node: dict, depth: int):
        nid = new_id()

        name = node.get("name", "NA")
        value = node.get("value", None)
        paper_ids = node.get("paper_ids", [])

        label = name
        if value is not None:
            label += f"  (n={value})"
        if show_paper_ids and paper_ids:
            label += "\n" + "\n".join(paper_ids[:5])
            if len(paper_ids) > 5:
                label += f"\n...(+{len(paper_ids)-5})"

        label = wrap_label(label, width=26)

        if depth == 0:
            dot.node(nid, label, fillcolor="#f2f2f2")
        elif depth == 1:
            dot.node(nid, label, fillcolor="#e8f0fe")
        elif depth == 2:
            dot.node(nid, label, fillcolor="#e8f0fe")  # A~F 같은 2단계도 강조
        else:
            if str(name).lower() == "unlabeled":
                dot.node(nid, label, fillcolor="#ffecec")
            else:
                dot.node(nid, label, fillcolor="#f6ffed")

        if parent_id is not None:
            dot.edge(parent_id, nid)

        for child in node.get("children", []):
            add_node(nid, child, depth + 1)

    add_node(None, tree, depth=0)
    return dot


def list_nodes_with_paths(tree: dict):
    """
    Return list of (path_str, node_name).
    path_str example: ROOT / A / A1
    """
    out = []

    def dfs(node, path):
        name = node.get("name", "")
        cur_path = path + [name]
        out.append((" / ".join(cur_path), name))
        for ch in node.get("children", []):
            dfs(ch, cur_path)

    dfs(tree, [])
    return out


def find_node_by_path(tree: dict, path_str: str) -> dict | None:
    """
    Find node by exact path string created by list_nodes_with_paths.
    """
    parts = [p.strip() for p in path_str.split("/")]

    # normalize: list_nodes_with_paths uses " / " joins, so split('/') yields segments with spaces
    parts = [p.strip() for p in parts if p.strip()]

    def dfs(node, cur):
        name = node.get("name", "")
        nxt = cur + [name]
        if " / ".join(nxt) == " / ".join(parts):
            return node
        for ch in node.get("children", []):
            got = dfs(ch, nxt)
            if got is not None:
                return got
        return None

    return dfs(tree, [])


def collect_paper_ids_under(node: dict) -> list[str]:
    """
    Collect all paper_ids in subtree.
    leaf nodes store paper_ids; internal nodes may not.
    """
    ids = set()

    def dfs(n):
        for pid in n.get("paper_ids", []) or []:
            ids.add(pid)
        for ch in n.get("children", []) or []:
            dfs(ch)

    dfs(node)
    return sorted(ids)


# -----------------------------
# Dependent Y extraction
# -----------------------------
def extract_dependent_y(paper: dict) -> str | None:
    ea = paper.get("empirical_analysis")
    if not isinstance(ea, dict):
        return None
    y = ea.get("dependent_variable_Y")
    if not y:
        return None
    return str(y).strip() if str(y).strip() else None


def paper_title(p: dict) -> str:
    m = p.get("metadata") or {}
    if isinstance(m, dict) and m.get("title"):
        return str(m["title"])
    return ""


# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("Insurance & Risk Management Literature Tree (Graph)")

# Paths
TREE_PATH = "tree.json"
PAPERS_PATH = "data/papers.jsonl"  # 필요 시 변경



show_ids = st.checkbox("Show paper_ids in leaf nodes", value=False)

tree = load_tree(TREE_PATH)
dot = build_graphviz(tree, show_paper_ids=show_ids)

st.graphviz_chart(dot, use_container_width=True)

# Sidebar: node selection (click 대신)
st.sidebar.header("Node selection")
nodes = list_nodes_with_paths(tree)

# 사용자가 보통 A1 같은 걸 고르기 쉽게 "name"도 같이 보여주는 옵션 라벨 구성
options = [p for (p, _name) in nodes]

default_idx = 0
# ROOT / A / A1 같은 게 있으면 기본값을 A1로 잡고 싶으면 여기서 설정 가능
selected_path = st.sidebar.selectbox("Select a node (e.g., ROOT / A / A1)", options, index=default_idx)

papers_idx = load_papers_index(PAPERS_PATH)

node = find_node_by_path(tree, selected_path)
if node is None:
    st.error("Selected node not found in tree.")
else:
    paper_ids = collect_paper_ids_under(node)

    st.subheader(f"Selection: {selected_path}")
    st.caption(f"Papers under this node: {len(paper_ids)}")

    # dependent Y 집계
    y_to_papers = defaultdict(list)
    missing = []

    for pid in paper_ids:
        p = papers_idx.get(norm_pid(pid))
        if not p:
            missing.append(pid)
            continue
        y = extract_dependent_y(p)
        if y is None:
            y_to_papers["(no empirical Y / theory / missing field)"].append(pid)
        else:
            y_to_papers[y].append(pid)

    # Topline: Y 리스트 + 개수
    # Topline: Y 리스트 + 개수
    rows = []
    for y, ids in sorted(y_to_papers.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rows.append({"dependent_variable_Y": y, "n_papers": len(ids)})

    st.markdown("### Dependent variables (Y) under this node")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ✅ rows가 비면 여기서 종료 (selectbox 옵션 비어서 터지는 것 방지)
    if len(rows) == 0:
        st.info("이 노드 아래에서 집계할 논문이 없어요 (paper_ids가 없거나 papers.jsonl 매칭 실패).")
        if missing:
            st.caption(f"tree에는 있는데 {PAPERS_PATH}에 없는 paper_id (처음 20개):")
            st.code("\n".join(missing[:20]))
        st.stop()

    # 상세: 선택한 Y의 논문 목록
    st.markdown("### Drill-down: papers by selected Y")
    y_options = [r["dependent_variable_Y"] for r in rows]
    chosen_y = st.selectbox("Choose a dependent Y", y_options)

    chosen_ids = y_to_papers.get(chosen_y, [])
    paper_list = []
    for pid in chosen_ids:
        p = papers_idx.get(norm_pid(pid))
        title = paper_title(p) if p else ""
        paper_list.append({"paper_id": pid, "title": title})

    st.dataframe(paper_list, use_container_width=True, hide_index=True)

    if missing:
        st.warning(f"{len(missing)} paper_ids were in tree.json but not found in {PAPERS_PATH}. (showing first 10)")
        st.code("\n".join(missing[:10]))