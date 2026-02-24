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

def extract_summary_subject(paper: dict) -> str | None:
    ea = paper.get("empirical_analysis")
    if not isinstance(ea, dict):
        return None

    subj = ea.get("subject")
    # subject가 list인 경우 (예시처럼)
    if isinstance(subj, list) and len(subj) > 0:
        s = str(subj[0]).strip()
        return s if s else None

    # 혹시 string으로 들어오는 경우도 커버
    if isinstance(subj, str):
        s = subj.strip()
        return s if s else None

    return None


def shorten(s: str | None, n: int = 220) -> str:
    if not s:
        return ""
    s = " ".join(str(s).split())
    return (s[: n - 1] + "…") if len(s) > n else s

def paper_title(p: dict) -> str:
    m = p.get("metadata") or {}
    if isinstance(m, dict) and m.get("title"):
        return str(m["title"])
    return ""

def paper_authors(p: dict) -> str:
    """
    metadata.authors를 모두 출력.
    'Family, Given; Family, Given; ...' 형식
    """
    m = p.get("metadata") or {}
    authors = m.get("authors")

    if not authors:
        return ""

    names = []

    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, dict):
                given = (a.get("given") or a.get("first") or "").strip()
                family = (a.get("family") or a.get("last") or "").strip()

                if family and given:
                    nm = f"{family}, {given}"
                else:
                    nm = (a.get("name") or family or given).strip()

                if nm:
                    names.append(nm)

            elif isinstance(a, str) and a.strip():
                names.append(a.strip())

    elif isinstance(authors, str):
        names.append(authors.strip())

    return "; ".join(names)

def paper_year(p: dict) -> str:
    m = p.get("metadata") or {}
    year = m.get("year")

    if not year:
        return ""

    return str(year).strip()

def paper_citation_brief(p: dict) -> str:
    authors = paper_authors(p)
    year = paper_year(p)

    if authors and year:
        return f"{authors} ({year})"
    elif authors:
        return authors
    elif year:
        return f"({year})"
    else:
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

# Sidebar: node selection (Level 1 -> Level 2 with ALL options)
st.sidebar.header("Node selection")

root_name = tree.get("name", "ROOT")
level1_nodes = tree.get("children", []) or []

def node_label(n: dict) -> str:
    name = n.get("name", "")
    value = n.get("value", None)
    return f"{name} (n={value})" if value is not None else name


# -------------------------
# Level 1 (with ALL)
# -------------------------
level1_labels = ["(All categories)"] + [node_label(n) for n in level1_nodes]

chosen_l1_label = st.sidebar.selectbox(
    "Level 1 (A~F)",
    level1_labels,
    index=0
)

# ---- Level1 = ALL ----
if chosen_l1_label == "(All categories)":
    selected_path = root_name

# ---- Level1 = specific (A~F) ----
else:
    l1_idx = level1_labels.index(chosen_l1_label) - 1
    l1_node = level1_nodes[l1_idx]

    level2_nodes = l1_node.get("children", []) or []

    # -------------------------
    # Level 2 (with ALL under L1)
    # -------------------------
    if not level2_nodes:
        selected_path = f"{root_name} / {l1_node.get('name','')}"
    else:
        level2_labels = ["(All under Level 1)"] + [node_label(n) for n in level2_nodes]

        chosen_l2_label = st.sidebar.selectbox(
            "Level 2 (A1~...)",
            level2_labels,
            index=0
        )

        if chosen_l2_label == "(All under Level 1)":
            selected_path = f"{root_name} / {l1_node.get('name','')}"
        else:
            l2_idx = level2_labels.index(chosen_l2_label) - 1
            l2_node = level2_nodes[l2_idx]
            selected_path = f"{root_name} / {l1_node.get('name','')} / {l2_node.get('name','')}"

papers_idx = load_papers_index(PAPERS_PATH)

node = find_node_by_path(tree, selected_path)
if node is None:
    st.error("Selected node not found in tree.")
else:
    paper_ids = collect_paper_ids_under(node)

    st.subheader(f"Selection: {selected_path}")
    st.caption(f"Papers under this category: {len(paper_ids)}")

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
    st.markdown("### Papers by Selected Dependent Variables (Y)")
    y_options = [r["dependent_variable_Y"] for r in rows]
    chosen_y = st.selectbox("## Choose a dependent Y", y_options)

    chosen_ids = y_to_papers.get(chosen_y, [])
    paper_list = []
    for pid in chosen_ids:
        p = papers_idx.get(norm_pid(pid))
        title = paper_title(p) if p else ""
        summary = extract_summary_subject(p) if p else ""
        summary = summary or ""
        citation = paper_citation_brief(p) if p else ""
        doi = norm_pid(pid)

        paper_list.append({
            "citation": citation,
            "title": title,
            "summary": summary,
            "doi": doi,
        })

    # --- 논문 브라우저 스타일 출력 ---
    st.markdown(f"#### **{chosen_y}** (n= {len(paper_list)})")

    for i, row in enumerate(paper_list, start=1):
        title = row.get("title", "").strip() or "(no title)"
        citation = row.get("citation", "").strip()
        summary = row.get("summary", "").strip()
        doi = row.get("doi","").strip()

        with st.expander(f"{i}. {title}", expanded=False):
            if citation:
                st.markdown(f"**{citation}**")
            if doi:
                st.caption(f"DOI: https://doi.org/{doi}")
            if summary:
                st.markdown(summary)
            else:
                st.caption("(No summary available)")



    if missing:
        st.warning(f"{len(missing)} paper_ids were in tree.json but not found in {PAPERS_PATH}. (showing first 10)")
        st.code("\n".join(missing[:10]))