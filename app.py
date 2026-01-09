# app_tree.py
import json
import textwrap
import streamlit as st
from graphviz import Digraph


def load_tree(path="tree.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wrap_label(s: str, width: int = 22) -> str:
    """Graphviz 노드 라벨 줄바꿈"""
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def build_graphviz(tree: dict, show_paper_ids: bool = False) -> Digraph:
    dot = Digraph("LiteratureTree")
    dot.attr(rankdir="TB")  # Top -> Bottom 트리
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

        # 라벨 구성
        label = name
        if value is not None:
            label += f"  (n={value})"
        if show_paper_ids and paper_ids:
            label += "\n" + "\n".join(paper_ids[:5])  # 너무 길어지면 상위 5개만
            if len(paper_ids) > 5:
                label += f"\n...(+{len(paper_ids)-5})"

        label = wrap_label(label, width=26)

        # 깊이에 따른 색상(원하면 바꿔도 됨)
        if depth == 0:
            dot.node(nid, label, fillcolor="#f2f2f2")  # ROOT
        elif depth == 1:
            dot.node(nid, label, fillcolor="#e8f0fe")  # L1
        else:
            # L2
            if name.lower() == "unlabeled":
                dot.node(nid, label, fillcolor="#ffecec")  # Unlabeled 강조
            else:
                dot.node(nid, label, fillcolor="#f6ffed")

        if parent_id is not None:
            dot.edge(parent_id, nid)

        for child in node.get("children", []):
            add_node(nid, child, depth + 1)

    add_node(None, tree, depth=0)
    return dot


st.set_page_config(layout="wide")
st.title("Insurance & Risk Management Literature Tree (Graph)")

show_ids = st.checkbox("Show paper_ids in leaf nodes", value=False)

tree = load_tree("tree.json")
dot = build_graphviz(tree, show_paper_ids=show_ids)

st.graphviz_chart(dot, use_container_width=True)

with st.expander("Raw Tree JSON"):
    st.json(tree)
