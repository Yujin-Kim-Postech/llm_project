# app.py
import json
import textwrap
from pathlib import Path
from collections import Counter, defaultdict

import streamlit as st
from graphviz import Digraph
import pandas as pd


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

def load_papers_index(papers_excel_path: str) -> dict:
    """
    normalized paper_id -> paper dict
    """
    idx = {}
    p = Path(papers_excel_path)
    if not p.exists():
        return idx

    try:
        df = pd.read_excel(p, engine="openpyxl")
    except ImportError as exc:
        st.error("Excel 파일을 읽으려면 openpyxl 패키지가 필요합니다. requirements.txt에 openpyxl을 추가하고 설치하세요.")
        st.stop()
        return idx
    for _, row in df.iterrows():
        doi = str(row.get("doi", "")).strip()
        if not doi:
            continue
        pid = norm_pid(doi)
        
        # Convert row to dict similar to jsonl structure
        paper = {
            "paper_id": doi,
            "metadata": {
                "title": str(row.get("title", "")),
                "journal": str(row.get("journal", "")),
                "year": row.get("year"),
                "authors": [{"family": str(row.get("first_author", ""))}],  # Simplified
            },
            "empirical_analysis": {
                "Dependent_Variable_Y": str(row.get("Dependent_Variable_Y", "")),
                "Proxy_for_Y": str(row.get("Proxy_for_Y", "")),
                "Independent_Variable_X": str(row.get("Independent_Variable_X", "")),
                "Proxy_for_X": str(row.get("Proxy_for_X", "")),
                "methodology": str(row.get("methodology", "")),
                "dataset_and_period": str(row.get("dataset_and_period", "")),
                "control_variables": str(row.get("control_variables", "")),
                "unit_of_analysis": str(row.get("unit_of_analysis", "")),
                "results": str(row.get("results", "")),
                "keywords": str(row.get("keywords", "")),
                "Topic_L1": str(row.get("Topic_L1", "")),
            },
            "Topic_L1": str(row.get("Topic_L1", "")),
            "Topic_L2": str(row.get("Topic_L2", "")),
            "study_type": str(row.get("study_type", "")),
        }
        if pid:
            idx[pid] = paper
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
    y = ea.get("Dependent_Variable_Y")
    if not y:
        return None
    return str(y).strip() if str(y).strip() else None

def extract_summary_subject(paper: dict) -> str | None:
    ea = paper.get("empirical_analysis")
    if not isinstance(ea, dict):
        return None

    subj = ea.get("Topic_L1")
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

def paper_journal(p: dict) -> str:
    """
    실제 저널명 출력용.
    우선순위: metadata.journal -> provenance.source_name
    """
    if not p:
        return ""

    m = p.get("metadata") or {}
    j = ""
    if isinstance(m, dict):
        j = (m.get("journal") or "").strip()
    if j:
        return j

    prov = p.get("provenance") or {}
    if isinstance(prov, dict):
        j2 = (prov.get("source_name") or "").strip()
        if j2:
            return j2

    return ""

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
PAPERS_PATH = "data/RQ_generator_dataset.xlsx"  # 필요 시 변경



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
paper_ids = []
papers_in_node = []

node = find_node_by_path(tree, selected_path)
if node is None:
    st.error("Selected node not found in tree.")
else:
    paper_ids = collect_paper_ids_under(node)
    papers_in_node = []
    for pid in paper_ids:
        p = papers_idx.get(norm_pid(pid))
        if p:
            papers_in_node.append(p)

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
        journal = paper_journal(p) if p else ""
        doi = norm_pid(pid)

        paper_list.append({
            "citation": citation,
            "title": title,
            "summary": summary,
            "journal": journal,
            "doi": doi,
        })

    # --- 고정 테이블 출력 ---
    st.markdown(f"#### **{chosen_y}** (n= {len(paper_list)})")

    table_rows = []
    for i, row in enumerate(paper_list, start=1):
        table_rows.append({
            "No": i,
            "Citation": row.get("citation", ""),
            "Journal": row.get("journal", ""),
            "Title": row.get("title", ""),
            "Summary": row.get("summary", ""),
            "DOI": f"https://doi.org/{row.get('doi','')}" if row.get("doi") else ""
        })

    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True
    )   


    if missing:
        st.warning(f"{len(missing)} paper_ids were in tree.json but not found in {PAPERS_PATH}. (showing first 10)")
        st.code("\n".join(missing[:10]))

import numpy as np
import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from openai import OpenAI

@st.cache_resource
def get_client():
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (StreamlitSecretNotFoundError, KeyError):
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. (.streamlit/secrets.toml 또는 환경변수로 설정)")
        st.stop()

    return OpenAI(api_key=api_key)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def build_context(papers_in_node: list[dict], max_papers: int = 25) -> str:
    # title/abstract만 넣어도 충분 (토큰 절약)
    chunks = []
    for p in papers_in_node[:max_papers]:
        title = (p.get("metadata", {}).get("title") or "").strip()
        abst  = (p.get("metadata", {}).get("abstract") or "").strip()
        if title:
            chunks.append(f"- {title}\n  {abst[:500]}")
    return "\n".join(chunks)

def generate_rqs(client, x, y, context, k=8):
    schema = {
        "name": "rq_bundle",
        "schema": {
            "type": "object",
            "properties": {
                "rqs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rq": {"type": "string"},
                            "x_used": {"type": "string"},
                            "y_used": {"type": "string"},
                            "motivation": {"type": "string"},
                            "suggested_design": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["rq", "x_used", "y_used", "motivation", "suggested_design", "keywords"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["rqs"],
            "additionalProperties": False
        },
        "strict": True
    }

    x_txt = x if x else "<미입력>"
    y_txt = y if y else "<미입력>"

    prompt = f"""
You will generate “new” and empirically testable research questions (RQs), avoiding questions that have already been addressed in the literature (see the context below).

User inputs:
- X: {x_txt}
- Y: {y_txt}

Rules:
- If X or Y is <not provided>, fill in the missing side with a “plausible candidate” based on the literature context and domain knowledge, and then generate the RQ.
- Preserve the user-provided side (X or Y) as much as possible; however, you may refine it into a more specific, measurable definition if needed.
- Propose {k} RQs, and for each RQ you must include x_used and y_used.

Literature context:
{context}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a careful research assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.7,
    )

    return json.loads(completion.choices[0].message.content)["rqs"]

def novelty_filter(client: OpenAI, rqs: list[dict], existing_texts: list[str], thr: float = 0.82):
    # existing_texts: 보통 title + abstract 결합한 fingerprint 추천
    rq_texts = [r["rq"] for r in rqs]
    E = embed_texts(client, existing_texts)     # (N, d)
    Q = embed_texts(client, rq_texts)           # (K, d)

    kept = []
    for i, r in enumerate(rqs):
        sims = [cosine_sim(Q[i], E[j]) for j in range(len(existing_texts))]
        mx = max(sims) if sims else 0.0
        r["max_similarity"] = mx
        r["is_novel_wrt_dataset"] = (mx < thr)
        kept.append(r)
    return kept

# -----------------------------
# Streamlit UI skeleton
# -----------------------------
st.title("RQ Generator")

x = st.text_input("X", placeholder="(미입력 가능)")
y = st.text_input("Y", placeholder="(미입력 가능)")

run = st.button("RQ 생성", disabled=not (x.strip() or y.strip()))

if run:
    if not (x.strip() or y.strip()):
        st.warning("X 또는 Y 중 하나 이상 입력해주세요.")
        st.stop()

    if len(papers_in_node) == 0:
        st.warning("현재 선택된 노드에서 papers.jsonl로 매칭된 논문이 없어 컨텍스트가 비어있습니다.")
        # 그래도 생성은 가능하게 두려면 st.stop()은 하지 말고 진행
        # st.stop()

    client = get_client()

    x_in = x.strip() or None
    y_in = y.strip() or None

    context = build_context(papers_in_node, max_papers=25)

    rqs = generate_rqs(client, x_in, y_in, context, k=8)

    existing_texts = []
    for p in papers_in_node:
        md = p.get("metadata", {}) or {}
        existing_texts.append((md.get("title", "") + " " + md.get("abstract", "")).strip())

    # existing_texts가 비면 임베딩 호출하지 않도록 처리
    if not any(t for t in existing_texts):
        for r in rqs:
            r["max_similarity"] = 0.0
            r["is_novel_wrt_dataset"] = True
        scored = rqs
    else:
        scored = novelty_filter(client, rqs, existing_texts, thr=0.82)

    st.dataframe(scored, use_container_width=True)