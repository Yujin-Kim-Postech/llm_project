# app.py
import json
import textwrap
from pathlib import Path
from collections import Counter, defaultdict
import ast
import html
import re

import streamlit as st
from graphviz import Digraph
import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
                "authors": [str(row.get("first_author", "")).strip()],
                "source_url": str(row.get("source_url","")),
                "study_type": str(row.get("study_type", "")),
                "Topic_L1": str(row.get("Topic_L1", "")),
                "Topic_L2": str(row.get("Topic_L2", ""))
            },
            "empirical_analysis": {
                "Dependent_Variable_Y": str(row.get("Dependent_Variable_Y", "")),
                "Y_Category": str(row.get("Y_Category", "")),
                "Proxy_for_Y": str(row.get("Proxy_for_Y", "")),
                "Independent_Variable_X": str(row.get("Independent_Variable_X", "")),
                "Proxy_for_X": str(row.get("Proxy_for_X", "")),
                "methodology": str(row.get("methodology", "")),
                "dataset_and_period": str(row.get("dataset_and_period", "")),
                "control_variables": str(row.get("control_variables", "")),
                "unit_of_analysis": str(row.get("unit_of_analysis", "")),
                "results": str(row.get("results", "")),
                "keywords": str(row.get("keywords", "")),
                
            },
            "Topic_L1": str(row.get("Topic_L1", "")),
            "Topic_L2": str(row.get("Topic_L2", "")),
            "study_type": str(row.get("study_type", ""))
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
def extract_y_category(paper: dict) -> str | None:
    ea = paper.get("empirical_analysis")
    if not isinstance(ea, dict):
        return None

    y_cat = ea.get("Y_Category")
    if y_cat and str(y_cat).strip():
        return str(y_cat).strip()

    return "(no Y category / missing field)"

def extract_summary_subject(paper: dict) -> str | None:
    ea = paper.get("empirical_analysis")
    if not isinstance(ea, dict):
        return None

    # 1순위: results
    results = ea.get("results")
    if isinstance(results, str):
        r = results.strip()
        if r:
            return r

    # 2순위: Topic_L2
    topic_l2 = paper.get("Topic_L2")
    if isinstance(topic_l2, str):
        t2 = topic_l2.strip()
        if t2:
            return t2

    # 3순위: Topic_L1
    topic_l1 = paper.get("Topic_L1")
    if isinstance(topic_l1, str):
        t1 = topic_l1.strip()
        if t1:
            return t1

    return None


def shorten(s: str | None, n: int = 220) -> str:
    if not s:
        return ""
    s = " ".join(str(s).split())
    return (s[: n - 1] + "…") if len(s) > n else s


def clean_summary_text(s, max_items=2, max_chars=260):
    if not s:
        return ""

    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", "", s)
    s = s.strip()

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            parsed = [str(x).strip() for x in parsed if str(x).strip()]
            s = " / ".join(parsed[:max_items])
    except Exception:
        pass

    s = " ".join(s.split())

    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"

    return s


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

def expand_terms(text):
    text = (text or "").lower()

    mapping = {
        "annual reports": [
            "annual report", "financial report", "disclosure",
            "risk disclosure", "corporate disclosure",
            "10-k", "reporting", "financial statement"
        ],
        "insurance demand": [
            "insurance demand", "insurance purchase",
            "insurance adoption", "take-up", "uptake",
            "insurance decision", "policy purchase"
        ],
    }

    terms = set(text.split())

    for k, v in mapping.items():
        if k in text:
            terms.update(v)

    return list(terms)

def relevance_score(p, x_terms, y_terms):
    text = " ".join([
        str(p.get("metadata", {}).get("title", "")),
        str(p.get("empirical_analysis", {}).get("Independent_Variable_X", "")),
        str(p.get("empirical_analysis", {}).get("Dependent_Variable_Y", "")),
        str(p.get("empirical_analysis", {}).get("keywords", "")),
        str(p.get("empirical_analysis", {}).get("results", "")),
    ]).lower()

    x_match = sum(1 for t in x_terms if t in text)
    y_match = sum(1 for t in y_terms if t in text)

    return x_match, y_match

def classify_papers(papers, input_x, input_y):
    x_terms = expand_terms(input_x)
    y_terms = expand_terms(input_y)

    xy, y_only, x_only, method = [], [], [], []

    for p in papers:
        x_m, y_m = relevance_score(p, x_terms, y_terms)

        if x_m > 0 and y_m > 0:
            xy.append(p)
        elif y_m > 0:
            y_only.append(p)
        elif x_m > 0:
            x_only.append(p)
        else:
            method.append(p)

    return xy, y_only, x_only, method

def sort_recent(papers):
    def get_year(p):
        try:
            return int(float(p.get("metadata", {}).get("year") or 0))
        except:
            return 0
    return sorted(papers, key=get_year, reverse=True)

def select_papers(papers, input_x, input_y, max_papers=10):
    xy, y_only, x_only, method = classify_papers(papers, input_x, input_y)

    xy = sort_recent(xy)
    y_only = sort_recent(y_only)
    x_only = sort_recent(x_only)
    method = sort_recent(method)

    selected = []

    # quota
    q_xy = 3
    q_y = 4
    q_x = 2
    q_m = 1

    selected += xy[:q_xy]
    selected += y_only[:q_y]
    selected += x_only[:q_x]
    selected += method[:q_m]

    # 부족하면 전체에서 채움
    if len(selected) < max_papers:
        remaining = sort_recent(papers)
        for p in remaining:
            if p not in selected:
                selected.append(p)
            if len(selected) >= max_papers:
                break

    return selected[:max_papers]

def infer_variable_role(var_text: str, var_name: str = "X") -> str:
    v = (var_text or "").strip().lower()

    if not v:
        return f"{var_name} is not provided."

    role_patterns = {
        "disclosure/document-based variable": [
            "report", "reports", "filing", "filings", "disclosure", "statement",
            "annual report", "interim report", "10-k", "10-q", "sfcr", "rsr"
        ],
        "external shock or event": [
            "pandemic", "covid", "crisis", "war", "disaster", "earthquake",
            "flood", "shock", "recession"
        ],
        "policy or regulatory variable": [
            "regulation", "policy", "law", "reform", "mandate", "supervision",
            "capital requirement", "solvency"
        ],
        "behavioral or perception-based variable": [
            "risk aversion", "risk perception", "trust", "literacy", "awareness",
            "preference", "behavior", "attitude"
        ],
        "market or macroeconomic condition": [
            "competition", "premium", "price", "interest rate", "inflation",
            "unemployment", "market condition", "gdp"
        ],
        "demand or adoption outcome": [
            "demand", "purchase", "adoption", "take-up", "uptake", "enrollment",
            "subscription", "renewal", "lapse"
        ],
        "loss or risk outcome": [
            "loss", "losses", "claim", "claims", "default", "failure",
            "bankruptcy", "operational risk", "mortality", "morbidity"
        ],
        "performance outcome": [
            "performance", "profitability", "revenue", "growth", "productivity",
            "roe", "roa", "margin"
        ],
    }

    matched_roles = [
        role for role, keywords in role_patterns.items()
        if any(k in v for k in keywords)
    ]

    if matched_roles:
        role_text = "; ".join(matched_roles)
        return (
            f"{var_name} appears to be a {role_text}. "
            f"Before generating research questions, define one precise and measurable interpretation of {var_name}. "
            f"If multiple interpretations are possible, choose the most academically plausible one for insurance, "
            f"risk management, business, or finance research, and use it consistently."
        )

    return (
        f"{var_name} may be broad or ambiguous. "
        f"Before generating research questions, define one precise and measurable interpretation of {var_name}. "
        f"Do not assume multiple meanings at once."
    )

@st.cache_resource
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence_transformers is required for embedding-based retrieval. "
            "Install sentence-transformers and retry, or use the keyword-based selection fallback."
        ) from exc

    return SentenceTransformer("all-MiniLM-L6-v2")

def paper_search_text(p: dict) -> str:
    ea = p.get("empirical_analysis", {}) or {}
    meta = p.get("metadata", {}) or {}

    return " ".join([
        str(meta.get("title", "")),
        str(meta.get("journal", "")),
        str(meta.get("study_type", "")),
        str(meta.get("Topic_L1", "")),
        str(meta.get("Topic_L2", "")),
        str(ea.get("Independent_Variable_X", "")),
        str(ea.get("Proxy_for_X", "")),
        str(ea.get("Dependent_Variable_Y", "")),
        str(ea.get("Proxy_for_Y", "")),
        str(ea.get("methodology", "")),
        str(ea.get("dataset_and_period", "")),
        str(ea.get("results", "")),
        str(ea.get("keywords", "")),
    ]).lower()

def get_year_safe(p: dict) -> int:
    try:
        return int(float(p.get("metadata", {}).get("year") or 0))
    except:
        return 0

def select_papers_by_embedding(
    papers: list[dict],
    input_x: str,
    input_y: str,
    max_papers: int = 10,
    recency_weight: float = 0.05,
):
    """
    Semantic similarity + recency 기반 prior studies 선택
    """

    if not papers:
        return []

    try:
        model = load_embedding_model()
    except Exception:
        return select_papers(papers, input_x, input_y, max_papers=max_papers)

    query = f"""
    Independent variable: {input_x}
    Dependent variable: {input_y}
    Research field: insurance, risk management, quantitative finance, business research
    """

    paper_texts = [paper_search_text(p) for p in papers]

    query_emb = model.encode([query], normalize_embeddings=True)
    paper_embs = model.encode(paper_texts, normalize_embeddings=True)

    semantic_sims = (paper_embs @ query_emb[0]).astype(float)

    years = np.array([get_year_safe(p) for p in papers], dtype=float)
    if len(years) > 0 and years.max() > years.min():
        recency_scores = (years - years.min()) / (years.max() - years.min())
    else:
        recency_scores = np.zeros(len(papers), dtype=float)

    x_terms = expand_terms(input_x)
    y_terms = expand_terms(input_y)

    scores = []

    query_text = f"{input_x or ''} {input_y or ''}".lower()
    for idx, p in enumerate(papers):
        text = " ".join([
            str(p.get("metadata", {}).get("title", "")),
            str(p.get("empirical_analysis", {}).get("Independent_Variable_X", "")),
            str(p.get("empirical_analysis", {}).get("Dependent_Variable_Y", "")),
            str(p.get("empirical_analysis", {}).get("keywords", "")),
            str(p.get("empirical_analysis", {}).get("results", "")),
        ]).lower()

        x_match = sum(1 for t in x_terms if t in text)
        y_match = sum(1 for t in y_terms if t in text)

        title = str(p.get("metadata", {}).get("title", "")).lower()
        x_field = " ".join([
            str(p.get("empirical_analysis", {}).get("Independent_Variable_X", "")),
            str(p.get("empirical_analysis", {}).get("Proxy_for_X", ""))
        ]).lower()
        y_field = " ".join([
            str(p.get("empirical_analysis", {}).get("Dependent_Variable_Y", "")),
            str(p.get("empirical_analysis", {}).get("Proxy_for_Y", ""))
        ]).lower()

        x_title_or_variable_match = float(any(t in title or t in x_field for t in x_terms))
        y_title_or_variable_match = float(any(t in title or t in y_field for t in y_terms))
        xy_both_match = 1.0 if x_match > 0 and y_match > 0 else 0.0

        topic_l1 = str(p.get("Topic_L1", "") or "").lower()
        topic_l2 = str(p.get("Topic_L2", "") or "").lower()
        category_match = float(any(t in query_text for t in [topic_l1, topic_l2] if t))

        recency_score = float(recency_scores[idx])

        if x_match == 0 and y_match == 0 and semantic_sims[idx] < 0.2:
            continue

        score = (
            3.0 * xy_both_match
            + 2.0 * x_title_or_variable_match
            + 1.5 * y_title_or_variable_match
            + 1.0 * semantic_sims[idx]
            + 0.3 * recency_score
            + 0.5 * category_match
        )

        scores.append((idx, score))

    if not scores:
        return select_papers(papers, input_x, input_y, max_papers=max_papers)

    ranked_idx = [idx for idx, _ in sorted(scores, key=lambda kv: kv[1], reverse=True)]
    selected = [papers[i] for i in ranked_idx[:max_papers]]
    return selected

def build_rq_prompt(input_x, input_y, papers_in_node, max_papers=10):
    target_papers = select_papers_by_embedding(
        papers=papers_in_node,
        input_x=input_x,
        input_y=input_y,
        max_papers=max_papers,
    )
    use_context = len(target_papers) >= 5

    # ✅ Step 3 — context_text 조건부 생성
    context_lines = []

    for i, p in enumerate(target_papers[:max_papers], start=1):
        title = paper_title(p)
        citation = paper_citation_brief(p)
        journal = paper_journal(p)
        x = p.get("empirical_analysis", {}).get("Independent_Variable_X", "")
        y = p.get("empirical_analysis", {}).get("Dependent_Variable_Y", "")

        context_lines.append(
            f"{i}. Title: {title}\n"
            f"   Citation: {citation}\n"
            f"   Journal: {journal}\n"
            f"   X: {x}\n"
            f"   Y: {y}"
        )

    context_text = "\n".join(context_lines)

    # 🔥 Step 4 — 프롬프트 분기 (핵심)
    if use_context:
        knowledge_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[RETRIEVED PRIOR STUDIES]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following studies are relevant to X and Y.

{context_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[IMPORTANT SOURCE VALIDATION WARNING]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The retrieved studies are provided as search-based contextual references.
Do NOT assume that all titles, authors, years, or findings are perfectly accurate.
Use them only as tentative signals of nearby literature.

When using a prior study:
- Do not invent findings beyond the listed X, Y, title, journal, and citation.
- If the study’s relevance is uncertain, classify it as Priority 4 or mention uncertainty.
- Research gaps should be framed conservatively unless the study is clearly relevant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[PRIOR STUDIES USAGE HIERARCHY]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use prior studies according to the following priority:

Priority 1 — Studies directly linking X and Y:
  → Use to identify the main research gap and positioning.

Priority 2 — Studies related to Y only:
  → Use to understand demand-side mechanisms and behavioral context.

Priority 3 — Studies related to X only:
  → Use to understand disclosure/document effects and information channels.

Priority 4 — Methodologically relevant studies only:
  → Extract method logic only. Do NOT force topical connections.

If no Priority 1 studies exist:
  → You MUST explicitly state that no direct X–Y literature exists.
  → You MUST frame the research gap as a missing linkage between two partially studied domains.
  → ALL research questions must be constructed as bridging mechanisms between X and Y.
  → Use your own knowledge for content-level gaps.

Do NOT treat all prior studies as equally relevant.
Do NOT force connections between studies and X–Y.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[PRIOR STUDIES SELF-ASSESSMENT — MANDATORY]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before using prior studies, assess their relevance:

STEP 1 — Classify each study:
  - Priority 1: directly links X and Y → use for gap identification
  - Priority 2: related to Y only → use for demand-side context
  - Priority 3: related to X only → use for supply/disclosure/contextual channel
  - Priority 4: unrelated to both → IGNORE entirely

STEP 2 — Output this classification block before RQ generation:
  "Prior studies assessment:
   - Priority 1 found: YES / NO
   - Priority 2 count: N
   - Priority 3 count: N
   - Priority 4 ignored: N"

STEP 3 — Apply fallback rules:
  If Priority 1 = NONE:
    → State explicitly: "No direct X–Y literature identified."
    → Treat this as a bridging opportunity between two separately studied domains.
    → Draw content-level gaps from your own knowledge of insurance, risk management, and finance literature, not from Priority 4 studies.

  If Priority 2 + Priority 3 together < 3:
    → Rely primarily on your own domain knowledge.
    → Use retrieved studies only where genuinely applicable.

Do NOT force unrelated prior studies into the rationale.
"""
    else:
        knowledge_block = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[KNOWLEDGE SOURCE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Identify research gaps using your own knowledge of the academic literature 
  in insurance, risk management, and finance.
- No directly relevant prior studies are provided.
"""

    x_interpretation = infer_variable_role(input_x, "X")
    y_interpretation = infer_variable_role(input_y, "Y")

    prompt = f"""
You are an expert academic research advisor in insurance, risk management, 
and quantitative finance.

Your task is to generate NOVEL research questions that have NOT yet been extensively studied in existing literature.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Student Input]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X (Independent Variable): {input_x if input_x else "(not provided)"}
Y (Dependent Variable): {input_y if input_y else "(not provided)"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[VARIABLE INTERPRETATION — DECLARE FIRST]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before generating research questions, explicitly declare:

- X is interpreted as: [your chosen definition]
- X measurable proxy: [specific variable]

- Y is interpreted as: [your chosen definition]
- Y measurable proxy: [specific variable]

- Rationale for interpretation: [1–2 sentences]

All research questions MUST be conceptually consistent with this declaration.
However, measurable proxies may be adapted across research questions
when the unit of analysis changes, such as individual, household, firm, insurer, or market level.

For each research question, specify:
- unit of analysis
- X proxy appropriate for that unit
- Y proxy appropriate for that unit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[RECENCY PRIORITY]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Give more weight to recent literature (last 5–10 years) when identifying research gaps.
- Avoid generating research questions that were already addressed in recent studies.

{knowledge_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LITERATURE CONSISTENCY]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Ensure that each research question is consistent with existing literature 
  but extends it in a meaningful way.
- Avoid proposing questions that are already well-established.
- Focus on identifying under-explored mechanisms, contexts, or interactions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[INTERPRETATION QUALITY REQUIREMENT]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The interpretation must be specific, testable, and realistic.
- Avoid vague definitions such as "general disclosure" or "overall demand".
- The proxy must be observable in real-world data (e.g., financial filings, surveys, administrative data).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[NOVELTY REQUIREMENTS — MANDATORY]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each research question MUST satisfy at least TWO of the following:

① Methodological novelty  
   (non-standard methods, advanced econometrics, or uncommon modeling approaches)

② Contextual novelty  
   (under-studied populations, dynamics, or market conditions)

③ Variable-combination novelty  
   (new mechanisms, nonlinearities, or overlooked interactions)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[DIVERSITY REQUIREMENT — CRITICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The 5 research questions MUST be methodologically diverse.
- At least 3 questions must use DIFFERENT methodological perspectives 
  (e.g., causal inference, behavioral modeling, market frictions, network effects, 
  structural modeling, policy evaluation).
- Do NOT generate multiple questions using the same core mechanism.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[X–Y LINKAGE REQUIREMENT — CRITICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Each research question MUST explicitly explain the mechanism linking X to Y.
- Specify whether X is treated as a shock, disclosure, policy, market condition, behavioral factor, or measurable firm-level variable.
- Specify how Y can be operationalized as an empirical dependent variable.
- Avoid vague associations such as “X affects Y” without a clear transmission channel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[MECHANISM REQUIREMENT — MANDATORY]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each research question, explicitly describe the causal or theoretical mechanism in 2–3 sentences.

The mechanism must include:
1. How X changes perception, constraint, information processing, trust, liquidity preference, or expected utility.
2. How that intermediate channel changes Y.
3. Why this mechanism is not already fully resolved by prior literature.

Avoid simply saying “X affects Y.”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[AVOID — LOW CONTRIBUTION PATTERNS]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Avoid research questions that:
- Only estimate average treatment effects without uncovering mechanisms
- Rely on widely used and saturated frameworks without extension
- Provide descriptive comparisons without causal or structural interpretation
- Do not introduce new variation, interaction, or nonlinearity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Task]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate 5 NOVEL research questions connecting X → Y.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Instructions]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Generate EXACTLY 5 research questions.
2. Each question must correspond to ONE of the following types.
   You do NOT need to use all five types if doing so weakens the research logic.
   However, avoid repeating the same type more than twice.

   Available types:
   - Causal relationship
   - Moderating effect
   - Mediating mechanism
   - Comparison across contexts
   - Policy or practical implication

3. For EACH question, provide:

a. Research Question

b. Novelty Check:
   - Criteria satisfied: (choose at least two among ① ② ③)
   - Why novel:

c. Research Gap:
   - Clearly state what prior literature has NOT addressed

d. Rationale:
   - Why it is academically meaningful

e. Data:
   - Specify required dataset (type, unit, key variables)

f. Method:
   - Suggested empirical strategy
   - Use advanced methods ONLY when appropriate
   - Ensure methodological diversity across questions

g. Method Justification:
   - Why this method is appropriate for the research question
   - What identification problem or modeling challenge it solves
   - Why a simpler method may be insufficient

For each suggested method, justify why the method is needed.
Do NOT list advanced methods unless they directly solve an identification, heterogeneity, mediation, selection, or structural modeling problem.
If a simple regression is sufficient, say so.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Output Format]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Type: Causal]

Research Question:
...

Novelty Check:
- Criteria satisfied:
- Why novel:

Research Gap:
...

Rationale:
...

Data:
...

Method:
...

Method Justification:
...

(Repeat until exactly 5 research questions are generated. Types may vary, but avoid excessive repetition.)
"""
    return prompt.strip()

# -----------------------------
# UI

# -----------------------------
st.set_page_config(layout="wide")

# Paths
TREE_PATH = "tree.json"
PAPERS_PATH = "data/RQ_generator_dataset.xlsx"  # 필요 시 변경

tree = load_tree(TREE_PATH)
papers_idx = load_papers_index(PAPERS_PATH)
root_name = tree.get("name", "ROOT")
level1_nodes = tree.get("children", []) or []

st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", ["Node Explorer", "RQ Generator"]);
show_ids = st.sidebar.checkbox("Show paper_ids in leaf nodes", value=False)

# label -> node 매핑
def node_label(n: dict) -> str:
    name = str(n.get("name", "")).strip()
    value = n.get("value", None)
    return f"{name} (n={value})" if value is not None else name

level1_label_to_node = {node_label(n): n for n in level1_nodes}

st.sidebar.header("Node selection")
chosen_l1_label = st.sidebar.selectbox(
    "Level 1",
    ["(All categories)"] + list(level1_label_to_node.keys()),
    index=0
)

if chosen_l1_label == "(All categories)":
    selected_path = root_name
else:
    l1_node = level1_label_to_node[chosen_l1_label]
    l1_name = str(l1_node.get("name", "")).strip()

    raw_level2_nodes = l1_node.get("children", []) or []

    if len(raw_level2_nodes) == 1:
        only_child = raw_level2_nodes[0]
        only_child_name = str(only_child.get("name", "")).strip()
        grand_children = only_child.get("children", []) or []

        if grand_children and len(only_child_name) <= 3:
            level2_nodes = grand_children
            level2_parent_path = f"{root_name} / {l1_name} / {only_child_name}"
        else:
            level2_nodes = raw_level2_nodes
            level2_parent_path = f"{root_name} / {l1_name}"
    else:
        level2_nodes = raw_level2_nodes
        level2_parent_path = f"{root_name} / {l1_name}"

    level2_label_to_node = {node_label(n): n for n in level2_nodes}

    chosen_l2_label = st.sidebar.selectbox(
        "Level 2",
        ["(All under selected Level 1)"] + list(level2_label_to_node.keys()),
        index=0
    )

    if chosen_l2_label == "(All under selected Level 1)":
        selected_path = level2_parent_path
    else:
        l2_node = level2_label_to_node[chosen_l2_label]
        l2_name = str(l2_node.get("name", "")).strip()
        selected_path = f"{level2_parent_path} / {l2_name}"

paper_ids = []
papers_in_node = []
node = find_node_by_path(tree, selected_path)
if node is not None:
    paper_ids = collect_paper_ids_under(node)
    for pid in paper_ids:
        p = papers_idx.get(norm_pid(pid))
        if p:
            papers_in_node.append(p)

if page == "Node Explorer":
    st.title("Insurance & Risk Management Literature Tree Explorer")
    st.info(
        "본 화면의 논문 요약, 변수 정보 및 카테고리 분류는 생성형 AI를 활용하여 추출·정리된 결과이며, "
        "연구 활용 전 원문 확인이 필요합니다."
    )
    dot = build_graphviz(tree, show_paper_ids=show_ids)
    st.graphviz_chart(dot, use_container_width=True)

    if node is None:
        st.error("Selected node not found in tree.")
    else:
        st.subheader(f"Selection: {selected_path}")
        st.caption(f"Papers under this category: {len(paper_ids)}")

        y_to_papers = defaultdict(list)
        missing = []

        for pid in paper_ids:
            p = papers_idx.get(norm_pid(pid))
            if not p:
                missing.append(pid)
                continue
            y_cat = extract_y_category(p)
            y_to_papers[y_cat].append(pid)

        rows = []
        for y, ids in sorted(y_to_papers.items(), key=lambda kv: (kv[0].lower(), -len(kv[1]))):
            rows.append({"Y_Category": y, "n_papers": len(ids)})

        st.markdown("### Y categories under this node")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if len(rows) == 0:
            st.info("이 노드 아래에서 집계할 논문이 없어요 (paper_ids가 없거나 papers.jsonl 매칭 실패).")
            if missing:
                st.caption(f"tree에는 있는데 {PAPERS_PATH}에 없는 paper_id (처음 20개):")
                st.code("\n".join(missing[:20]))
            st.stop()

        st.markdown("### Papers by Selected Y Category")
        y_options = [r["Y_Category"] for r in rows]
        chosen_y = st.selectbox("Choose a Y Category", y_options)

        chosen_ids = y_to_papers.get(chosen_y, [])
        paper_list = []
        for pid in chosen_ids:
            p = papers_idx.get(norm_pid(pid))
            title = paper_title(p) if p else ""
            summary = extract_summary_subject(p) if p else ""
            summary = clean_summary_text(summary, max_items=2, max_chars=260)
            citation = paper_citation_brief(p) if p else ""
            journal = paper_journal(p) if p else ""
            doi = norm_pid(pid)
            paper_list.append({
                "citation": citation,
                "title": title,
                "summary": summary,
                "journal": journal,
                "doi": doi,
                "dependent_y": p.get("empirical_analysis", {}).get("Dependent_Variable_Y", "") if p else "",
                "independent_x": p.get("empirical_analysis", {}).get("Independent_Variable_X", "") if p else "",
            })

        st.markdown(f"#### **{chosen_y}** (n= {len(paper_list)})")
        table_rows = []
        for i, row in enumerate(paper_list, start=1):
            table_rows.append({
                "No": i,
                "Title": row.get("title", ""),
                "Author (Year)": row.get("citation", ""),
                "Journal": row.get("journal", ""),
                "Dependent Y": row.get("dependent_y", ""),
                "Summary": row.get("summary", ""),
                "DOI": f"https://doi.org/{row.get('doi','')}" if row.get("doi") else ""
            })

        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
            height=650
        )

        if missing:
            st.warning(f"{len(missing)} paper_ids were in tree.json but not found in {PAPERS_PATH}. (showing first 10)")
            st.code("\n".join(missing[:10]))

else:
    st.title("RQ Prompt Generator")
    st.info("""
    이 도구는 입력한 X와 Y를 기반으로 데이터셋 내 유사 연구를 참고하여  
    생성형 AI에 입력할 수 있는 Research Question 생성용 프롬프트를 제공합니다.

    ※ X와 Y를 모두 입력해야 합니다.  
    ※ X 또는 Y가 모호한 경우, 프롬프트 내에서 변수 해석을 먼저 명확히 하도록 설계되어 있습니다.  
    ※ Node Level 1과 Level 2를 모두 선택하지 않은 경우 많은 시간이 소요될 수 있습니다.

    ※ 본 시스템은 생성형 AI를 직접 호출하지 않습니다.  
    프롬프트를 복사하여 ChatGPT, Claude, Gemini 등에 붙여넣어 사용하세요.
    """)

    st.markdown("---")

    st.markdown("### 📊 선행논문 Priority 요구조건")

    st.markdown("""
    **Priority 1**  
    X와 Y 간의 직접적인 관계를 분석한 연구
                
    **Priority 2**  
    종속변수(Y)의 수요, 결정요인, 결과 등에 초점을 둔 연구
                
    **Priority 3**  
    독립변수(X)의 특성, 행동, 결정요인 등에 초점을 둔 연구
                
    **Priority 4**  
    주제와 직접 관련은 없지만 방법론적으로 참고 가능한 연구
    """)

    st.markdown("---")

    st.markdown("### 📊 Research Question Novelty 요구조건")

    st.markdown("""
    ① **방법론적 참신성 (Methodological Novelty)**  
    기존 연구에서 일반적으로 사용되지 않았던 비표준적 방법론, 고급 계량경제 기법, 또는 새로운 모델링 접근

    ② **맥락적 참신성 (Contextual Novelty)**  
    특정 집단, 시장 환경, 또는 시기적·제도적 변화 맥락 분석

    ③ **변수 결합의 참신성 (Variable-Combination Novelty)**  
    변수 간 새로운 결합, 상호작용 효과, 비선형 관계, 또는 잠재적 메커니즘 탐색
    """)

    st.markdown("---")

    st.caption(f"Current node: {selected_path} — {len(papers_in_node)} papers available")

    x = st.text_input("X", placeholder="예: pandemic, regulation, risk disclosure")
    y = st.text_input("Y", placeholder="예: insurance demand, operational risk losses, firm performance")
    run = st.button("RQ Prompt 생성", disabled=not (x.strip() and y.strip()))

    if run:
        if not x.strip() or not y.strip():
            st.warning("X와 Y를 모두 입력해주세요.")
        elif len(papers_in_node) == 0:
            st.warning("현재 선택된 노드에서 사용할 논문이 없습니다.")
        else:
            prompt = build_rq_prompt(
                input_x=x.strip(),
                input_y=y.strip(),
                papers_in_node=papers_in_node,
                max_papers=10
            )
            st.markdown("### 생성형 AI에 사용할 프롬프트")
            st.caption("📋 우측 상단 복사 버튼을 눌러 ChatGPT / Claude / Gemini에 붙여넣으세요")
            st.code(prompt, language="text")

