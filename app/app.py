# app/app.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

TV_PATH = Path("data/features/training_view.parquet")

# ---------- Streamlit UI setup ----------
st.set_page_config(page_title="CareerPath AI", page_icon="🧭", layout="wide")
st.markdown(
    """
    <style>
    .role-card {padding: 0.8rem 1rem; border-radius: 10px; border: 1px solid rgba(0,0,0,0.08); margin-bottom: 0.6rem;}
    .faint {opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Aspira AI — Intelligent Career Recommender")
st.caption("Describe your skills or interests, and I’ll suggest matching roles using semantic search and AI-powered embeddings.")

# ---------- Query expansion map ----------
EXPAND_MAP = {
    "frontend": ["ui", "ux", "web", "react", "typescript", "css", "javascript"],
    "back-end": ["api", "microservices", "java", "spring", "python", "django", "flask"],
    "full-stack": ["frontend", "backend", "react", "node", "typescript"],
    "data engineer": ["etl", "spark", "airflow", "aws", "glue", "pipelines"],
    "mlops": ["kubernetes", "docker", "airflow", "mlflow", "terraform", "ci/cd"],
    "analyst": ["sql", "excel", "tableau", "power bi", "dashboards", "analytics"],
    "product manager": ["stakeholder", "roadmap", "metrics", "a/b testing", "user research"],
    "devops": ["docker", "kubernetes", "ci/cd", "terraform", "ansible"],
    "cloud": ["aws", "gcp", "azure", "iam", "lambda", "kinesis"],
}

def expand_query(q: str) -> str:
    q_low = q.lower()
    extras = []
    for k, vals in EXPAND_MAP.items():
        if k in q_low:
            extras.extend(vals)
    if not extras:
        return q
    extras = sorted(set(extras))
    return q + " " + " ".join(extras)

# ---------- Safer tokenizer + keyword boost ----------
def _tokens(q: str) -> set[str]:
    """
    Tokenize text safely and retain tech tokens like 'c++', 'ci/cd', 'node.js'.
    """
    q = q.lower()
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_/\+\-]*", q.lower()))

def keyword_boost(role: str, q: str) -> float:
    words = _tokens(q)
    base = " " + role.lower() + " "
    hits = sum(1 for w in words if f" {w} " in base)
    return min(0.05, 0.01 * hits)

# ---------- Cached loading ----------
@st.cache_resource
def load_training_view():
    if not TV_PATH.exists():
        raise FileNotFoundError(f"Missing {TV_PATH}. Build features first.")
    df = pd.read_parquet(TV_PATH)

    # Get clean roles
    role_series = df.get("canonical_role")
    if role_series is None:
        role_series = pd.Series([""] * len(df))
    role_fallback = df.get("raw_title")
    if role_fallback is None:
        role_fallback = pd.Series(["Unknown"] * len(df))
    roles = role_series.fillna("").astype(str).str.strip()
    roles = roles.mask(roles.eq(""), role_fallback.fillna("Unknown").astype(str).str.strip())

    # Prepare embeddings matrix
    vecs = df["text_vec"].to_numpy()
    mat = np.vstack([np.asarray(v, dtype="float32").ravel() for v in vecs]).astype("float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms

    return df, roles, mat

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

df, roles, MAT = load_training_view()
model = load_model()

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    TOPK_ROLES = st.slider("Top roles", 3, 15, 7, 1)
    TOPK_NEIGHBORS = st.slider("Neighbor pool (rows scanned)", 50, 500, 300, 50)
    MIN_SCORE = st.slider("Min score filter", 0.0, 0.6, 0.20, 0.01)
    st.divider()
    st.subheader("Try examples")
    examples = [
        "aws python data engineer",
        "frontend react typescript ui",
        "analyst sql tableau powerbi",
        "mlops kubernetes airflow",
        "product manager a/b testing roadmap metrics",
    ]
    for e in examples:
        if st.button(e, use_container_width=True):
            st.session_state["q"] = e

# ---------- Input area ----------
if "q" not in st.session_state:
    st.session_state["q"] = ""

col1, col2 = st.columns([3, 1])
with col1:
    q = st.text_area(
        "Describe yourself (skills, tools, background)",
        value=st.session_state["q"],
        placeholder="e.g., data engineer with python, spark, aws; built ETL pipelines",
        height=120,
    )
with col2:
    st.write("")
    run = st.button("Recommend", use_container_width=True)
    clear = st.button("Clear", use_container_width=True)
    if clear:
        st.session_state["q"] = ""
        q = ""

# ---------- Role recommendations ----------
if run:
    st.session_state["q"] = q
    txt = q.strip()
    if not txt:
        st.warning("Please enter a few skills/keywords.")
    else:
        q_expanded = expand_query(txt)
        qv = model.encode([q_expanded], normalize_embeddings=True)[0].astype("float32")

        sims = MAT @ qv
        k = min(TOPK_NEIGHBORS, len(sims) - 1) if len(sims) > 1 else 1
        cand_idx = np.argpartition(-sims, k)[:k]
        cand_roles = roles.iloc[cand_idx].values
        cand_sims = sims[cand_idx].astype("float32")

        by_role = {}
        for role, sim in zip(cand_roles, cand_sims):
            by_role.setdefault(role, []).append(float(sim))

        scored = []
        for role, arr in by_role.items():
            arr.sort(reverse=True)
            top3 = arr[:3] if len(arr) >= 3 else arr
            max_sim = arr[0]
            mean_top3 = float(np.mean(top3))
            score = 0.7 * max_sim + 0.3 * mean_top3 + keyword_boost(role, txt)
            if score >= MIN_SCORE:
                scored.append((role, score, max_sim))

        scored.sort(key=lambda x: -x[1])
        scored = scored[:TOPK_ROLES]

        st.subheader("Top Recommended Roles")
        if not scored:
            st.info("No strong matches — try adding more specific skills.")
        else:
            for role, score, max_sim in scored:
                with st.container():
                    st.markdown(
                        f"<div class='role-card'><b>{role}</b><br>"
                        f"<span class='faint'>score ≈ {score:.3f}  •  cosine ≈ {max_sim:.3f}</span></div>",
                        unsafe_allow_html=True,
                    )
                st.progress(min(max(score, 0.0), 1.0))

# ---------- Footer ----------
st.caption("Aspira AI — Because every dream deserves a direction ✨")
