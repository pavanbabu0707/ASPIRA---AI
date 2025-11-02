# app/semantic_app.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ====== Config ======
TV_PATH = Path("data/features/training_view.parquet")  # change if your parquet lives elsewhere
MODEL_NAME = "all-MiniLM-L6-v2"

# ====== UI ======
st.set_page_config(page_title="AspiraAI – Semantic Search", page_icon="🔎", layout="wide")
st.title("AspiraAI 🔎 — Semantic Role Finder")
st.caption("Type a free-text profile (skills, tools, interests). We embed your text and match roles from the parquet.")

# ====== Helpers ======
EXPAND_MAP = {
    "frontend": ["ui", "ux", "react", "typescript", "css", "javascript"],
    "back-end": ["api", "microservices", "java", "spring", "python", "django", "flask"],
    "full-stack": ["frontend", "backend", "react", "node", "typescript"],
    "data engineer": ["etl", "spark", "airflow", "aws", "glue", "pipelines", "databricks"],
    "mlops": ["kubernetes", "docker", "mlflow", "terraform", "ci/cd", "airflow"],
    "analyst": ["sql", "excel", "tableau", "power bi", "dashboards", "analytics"],
    "product manager": ["stakeholder", "roadmap", "metrics", "a/b testing", "user research"],
    "devops": ["docker", "kubernetes", "ci/cd", "terraform", "ansible"],
    "cloud": ["aws", "gcp", "azure", "iam", "lambda"],
}

def expand_query(q: str) -> str:
    q_low = q.lower()
    extras = []
    for k, vals in EXPAND_MAP.items():
        if k in q_low:
            extras.extend(vals)
    return q if not extras else q + " " + " ".join(sorted(set(extras)))

def safe_tokenize(q: str) -> set[str]:
    # keep tokens like c++, ci/cd, node.js
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_/\+\.\-]*", q.lower()))

def keyword_boost(role: str, q: str) -> float:
    words = safe_tokenize(q)
    base = f" {role.lower()} "
    hits = sum(1 for w in words if f" {w} " in base)
    return min(0.05, 0.01 * hits)

# ====== Data / Model cache ======
@st.cache_resource
def load_training_view(tv_path: Path):
    if not tv_path.exists():
        raise FileNotFoundError(
            f"Missing parquet at {tv_path}.\n"
            "• Copy/build it to data/features/training_view.parquet, or\n"
            "• Update TV_PATH in app/semantic_app.py to the correct location."
        )
    df = pd.read_parquet(tv_path)

    # roles (with fallback)
    roles = df.get("canonical_role")
    if roles is None:
        roles = pd.Series([""] * len(df))
    fallback = df.get("raw_title")
    if fallback is None:
        fallback = pd.Series(["Unknown"] * len(df))
    roles = roles.fillna("").astype(str).str.strip()
    roles = roles.mask(roles.eq(""), fallback.fillna("Unknown").astype(str).str.strip())

    # text embedding column
    vecs = df["text_vec"].to_numpy()
    mat = np.vstack([np.asarray(v, dtype="float32").ravel() for v in vecs]).astype("float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms

    # Optional metadata to show later
    show_cols = [c for c in ["onet_code", "title", "description"] if c in df.columns]
    meta = df[show_cols].reset_index(drop=True)

    return df, roles.reset_index(drop=True), mat, meta

@st.cache_resource
def load_encoder(name: str):
    return SentenceTransformer(name)

# Load resources
try:
    df, roles, MAT, meta = load_training_view(TV_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

model = load_encoder(MODEL_NAME)

# ====== Sidebar ======
with st.sidebar:
    st.header("Controls")
    topk_roles = st.slider("Top roles", 3, 20, 8, 1)
    pool = st.slider("Candidate pool (rows scanned)", 50, max(100, len(MAT)), min(300, len(MAT)), 50)
    min_score = st.slider("Min score filter", 0.0, 0.6, 0.20, 0.01)
    show_table = st.checkbox("Show results table", value=True)
    st.divider()
    st.subheader("Examples")
    examples = [
        "aws python data engineer (etl, spark, airflow, databricks)",
        "frontend react typescript ui, design systems",
        "analyst sql tableau powerbi, A/B testing",
        "mlops kubernetes airflow mlflow on AWS",
        "product manager user research metrics roadmap",
    ]
    for e in examples:
        if st.button(e, use_container_width=True):
            st.session_state["q"] = e

# ====== Main input ======
if "q" not in st.session_state:
    st.session_state["q"] = ""

col1, col2 = st.columns([3, 1])
with col1:
    q = st.text_area(
        "Describe your skills / experience",
        value=st.session_state["q"],
        placeholder="e.g., data engineer with python, spark, aws; built ETL with airflow; dashboards in Tableau",
        height=120,
    )
with col2:
    st.write("")
    run = st.button("Search", use_container_width=True)
    clear = st.button("Clear", use_container_width=True)
    if clear:
        st.session_state["q"] = ""
        q = ""

# ====== Search ======
if run:
    st.session_state["q"] = q
    txt = q.strip()
    if not txt:
        st.warning("Please enter a few skills/keywords.")
        st.stop()

    q_expanded = expand_query(txt)
    qv = model.encode([q_expanded], normalize_embeddings=True)[0].astype("float32")

    sims = MAT @ qv
    k = min(pool, len(sims) - 1) if len(sims) > 1 else 1
    cand_idx = np.argpartition(-sims, k)[:k]
    cand_roles = roles.iloc[cand_idx].values
    cand_sims = sims[cand_idx].astype("float32")

    # aggregate by role
    by_role: dict[str, list[float]] = {}
    for role, sim in zip(cand_roles, cand_sims):
        by_role.setdefault(role, []).append(float(sim))

    rows = []
    for role, arr in by_role.items():
        arr.sort(reverse=True)
        top3 = arr[:3] if len(arr) >= 3 else arr
        max_sim = arr[0]
        mean_top3 = float(np.mean(top3))
        score = 0.7 * max_sim + 0.3 * mean_top3 + keyword_boost(role, txt)
        if score >= min_score:
            rows.append((role, score, max_sim, mean_top3, len(arr)))

    rows.sort(key=lambda x: -x[1])
    rows = rows[:topk_roles]

    st.subheader("Top Matches")
    if not rows:
        st.info("No strong matches — try adding more specific skills.")
    else:
        for role, score, max_sim, mean_top3, n in rows:
            st.markdown(
                f"**{role}**  \n"
                f"<span style='opacity:.75'>score ≈ {score:.3f} • cosine ≈ {max_sim:.3f} • support={n}</span>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0.0), 1.0))

        if show_table:
            st.divider()
            st.write("Details table")
            df_out = pd.DataFrame(
                [{"role": r, "score": s, "cosine": c, "mean_top3": m, "support": n} for r, s, c, m, n in rows]
            )
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="semantic_matches.csv",
                mime="text/csv",
            )

st.caption("Runs locally on your parquet features (no API calls).")
