# app/app.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

APP_NAME = "Aspira AI — Role Recommender"
CENTROIDS = Path("models/role_centroids.parquet")
TRAINING_VIEW = Path("data/features/training_view.parquet")

st.set_page_config(page_title="Aspira AI", page_icon="🧭", layout="centered")
st.title(APP_NAME)
st.caption("Type a short profile (skills, tools, interests). I’ll suggest likely roles.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def _ensure_matrix(vec_series):
    arrs = []
    for v in vec_series:
        a = np.asarray(v, dtype=np.float32)
        arrs.append(a)
    mat = np.vstack(arrs).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms  # unit rows → cosine = dot

@st.cache_resource
def load_corpus():
    # Try precomputed centroids
    if CENTROIDS.exists():
        dfc = pd.read_parquet(CENTROIDS)
        role_col = "canonical_role" if "canonical_role" in dfc.columns else ("role" if "role" in dfc.columns else None)
        if role_col is None:
            # fallback: first column with 'role' in name, else first column
            cand = [c for c in dfc.columns if "role" in c.lower()]
            role_col = cand[0] if cand else dfc.columns[0]
        vec_col = "centroid" if "centroid" in dfc.columns else ("text_vec" if "text_vec" in dfc.columns else None)
        if vec_col is None:
            raise RuntimeError("role_centroids.parquet missing a 'centroid' column.")
        df_use = dfc[[role_col, vec_col]].rename(columns={role_col: "role", vec_col: "vec"}).copy()
        mat = _ensure_matrix(df_use["vec"])
        return df_use[["role"]].reset_index(drop=True), mat, "centroids"

    # Fallback: build role centroids from training_view
    if TRAINING_VIEW.exists():
        dft = pd.read_parquet(TRAINING_VIEW)
        role = dft["canonical_role"].fillna(dft.get("raw_title")).fillna("Unknown").astype(str)
        dft = dft.assign(role=role)
        dft["vec"] = dft["text_vec"].apply(lambda v: np.asarray(v, dtype=np.float32))
        grouped = dft.groupby("role", as_index=False)["vec"].apply(lambda s: np.mean(np.stack(s.values), axis=0))
        mat = _ensure_matrix(grouped["vec"])
        return grouped[["role"]].reset_index(drop=True), mat, "training_view (derived)"

    raise RuntimeError("No corpus found. Expected models/role_centroids.parquet OR data/features/training_view.parquet")

def search(query, mat, df_roles, model, topk):
    qv = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sims = mat @ qv  # cosine
    k = min(topk * 5, len(sims))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])][:topk]
    roles = df_roles["role"].iloc[idx].tolist()
    scores = sims[idx]
    return list(zip(roles, scores))

model = load_model()
try:
    df_roles, MAT, source = load_corpus()
    st.caption(f"Loaded corpus from **{source}** · roles: {len(df_roles)} · dim: {MAT.shape[1]}")
except Exception as e:
    st.error(str(e))
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = []

col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    q = st.text_area(
        "Describe yourself (skills, tools, background)",
        placeholder="e.g., data engineer with python, spark, aws; built ETL pipelines",
        height=120,
    )
with col2:
    topk = st.slider("Top K", min_value=3, max_value=15, value=7, step=1)
    run_demo = st.button("▶ Demo (5 searches)")

if st.button("Recommend"):
    txt = (q or "").strip()
    if not txt:
        st.warning("Please enter a short profile/skills.")
        st.stop()
    st.session_state["history"].append(txt)
    results = search(txt, MAT, df_roles, model, topk)
    st.subheader("Top roles")
    if not results or max(s for _, s in results) < 0.05:
        st.info("Low similarity. Try adding more skills or context.")
    for role, s in results:
        # map cosine [0,0.7] → [0,100]% for friendlier confidence display
        pct = int(np.interp(s, [0.0, 0.7], [0, 100]))
        st.write(f"**{role}** — cosine ≈ {s:.3f} · confidence ≈ {pct}%")
        st.progress(min(max(float(s), 0.0), 1.0))

if run_demo:
    DEMO_QUERIES = [
        "developer full stack react node",
        "data engineer python spark aws",
        "data analyst sql tableau business",
        "cloud devops aws kubernetes",
        "cyber security soc incident response",
    ]
    st.subheader("Demo searches")
    for dq in DEMO_QUERIES:
        res = search(dq, MAT, df_roles, model, topk=5)
        with st.expander(f"🔎 {dq}"):
            for role, s in res:
                pct = int(np.interp(s, [0.0, 0.7], [0, 100]))
                st.write(f"**{role}** — cosine ≈ {s:.3f} · confidence ≈ {pct}%")

with st.sidebar:
    st.header("History")
    if st.session_state["history"]:
        for h in reversed(st.session_state["history"][-5:]):
            st.write("•", h)
    else:
        st.caption("No queries yet")
