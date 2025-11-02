import os
import json
import requests
import numpy as np
import streamlit as st

BASE_URL = os.getenv("ASPIRA_API_BASE", "http://127.0.0.1:8000")

def api_post(path, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(f"{BASE_URL}{path}", json=body, headers=headers, timeout=20)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

def api_get(path, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=20)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

st.set_page_config(page_title="AspiraAI – Skills Builder", page_icon="🧩", layout="centered")
st.title("AspiraAI 🧩 – Skills → Career Recommendations")
st.caption("Pick your skills (Python, AWS, SQL…) and I’ll map them to a vector for recommendations via FastAPI.")

# --- session state ---
if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = 1

with st.expander("Auth (optional)"):
    c1, c2 = st.columns(2)
    with c1:
        r_email = st.text_input("Register: Email", key="reg_email2", placeholder="you@example.com")
        r_name  = st.text_input("Full name", key="reg_full2", placeholder="Your Name")
        r_pwd   = st.text_input("Password", key="reg_pwd2", type="password")
        if st.button("Register", key="reg_btn2"):
            try:
                out = api_post("/auth/register", {"email": r_email, "password": r_pwd, "full_name": r_name})
                st.success(f"Registered id={out.get('id')} email={out.get('email')}")
            except Exception as e:
                st.error(str(e))
    with c2:
        l_email = st.text_input("Login: Email", key="log_email2", placeholder="you@example.com")
        l_pwd   = st.text_input("Login: Password", key="log_pwd2", type="password")
        if st.button("Login", key="log_btn2"):
            try:
                out = api_post("/auth/login", {"email": l_email, "password": l_pwd})
                st.session_state.token = out.get("access_token")
                st.success("Logged in. Token stored.")
            except Exception as e:
                st.error(str(e))

st.subheader("User / Output")
colA, colB = st.columns(2)
with colA:
    st.session_state.user_id = st.number_input("User ID", min_value=1, step=1, value=st.session_state.user_id)
with colB:
    top_k = st.number_input("Top-K", min_value=1, max_value=20, step=1, value=3)

st.divider()
st.subheader("Pick your skills")

# --- taxonomy: skill -> 5-dim weights (Coding, ML/Math, DataEng, Systems/Cloud, Finance) ---
def V(c=0, ml=0, de=0, sys=0, fin=0):
    return np.array([c, ml, de, sys, fin], dtype="float32")

SKILL_MAP = {
    # Coding / Languages / Stacks
    "python": V(c=0.9, ml=0.5),
    "java": V(c=0.85),
    "c++": V(c=0.9),
    "typescript": V(c=0.8),
    "javascript": V(c=0.8),
    "react": V(c=0.7),
    "node.js": V(c=0.7, de=0.2),

    # Data / Analytics
    "sql": V(c=0.6, de=0.6),
    "pandas": V(c=0.7, de=0.5),
    "numpy": V(c=0.7, ml=0.4),
    "tableau": V(de=0.4),
    "power bi": V(de=0.4),

    # ML / DS
    "scikit-learn": V(ml=0.8, c=0.6),
    "pytorch": V(ml=0.9, c=0.7),
    "tensorflow": V(ml=0.9, c=0.6),
    "statistics": V(ml=0.8),
    "linear algebra": V(ml=0.8),

    # Data Eng / Pipelines
    "spark": V(de=0.9, c=0.5),
    "airflow": V(de=0.85),
    "dbt": V(de=0.8),
    "kafka": V(de=0.85, sys=0.3),
    "etl": V(de=0.9),
    "databricks": V(de=0.85),

    # Cloud / DevOps / Systems
    "aws": V(sys=0.9, de=0.5),
    "gcp": V(sys=0.85, de=0.5),
    "azure": V(sys=0.85, de=0.5),
    "docker": V(sys=0.85),
    "kubernetes": V(sys=0.9),
    "terraform": V(sys=0.8),

    # Finance / Quant
    "options": V(fin=0.85, ml=0.4, c=0.3),
    "time series": V(ml=0.7, fin=0.5),
    "risk models": V(fin=0.85, ml=0.5),
    "market microstructure": V(fin=0.85),
    "portfolio theory": V(fin=0.85, ml=0.4),
}

CATEGORIES = {
    "Coding / Languages": ["python","java","c++","typescript","javascript","react","node.js"],
    "Data / Analytics": ["sql","pandas","numpy","tableau","power bi"],
    "Machine Learning / DS": ["scikit-learn","pytorch","tensorflow","statistics","linear algebra"],
    "Data Engineering": ["spark","airflow","dbt","kafka","etl","databricks"],
    "Cloud / DevOps / Systems": ["aws","gcp","azure","docker","kubernetes","terraform"],
    "Finance / Quant": ["options","time series","risk models","market microstructure","portfolio theory"],
}

# collect chosen skills
chosen = []
for cat, skills in CATEGORIES.items():
    with st.expander(cat, expanded=False):
        cols = st.columns(3)
        for i, skill in enumerate(skills):
            if cols[i % 3].checkbox(skill, key=f"chk_{skill}"):
                chosen.append(skill)

# weight sliders for each dimension
st.subheader("Dimension Weights")
w_c   = st.slider("Coding / Software", 0.0, 2.0, 1.0, 0.1)
w_ml  = st.slider("Math / ML", 0.0, 2.0, 1.0, 0.1)
w_de  = st.slider("Data Eng / Pipelines", 0.0, 2.0, 1.0, 0.1)
w_sys = st.slider("Systems / Cloud / DevOps", 0.0, 2.0, 1.0, 0.1)
w_fin = st.slider("Finance / Quant", 0.0, 2.0, 1.0, 0.1)
W = np.array([w_c, w_ml, w_de, w_sys, w_fin], dtype="float32")

# build vector
def build_vector(skills: list[str]) -> list[float]:
    if not skills:
        return [0,0,0,0,0]
    mats = []
    for s in skills:
        v = SKILL_MAP.get(s.lower())
        if v is not None:
            mats.append(v)
    if not mats:
        return [0,0,0,0,0]
    M = np.vstack(mats)
    v = M.mean(axis=0)  # average contributions
    v = v * W          # apply dimension weights
    # normalize to [0,1]
    vmax = float(np.max(v)) if np.max(v) > 0 else 1.0
    v = v / vmax
    return [float(x) for x in v]

st.divider()
st.subheader("Actions")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Preview Vector"):
        vec = build_vector(chosen)
        st.json({"skills_selected": chosen, "vector": vec})

with col2:
    if st.button("Submit Survey + Recommend"):
        vec = build_vector(chosen)
        try:
            # submit survey
            out1 = api_post("/survey/submit", {"user_id": st.session_state.user_id, "answers": vec}, token=st.session_state.token)
            # get recs
            out2 = api_get(f"/survey/recommend/{st.session_state.user_id}?top_k={top_k}", token=st.session_state.token)
            st.success(f"Saved survey id={out1.get('id')}")
            st.json(out2)
        except Exception as e:
            st.error(str(e))

with col3:
    if st.button("View My Surveys"):
        try:
            out = api_get(f"/survey/user/{st.session_state.user_id}", token=st.session_state.token)
            st.json(out)
        except Exception as e:
            st.error(str(e))

st.divider()
with st.expander("Careers (from DB)"):
    try:
        st.json(api_get("/careers"))
    except Exception as e:
        st.caption(str(e))

st.caption(f"API: {BASE_URL}")
