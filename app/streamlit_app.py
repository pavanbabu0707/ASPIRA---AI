import os
import requests
import streamlit as st

BASE_URL = os.getenv("ASPIRA_API_BASE", "http://127.0.0.1:8000")

# ---------- HTTP helpers ----------
def api_post(path, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(f"{BASE_URL}{path}", json=body, headers=headers, timeout=30)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

def api_get(path, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=30)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

def api_post_multipart(path, data=None, file_tuple=None, token=None):
    """
    data: dict of form fields
    file_tuple: (field_name, (filename, bytes, mime))
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    files = None
    if file_tuple is not None:
        files = {file_tuple[0]: file_tuple[1]}
    r = requests.post(f"{BASE_URL}{path}", data=data or {}, files=files, headers=headers, timeout=90)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

# ---------- UI ----------
st.set_page_config(page_title="AspiraAI", page_icon="🎯", layout="wide")
st.title("AspiraAI 🎯")
st.caption("Submit a survey, upload a resume, and get career recommendations (FastAPI backend).")

# Session state
if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "resume_preview" not in st.session_state:
    st.session_state.resume_preview = None
if "resume_id" not in st.session_state:
    st.session_state.resume_id = None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("User / Output")
    st.session_state.user_id = st.number_input("User ID", min_value=1, step=1, value=st.session_state.user_id)
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5)
    use_resume = st.checkbox("Use resume", value=True, help="If ON, backend prefers your uploaded resume text over headline")

    st.markdown("---")
    st.subheader("Survey (0.0–1.0)")
    s1 = st.slider("Skill 1", 0.0, 1.0, 0.8, 0.05)
    s2 = st.slider("Skill 2", 0.0, 1.0, 0.8, 0.05)
    s3 = st.slider("Skill 3", 0.0, 1.0, 0.6, 0.05)
    s4 = st.slider("Skill 4", 0.0, 1.0, 0.4, 0.05)
    s5 = st.slider("Skill 5", 0.0, 1.0, 0.3, 0.05)
    answers = [s1, s2, s3, s4, s5]
    if st.button("Submit Survey"):
        try:
            out = api_post("/survey/submit", {"user_id": st.session_state.user_id, "answers": answers}, token=st.session_state.token)
            st.success(f"Saved survey ({out.get('id','ok')})")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("Tuning (weights)")
    alpha = st.slider("α content", 0.0, 1.0, 0.85, 0.01)
    beta  = st.slider("β survey", 0.0, 1.0, 0.30, 0.01)
    gamma = st.slider("γ overlap", 0.0, 0.5, 0.15, 0.01)
    tau   = st.slider("τ sharpen", 1.0, 3.0, 1.0, 0.1, help="1.0 = no sharpening; >1 squeezes lower scores")

# ---------- Resume upload ----------
with st.expander("Resume (Upload PDF/DOCX or paste text)"):
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
        if uploaded is not None:
            st.caption(f"Selected: {uploaded.name}")
    with right:
        resume_text = st.text_area("…or paste resume / LinkedIn summary text", height=160)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Upload / Parse Resume"):
            try:
                if uploaded is not None:
                    mime = (
                        "application/pdf"
                        if uploaded.name.lower().endswith(".pdf")
                        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    out = api_post_multipart(
                        "/nlp/resume/upload",
                        data={"user_id": str(st.session_state.user_id)},
                        file_tuple=("file", (uploaded.name, uploaded.getvalue(), mime)),
                        token=st.session_state.token,
                    )
                else:
                    if not resume_text or len(resume_text.strip()) < 20:
                        raise RuntimeError("Please paste at least ~20 characters of text if not uploading a file.")
                    out = api_post_multipart(
                        "/nlp/resume/upload",
                        data={"user_id": str(st.session_state.user_id), "raw_text": resume_text},
                        file_tuple=None,
                        token=st.session_state.token,
                    )
                st.session_state.resume_preview = out.get("preview", {})
                st.session_state.resume_id = out.get("resume_id")
                st.success(f"Resume parsed. ID: {st.session_state.resume_id}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    with col_b:
        if st.button("Clear Resume Preview"):
            st.session_state.resume_preview = None
            st.session_state.resume_id = None

    if st.session_state.resume_preview:
        st.caption("Parsed resume preview")
        st.json(st.session_state.resume_preview)

st.divider()

# ---------- Recommend ----------
st.subheader("Recommendations")
headline = st.text_area("Interests/skills summary (used if no resume)", value="Python, SQL, Airflow, AWS, Spark, MLOps, finance")

if st.button("Get Recommendations"):
    payload = {
        "user_id": str(st.session_state.user_id),
        "headline": headline,
        "top_k": int(top_k),
        "use_resume": bool(use_resume),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "tau": float(tau),
    }
    try:
        out = api_post("/careers/recommend", payload, token=st.session_state.token)
        st.write(f"Query: {'resume' if out.get('used_resume') else 'headline'}")
        items = out.get("items", [])
        for i, item in enumerate(items, start=1):
            st.markdown(
                f"**{i}. {item['title']}** — score: {item['score']} | "
                f"norm: {item.get('norm_score', 0)} | conf: {item.get('confidence', 0)}  \n"
                f"_content:_ {item['content_sim']} | _survey:_ {item['survey_aff']} | _overlap:_ {item['overlap']}"
            )
            st.caption(item.get("description", ""))
        if not items:
            st.info("No results.")
    except Exception as e:
        st.error(str(e))

st.caption(f"API: {BASE_URL}")
