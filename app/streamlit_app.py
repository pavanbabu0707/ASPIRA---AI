import os
import requests
import streamlit as st

# ---------- Config ----------
BASE_URL = os.getenv("ASPIRA_API_BASE", "http://127.0.0.1:8000")

# ---------- Helpers ----------
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

# ---------- Page ----------
st.set_page_config(page_title="AspiraAI", page_icon="🎯", layout="wide")
st.title("AspiraAI 🎯")
st.caption("Submit a survey, upload a resume, and get career recommendations (FastAPI backend).")

# ---------- Session ----------
if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "resume_preview" not in st.session_state:
    st.session_state.resume_preview = None
if "resume_id" not in st.session_state:
    st.session_state.resume_id = None

# ---------- Auth (optional) ----------
with st.expander("Auth (optional)"):
    tab_reg, tab_login = st.tabs(["Register", "Login"])
    with tab_reg:
        r_email = st.text_input("Email", key="reg_email", placeholder="you@example.com")
        r_name  = st.text_input("Full name", key="reg_full", placeholder="Your Name")
        r_pwd   = st.text_input("Password", key="reg_pwd", type="password")
        if st.button("Register"):
            try:
                out = api_post("/auth/register", {"email": r_email, "password": r_pwd, "full_name": r_name})
                st.success(f"Registered user id={out.get('id')} email={out.get('email')}")
            except Exception as e:
                st.error(str(e))
    with tab_login:
        l_email = st.text_input("Email", key="log_email", placeholder="you@example.com")
        l_pwd   = st.text_input("Password", key="log_pwd", type="password")
        if st.button("Login"):
            try:
                out = api_post("/auth/login", {"email": l_email, "password": l_pwd})
                st.session_state.token = out.get("access_token")
                st.success("Logged in. Token stored.")
            except Exception as e:
                st.error(str(e))

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("User / Output")
    st.session_state.user_id = st.number_input("User ID", min_value=1, step=1, value=st.session_state.get("user_id", 1))
    top_k = st.number_input("Top-K", min_value=1, max_value=20, step=1, value=5)
    use_resume = st.checkbox("Use resume for content", value=True, help="Bias content matching with your uploaded resume text (backend hook in Phase 4).")

    st.markdown("---")
    st.header("Survey (0.0–1.0)")
    s1 = st.slider("Skill 1", 0.0, 1.0, 0.7, 0.1)
    s2 = st.slider("Skill 2", 0.0, 1.0, 0.8, 0.1)
    s3 = st.slider("Skill 3", 0.0, 1.0, 0.5, 0.1)
    s4 = st.slider("Skill 4", 0.0, 1.0, 0.2, 0.1)
    s5 = st.slider("Skill 5", 0.0, 1.0, 0.1, 0.1)
    answers = [s1, s2, s3, s4, s5]
    if st.button("Submit Survey"):
        try:
            out = api_post("/survey/submit", {"user_id": st.session_state.user_id, "answers": answers}, token=st.session_state.token)
            st.success(f"Saved survey ({out.get('id','ok')})")
        except Exception as e:
            st.error(str(e))

# ---------- Resume (upload or paste) ----------
with st.expander("Resume (Upload PDF/DOCX or paste text)"):
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
        if uploaded is not None:
            st.caption(f"Selected: {uploaded.name}")
    with right:
        resume_text = st.text_area("…or paste resume / LinkedIn summary text",
                                   placeholder="Paste here if you don't want to upload a file.",
                                   height=160)
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Upload / Parse Resume"):
            try:
                if uploaded is not None:
                    mime = "application/pdf" if uploaded.name.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    out = api_post_multipart(
                        "/nlp/resume/upload",
                        data={"user_id": str(st.session_state.user_id)},
                        file_tuple=("file", (uploaded.name, uploaded.getvalue(), mime)),
                        token=st.session_state.token
                    )
                else:
                    text = (resume_text or "").strip()
                    if len(text) < 20:
                        raise RuntimeError("Please paste at least ~20 characters of text if not uploading a file.")
                    out = api_post_multipart(
                        "/nlp/resume/upload",
                        data={"user_id": str(st.session_state.user_id), "raw_text": text},
                        file_tuple=None,
                        token=st.session_state.token
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

# ---------- Recommendations (Hybrid endpoint) ----------
st.subheader("Recommendations")
headline = st.text_area("Interests/skills summary (used if no resume)", value="SQL, Python, ML Ops, finance, data pipelines")

if st.button("Get Recommendations"):
    payload = {
        "user_id": str(st.session_state.user_id),
        "headline": headline,
        "top_k": int(top_k)
    }
    try:
        out = api_post("/careers/recommend", payload, token=st.session_state.token)
        if use_resume and st.session_state.get("resume_id"):
            st.info("Resume uploaded — hybrid content matching will be enabled in the next backend step.")
        st.write(f"**Query**: {out.get('query','')}")
        items = out.get("items", [])
        if not items:
            st.warning("No results.")
        for it in items:
            with st.container(border=True):
                st.markdown(f"### {it['title']} — score: `{it.get('score','')}`")
                st.caption(f"Level: {it.get('level','')}, Tags: {it.get('tags','')}")
                st.markdown(f"**Skills:** {it.get('skills','')}")
                st.write(it.get("description",""))
    except Exception as e:
        st.error(str(e))

# ---------- Careers list (debug/peek) ----------
with st.expander("Careers (from DB)"):
    try:
        careers = api_get("/careers")
        st.json(careers)
    except Exception as e:
        st.info("Start the backend first.")
        st.caption(str(e))

st.caption(f"API: {BASE_URL}")
