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

# ---------- UI ----------
st.set_page_config(page_title="AspiraAI", page_icon="🎯", layout="centered")
st.title("AspiraAI 🎯")
st.caption("Submit a survey and get career recommendations (via FastAPI).")

# session state
if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = 1

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

st.subheader("User / Output")
c1, c2 = st.columns(2)
with c1:
    st.session_state.user_id = st.number_input("User ID", min_value=1, step=1, value=st.session_state.user_id)
with c2:
    top_k = st.number_input("Top-K", min_value=1, max_value=20, step=1, value=3)

st.subheader("Survey Answers (0.0 – 1.0)")
s1 = st.slider("Skill 1", 0.0, 1.0, 0.7, 0.1)
s2 = st.slider("Skill 2", 0.0, 1.0, 0.8, 0.1)
s3 = st.slider("Skill 3", 0.0, 1.0, 0.5, 0.1)
s4 = st.slider("Skill 4", 0.0, 1.0, 0.2, 0.1)
s5 = st.slider("Skill 5", 0.0, 1.0, 0.1, 0.1)
answers = [s1, s2, s3, s4, s5]

c3, c4, c5 = st.columns(3)
with c3:
    if st.button("Submit Survey"):
        try:
            out = api_post("/survey/submit", {"user_id": st.session_state.user_id, "answers": answers}, token=st.session_state.token)
            st.success(f"Saved survey id={out.get('id')}")
        except Exception as e:
            st.error(str(e))
with c4:
    if st.button("Get Recommendations"):
        try:
            out = api_get(f"/survey/recommend/{st.session_state.user_id}?top_k={top_k}", token=st.session_state.token)
            st.json(out)
        except Exception as e:
            st.error(str(e))
with c5:
    if st.button("View My Surveys"):
        try:
            out = api_get(f"/survey/user/{st.session_state.user_id}", token=st.session_state.token)
            st.json(out)
        except Exception as e:
            st.error(str(e))

st.divider()
with st.expander("Careers (from DB)"):
    try:
        careers = api_get("/careers")
        st.json(careers)
    except Exception as e:
        st.info("Start the backend first.")
        st.caption(str(e))

st.caption(f"API: {BASE_URL}")
