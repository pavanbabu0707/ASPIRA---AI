import os
import requests
import streamlit as st
import numpy as np

# ---------- Config ----------
BASE_URL = os.getenv("ASPIRA_API_BASE", "http://127.0.0.1:8000")

# ---------- API Helpers ----------
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
        # e.g., ("file", ("resume.pdf", b"...", "application/pdf"))
        files = {file_tuple[0]: file_tuple[1]}
    r = requests.post(
        f"{BASE_URL}{path}",
        data=data or {},
        files=files,
        headers=headers,
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

# ---------- Personality / Trait Helpers ----------

TRAIT_NAMES = ["logic", "creative", "human", "risk"]

def blank_traits():
    return {t: 0.0 for t in TRAIT_NAMES}

# Mapping from persona answers -> trait contributions
QUESTION_CONFIG = {
    "Q1": {
        "label": "Q1. When you face a new problem…",
        "options": [
            "Break it down logically",
            "Brainstorm wild ideas",
            "Trust intuition",
            "Talk it out",
        ],
        "map": {
            "Break it down logically": {"logic": 2.0},
            "Brainstorm wild ideas": {"creative": 2.0},
            "Trust intuition": {"creative": 1.0, "risk": 1.0},
            "Talk it out": {"human": 2.0},
        },
    },
    "Q2": {
        "label": "Q2. Your brain feels happiest when...",
        "options": [
            "Solving puzzles",
            "Organizing and planning",
            "Creating something beautiful",
            "Analyzing behavior",
        ],
        "map": {
            "Solving puzzles": {"logic": 2.0},
            "Organizing and planning": {"logic": 1.5, "risk": -0.5},
            "Creating something beautiful": {"creative": 2.0},
            "Analyzing behavior": {"logic": 1.0, "human": 1.0},
        },
    },
    "Q3": {
        "label": "Q3. You’d rather be known for…",
        "options": [
            "Building something useful",
            "Inspiring creatively",
            "Leading change",
            "Understanding deeply",
        ],
        "map": {
            "Building something useful": {"logic": 1.5},
            "Inspiring creatively": {"creative": 2.0},
            "Leading change": {"human": 2.0, "risk": 1.0},
            "Understanding deeply": {"logic": 1.5},
        },
    },
    "Q4": {
        "label": "Q4. You care most about...",
        "options": [
            "Growth",
            "Freedom",
            "Impact",
            "Expression",
        ],
        "map": {
            "Growth": {"risk": 1.5},
            "Freedom": {"risk": 2.0},
            "Impact": {"human": 2.0},
            "Expression": {"creative": 1.5},
        },
    },
    "Q5": {
        "label": "Q5. What does “success” mean to you?",
        "options": [
            "Fulfillment",
            "Mastery",
            "Recognition",
            "Flexibility",
            "Balance",
        ],
        "map": {
            "Fulfillment": {"human": 1.5},
            "Mastery": {"logic": 1.5},
            "Recognition": {"risk": 1.0},
            "Flexibility": {"risk": 1.5},
            "Balance": {"human": 1.0, "risk": -0.5},
        },
    },
    "Q6": {
        "label": "Q6. When something doesn’t go as planned…",
        "options": [
            "Reflect and learn",
            "Try something new",
            "Fix it",
            "Adapt with others",
        ],
        "map": {
            "Reflect and learn": {"logic": 1.0, "human": 0.5},
            "Try something new": {"risk": 1.5},
            "Fix it": {"logic": 1.5},
            "Adapt with others": {"human": 1.5},
        },
    },
    "Q7": {
        "label": "Q7. What pushes you to keep going when things get tough?",
        "options": [
            "Challenge",
            "Responsibility",
            "Passion",
            "Freedom",
        ],
        "map": {
            "Challenge": {"risk": 1.5},
            "Responsibility": {"human": 1.5},
            "Passion": {"creative": 1.5},
            "Freedom": {"risk": 2.0},
        },
    },
}

# Career "persona" profiles in (logic, creative, human, risk) space
CAREER_PERSONA = {
    "Software Developer": {
        "logic": 0.8,
        "creative": 0.4,
        "human": 0.3,
        "risk": 0.4,
    },
    "Data Scientist": {
        "logic": 0.9,
        "creative": 0.5,
        "human": 0.4,
        "risk": 0.5,
    },
    "Machine Learning Engineer": {
        "logic": 0.9,
        "creative": 0.6,
        "human": 0.3,
        "risk": 0.6,
    },
    "Financial Quant Analyst": {
        "logic": 1.0,
        "creative": 0.3,
        "human": 0.2,
        "risk": 0.7,
    },
}

def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a.dot(b) / (na * nb))

def persona_from_answers(answers: dict):
    """
    answers: dict like {"Q1": "Break it down logically", ...}
    Returns: (traits_dict, persona_name, career_scores_sorted)
    """
    traits = blank_traits()

    for q_key, cfg in QUESTION_CONFIG.items():
        ans = answers.get(q_key)
        if not ans:
            continue
        contrib = cfg["map"].get(ans)
        if contrib:
            for t, v in contrib.items():
                traits[t] += v

    # normalize traits to 0-1 range for readability
    arr = np.array([traits[t] for t in TRAIT_NAMES], dtype=float)
    if arr.max() > arr.min():
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr_norm = arr

    traits_norm = {t: float(x) for t, x in zip(TRAIT_NAMES, arr_norm)}

    # persona label from strongest trait
    strongest = max(traits_norm, key=traits_norm.get)
    if strongest == "logic":
        persona_name = "Analytical Problem-Solver"
    elif strongest == "creative":
        persona_name = "Creative Explorer"
    elif strongest == "human":
        persona_name = "Human-centered Contributor"
    else:
        persona_name = "Adventurous Builder"

    # match careers in same 4D trait space
    scores = []
    for career, vec_dict in CAREER_PERSONA.items():
        c_vec = np.array(
            [vec_dict["logic"], vec_dict["creative"], vec_dict["human"], vec_dict["risk"]],
            dtype=float,
        )
        score = cosine(arr_norm, c_vec)
        scores.append((career, score))

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return traits_norm, persona_name, scores_sorted

# ---------- Streamlit UI Setup ----------
st.set_page_config(page_title="AspiraAI", page_icon="🎯", layout="wide")
st.title("AspiraAI 🎯")
st.caption("Submit a survey, upload a resume, and get career recommendations (FastAPI backend).")

# session state
if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "resume_preview" not in st.session_state:
    st.session_state.resume_preview = None
if "resume_id" not in st.session_state:
    st.session_state.resume_id = None
if "persona_traits" not in st.session_state:
    st.session_state.persona_traits = None
if "persona_name" not in st.session_state:
    st.session_state.persona_name = None
if "persona_career_scores" not in st.session_state:
    st.session_state.persona_career_scores = None

# --- Sidebar: User, Survey sliders, toggles ---
with st.sidebar:
    st.header("User / Output")
    st.session_state.user_id = st.number_input(
        "User ID", min_value=1, step=1, value=st.session_state.get("user_id", 1)
    )
    top_k = st.number_input("Top-K careers", min_value=1, max_value=20, step=1, value=5)
    use_resume = st.checkbox(
        "Use resume for content",
        value=True,
        help="If checked and a resume is uploaded, recommendations will favor resume content.",
    )

    st.markdown("---")
    st.header("Skill Survey (0.0–1.0)")
    s1 = st.slider("Skill 1", 0.0, 1.0, 0.7, 0.1)
    s2 = st.slider("Skill 2", 0.0, 1.0, 0.8, 0.1)
    s3 = st.slider("Skill 3", 0.0, 1.0, 0.5, 0.1)
    s4 = st.slider("Skill 4", 0.0, 1.0, 0.2, 0.1)
    s5 = st.slider("Skill 5", 0.0, 1.0, 0.1, 0.1)
    answers = [s1, s2, s3, s4, s5]

    if st.button("Submit Survey"):
        try:
            out = api_post(
                "/survey/submit",
                {"user_id": st.session_state.user_id, "answers": answers},
                token=st.session_state.token,
            )
            st.success(f"Saved survey ({out.get('id', 'ok')})")
        except Exception as e:
            st.error(str(e))

# --- Main Layout ---

# ---------- Auth ----------
with st.expander("Auth (optional)"):
    tab_reg, tab_login = st.tabs(["Register", "Login"])

    with tab_reg:
        r_email = st.text_input("Email", key="reg_email", placeholder="you@example.com")
        r_name = st.text_input("Full name", key="reg_full", placeholder="Your Name")
        r_pwd = st.text_input("Password", key="reg_pwd", type="password")
        if st.button("Register"):
            try:
                out = api_post(
                    "/auth/register",
                    {"email": r_email, "password": r_pwd, "full_name": r_name},
                )
                st.success(f"Registered user id={out.get('id')} email={out.get('email')}")
            except Exception as e:
                st.error(str(e))

    with tab_login:
        l_email = st.text_input("Email", key="log_email", placeholder="you@example.com")
        l_pwd = st.text_input("Password", key="log_pwd", type="password")
        if st.button("Login"):
            try:
                out = api_post("/auth/login", {"email": l_email, "password": l_pwd})
                st.session_state.token = out.get("access_token")
                st.success("Logged in. Token stored.")
            except Exception as e:
                st.error(str(e))

# ---------- Resume Upload ----------
with st.expander("Resume (Upload PDF/DOCX or paste text)"):
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
        if uploaded is not None:
            st.caption(f"Selected: {uploaded.name}")
    with right:
        resume_text = st.text_area(
            "…or paste resume / LinkedIn summary text",
            placeholder="Paste here if you don't want to upload a file.",
            height=160,
        )

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
                        file_tuple=(
                            "file",
                            (uploaded.name, uploaded.getvalue(), mime),
                        ),
                        token=st.session_state.token,
                    )
                else:
                    txt = (resume_text or "").strip()
                    if len(txt) < 20:
                        raise RuntimeError(
                            "Please paste at least ~20 characters of text if not uploading a file."
                        )
                    out = api_post_multipart(
                        "/nlp/resume/upload",
                        data={
                            "user_id": str(st.session_state.user_id),
                            "raw_text": txt,
                        },
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

# ---------- Personality & Mindset ----------
with st.expander("Personality & Mindset (Q1–Q7) – optional but powerful"):
    st.write(
        "These are based on the Google Form questions. "
        "Your answers here build a lightweight trait profile: "
        "`logic`, `creative`, `human`, `risk`."
    )

    answers_persona = {}
    for q_key, cfg in QUESTION_CONFIG.items():
        answers_persona[q_key] = st.radio(
            cfg["label"],
            cfg["options"],
            key=f"persona_{q_key}",
            horizontal=False,
        )

    if st.button("Analyze my personality profile"):
        traits_norm, persona_name, scores_sorted = persona_from_answers(answers_persona)
        st.session_state.persona_traits = traits_norm
        st.session_state.persona_name = persona_name
        st.session_state.persona_career_scores = scores_sorted

    if st.session_state.persona_traits:
        st.markdown("#### Your persona snapshot")
        cols = st.columns(4)
        for i, t in enumerate(TRAIT_NAMES):
            cols[i].metric(t.capitalize(), f"{st.session_state.persona_traits[t]:.2f}")

        st.markdown(f"**Persona label:** `{st.session_state.persona_name}`")

        st.markdown("#### Top career matches (personality-only)")
        for i, (career, score) in enumerate(
            st.session_state.persona_career_scores[:3], start=1
        ):
            st.write(f"{i}. **{career}** — similarity: `{score:.3f}`")

        st.info(
            "Use this as guidance: if your top persona match is, for example, "
            "**Machine Learning Engineer**, you can set your skill sliders and resume "
            "to emphasize ML, MLOps, and experimentation."
        )

st.divider()

# ---------- Recommendations (hybrid, using backend) ----------
st.subheader("Recommendations")

headline_default = "SQL, Python, ML Ops, finance, data pipelines"
headline = st.text_area(
    "Interests/skills summary (used if no resume)",
    value=headline_default,
    help="This is sent to the backend when no rich resume is available.",
)

if st.button("Get Recommendations (Hybrid Engine)"):
    payload = {
        "user_id": str(st.session_state.user_id),
        "headline": headline,
        "top_k": int(top_k),
    }
    try:
        out = api_post("/careers/recommend", payload, token=st.session_state.token)
        if use_resume and st.session_state.get("resume_id"):
            st.info(
                "Resume uploaded — backend is using resume + skills to rank careers. "
                "Personality (above) is currently used as a guidance layer for you."
            )
        if st.session_state.persona_name:
            st.caption(
                f"Persona context: you look like a **{st.session_state.persona_name}** "
                f"with top matches: {', '.join([c for c,_ in (st.session_state.persona_career_scores or [])[:2]])}"
            )
        st.json(out)
    except Exception as e:
        st.error(str(e))

st.divider()

# ---------- Careers from DB ----------
with st.expander("Careers (from DB)"):
    try:
        careers = api_get("/careers")
        st.json(careers)
    except Exception as e:
        st.info("Start the backend first.")
        st.caption(str(e))

st.caption(f"API: {BASE_URL}")
