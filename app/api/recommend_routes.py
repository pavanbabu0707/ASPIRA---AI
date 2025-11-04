# app/api/recommend_routes.py
import os
import json
from typing import Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select, desc
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models_extra import Resume, Embedding

# --- Import your existing models (handles both common layouts) ---
try:
    # e.g., app/models/career.py, app/models/survey.py
    from app.models.career import Career
    from app.models.survey import SurveyResponse as Survey
except Exception:
    # e.g., app/db/models/career.py, app/db/models/survey.py
    from app.models.career import Career
    from app.models.survey import SurveyResponse as Survey
    

# Embedding helpers
from app.nlp.embeddings import embed_texts, upsert_embedding

# ---------- DB session ----------
DB_URL = os.getenv("DB_URL", "sqlite+aiosqlite:///./aspira_ai_db.db").replace("aiosqlite", "pysqlite")
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)

# ---------- Weights (defaults from .env or fallback) ----------
ALPHA = float(os.getenv("W_ALPHA", 0.55))  # content similarity (resume/headline vs. career text)
BETA  = float(os.getenv("W_BETA",  0.25))  # survey affinity  (survey vector vs. career skills vector)
# Future hooks:
# GAMMA = float(os.getenv("W_GAMMA", 0.15))  # skills overlap bonus
# DELTA = float(os.getenv("W_DELTA", 0.05))  # title/tag boost

router = APIRouter(prefix="/careers", tags=["careers-recommend"])


# ---------- Schemas ----------
class RecommendIn(BaseModel):
    user_id: str | int
    headline: Optional[str] = Field(default="", description="Fallback free-text if no resume")
    top_k: int = 10
    use_resume: bool = True


# ---------- Helpers ----------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2:
        a = a[0]
    if b.ndim == 2:
        b = b[0]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _as_vec5(x) -> Optional[np.ndarray]:
    """Try to coerce several possible fields into a float32 len-5 vector."""
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = json.loads(x)
        v = np.array(x, dtype=np.float32)
        if v.shape[0] == 5:
            return v
    except Exception:
        pass
    return None

def _get_latest_survey_vec(s, user_id) -> Optional[np.ndarray]:
    """
    Works with SurveyResponse; tolerates different field names:
      - answers
      - responses
      - response_vector
      - vector
    Orders by created_at desc if available, else by id desc.
    """
    order_col = getattr(Survey, "created_at", getattr(Survey, "id"))

    # Try string user_id first
    q = s.execute(
        select(Survey)
        .where(getattr(Survey, "user_id") == str(user_id))
        .order_by(desc(order_col))
    ).scalars().first()

    # If not found, try int user_id
    if not q:
        try:
            q = s.execute(
                select(Survey)
                .where(getattr(Survey, "user_id") == int(user_id))
                .order_by(desc(order_col))
            ).scalars().first()
        except Exception:
            q = None

    if not q:
        return None

    # Probe multiple possible attribute names on SurveyResponse
    for attr in ("answers", "responses", "response_vector", "vector"):
        v = _as_vec5(getattr(q, attr, None))
        if v is not None:
            return v

    return None



def _career_vec_from_db(c: Career) -> Optional[np.ndarray]:
    """Return the fixed 5-dim career skills vector (from DB JSON/text) or None."""
    try:
        sv = c.skills_vector if isinstance(c.skills_vector, list) else json.loads(c.skills_vector)
        v = np.array(sv, dtype=np.float32)
        if v.shape[0] != 5:
            return None
        return v
    except Exception:
        return None


def _ensure_career_embedding(s, career_id: int, title: str, desc_text: str) -> np.ndarray:
    """
    Ensure we have a semantic embedding for this career (title + description) in the embeddings table.
    Returns the vector (np.float32, shape [dim]).
    """
    emb = s.execute(
        select(Embedding).where(
            Embedding.ref_type == "career",
            Embedding.ref_id == career_id,
        )
    ).scalars().first()

    if emb:
        return np.frombuffer(emb.vector, dtype=np.float32)

    text = f"{title}. {desc_text}".strip()
    X = embed_texts([text])  # shape (1, dim)
    upsert_embedding(ref_type="career", ref_id=career_id, vec=X)
    return X[0]


def _user_text_for_embedding(s, payload: RecommendIn) -> str:
    """Prefer resume text if present & use_resume=True; else fall back to headline."""
    if payload.use_resume:
        res = s.execute(select(Resume).where(Resume.user_id == str(payload.user_id))).scalars().first()
        if res and (res.raw_text and len(res.raw_text.strip()) >= 20):
            return res.raw_text
    return (payload.headline or "").strip()


# ---------- Route ----------
@router.post("/recommend")
def recommend(payload: RecommendIn):
    with SessionLocal() as s:
        # 1) Build user content vector (resume or headline)
        text = _user_text_for_embedding(s, payload)
        if len(text) > 0:
            user_vec = embed_texts([text])[0]  # shape [dim]
        else:
            user_vec = None  # valid; content_sim will be 0

        # 2) Latest survey vector (optional)
        survey_vec = _get_latest_survey_vec(s, payload.user_id)

        # 3) Fetch all careers
        careers: List[Career] = s.execute(select(Career)).scalars().all()
        if not careers:
            raise HTTPException(status_code=404, detail="No careers found in DB")

        # 4) Score careers
        scored = []
        for c in careers:
            # content similarity (resume/headline vs. career text)
            content_sim = 0.0
            if user_vec is not None:
                c_emb = _ensure_career_embedding(s, c.id, c.title, c.description or "")
                content_sim = _cosine(user_vec, c_emb)

            # survey affinity (survey vector vs. career skills vector)
            survey_aff = 0.0
            if survey_vec is not None:
                c_vec = _career_vec_from_db(c)
                if c_vec is not None:
                    survey_aff = _cosine(survey_vec, c_vec)

            score = ALPHA * content_sim + BETA * survey_aff
            scored.append(
                {
                    "id": c.id,
                    "title": c.title,
                    "description": c.description,
                    "score": round(float(score), 4),
                    "content_sim": round(float(content_sim), 4),
                    "survey_aff": round(float(survey_aff), 4),
                    "skills": c.skills_vector,
                }
            )

        # 5) Top-K
        k = max(1, int(payload.top_k))
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:k]

        return {
            "ok": True,
            "query": "resume" if (payload.use_resume and len(text) > 0) else "headline",
            "used_resume": bool(payload.use_resume and len(text) > 0),
            "items": top,
        }
