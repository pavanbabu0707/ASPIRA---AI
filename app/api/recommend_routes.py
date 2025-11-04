# app/api/recommend_routes.py
import json
import os
from typing import Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models_extra import Resume, Embedding
from app.models.career import Career
from app.models.survey import SurveyResponse as Survey
from app.nlp.embeddings import embed_texts, upsert_embedding, get_cached_embedding

ALPHA = float(os.getenv("W_ALPHA", 0.55))  # content similarity (resume/headline ↔ career text)
BETA  = float(os.getenv("W_BETA",  0.25))  # survey affinity  (survey vec ↔ career skills vec)

router = APIRouter(prefix="/careers", tags=["careers-recommend"])

# ---------- Schemas ----------
class RecommendIn(BaseModel):
    user_id: str | int
    headline: Optional[str] = Field(default="", description="Used if no resume or use_resume=False")
    top_k: int = 10
    use_resume: bool = True

# ---------- Helpers ----------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2: a = a[0]
    if b.ndim == 2: b = b[0]
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _as_vec5(x) -> np.ndarray | None:
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

def _latest_survey_vec(s: Session, user_id) -> np.ndarray | None:
    # order by created_at desc if available, else id desc
    order_col = getattr(Survey, "created_at", getattr(Survey, "id"))
    # try string id
    q = s.execute(
        select(Survey).where(getattr(Survey, "user_id") == str(user_id)).order_by(desc(order_col))
    ).scalars().first()

    if not q:
        # try int id
        try:
            q = s.execute(
                select(Survey).where(getattr(Survey, "user_id") == int(user_id)).order_by(desc(order_col))
            ).scalars().first()
        except Exception:
            q = None

    if not q:
        return None

    for attr in ("answers", "responses", "response_vector", "vector"):
        v = _as_vec5(getattr(q, attr, None))
        if v is not None:
            return v
    return None

def _career_vec_from_db(c: Career) -> np.ndarray | None:
    try:
        sv = c.skills_vector if isinstance(c.skills_vector, list) else json.loads(c.skills_vector)
        v = np.array(sv, dtype=np.float32)
        if v.shape[0] == 5:
            return v
    except Exception:
        pass
    return None

def _ensure_career_emb(s: Session, career_id: int, title: str, desc_text: str) -> np.ndarray:
    v = get_cached_embedding(s, "career", career_id)
    if v is not None:
        return v
    text = f"{title}. {desc_text}".strip()
    vec = embed_texts([text])[0]
    upsert_embedding("career", career_id, vec, s=s)  # SAME session
    return vec

def _user_text(s: Session, user_id, headline: str, use_resume: bool) -> str:
    if use_resume:
        res = s.execute(select(Resume).where(Resume.user_id == str(user_id))).scalars().first()
        if res and res.raw_text and len(res.raw_text.strip()) >= 20:
            return res.raw_text
    return (headline or "").strip()

# ---------- Route ----------
@router.post("/recommend")
def recommend(payload: RecommendIn):
    with SessionLocal() as s:
        # 1) Build user content vector (resume or headline)
        text = _user_text(s, payload.user_id, payload.headline, payload.use_resume)
        user_vec = embed_texts([text])[0] if text else None

        # 2) Latest survey vector (optional)
        survey_vec = _latest_survey_vec(s, payload.user_id)

        # 3) All careers
        careers: List[Career] = s.execute(select(Career)).scalars().all()
        if not careers:
            raise HTTPException(status_code=404, detail="No careers found in DB")

        # 4) Score
        results = []
        for c in careers:
            content_sim = 0.0
            if user_vec is not None:
                c_emb = _ensure_career_emb(s, c.id, c.title, c.description or "")
                content_sim = _cosine(user_vec, c_emb)

            survey_aff = 0.0
            if survey_vec is not None:
                c_vec = _career_vec_from_db(c)
                if c_vec is not None:
                    survey_aff = _cosine(survey_vec, c_vec)

            score = ALPHA * content_sim + BETA * survey_aff
            results.append({
                "id": c.id,
                "title": c.title,
                "description": c.description,
                "score": round(float(score), 4),
                "content_sim": round(float(content_sim), 4),
                "survey_aff": round(float(survey_aff), 4),
                "skills": c.skills_vector,
            })

        # 5) Top-K
        results.sort(key=lambda x: x["score"], reverse=True)
        k = max(1, int(payload.top_k))
        return {
            "ok": True,
            "query": "resume" if (payload.use_resume and len(text) > 0) else "headline",
            "used_resume": bool(payload.use_resume and len(text) > 0),
            "items": results[:k],
        }
