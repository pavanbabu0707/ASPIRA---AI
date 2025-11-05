# app/api/recommend_routes.py
import os, re, json
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

# ---------- Tunables ----------
ALPHA = float(os.getenv("W_ALPHA", 0.70))   # content (resume/headline ↔ career text)
BETA  = float(os.getenv("W_BETA",  0.25))   # survey (5-d vector ↔ career skills vector)
GAMMA = float(os.getenv("W_GAMMA", 0.05))   # keyword overlap
TAU   = float(os.getenv("W_TAU",   1.5))    # sharpening exponent (>1 = sharper contrast)

router = APIRouter(prefix="/careers", tags=["careers-recommend"])

# ---------- Schemas ----------
class RecommendIn(BaseModel):
    user_id: str | int
    headline: Optional[str] = Field(default="", description="Used if no resume or use_resume=False")
    top_k: int = 10
    use_resume: bool = True

    # Optional overrides for weights
    alpha: Optional[float] = None
    beta:  Optional[float] = None
    gamma: Optional[float] = None
    tau:   Optional[float] = None


# ---------- Helper Functions ----------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2: a = a[0]
    if b.ndim == 2: b = b[0]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def _as_vec5(x) -> np.ndarray | None:
    if x is None: return None
    try:
        if isinstance(x, str): x = json.loads(x)
        v = np.array(x, dtype=np.float32)
        if v.shape[0] == 5: return v
    except Exception:
        pass
    return None


def _latest_survey_vec(s: Session, user_id) -> np.ndarray | None:
    order_col = getattr(Survey, "created_at", getattr(Survey, "id"))
    q = s.execute(
        select(Survey)
        .where(getattr(Survey, "user_id") == str(user_id))
        .order_by(desc(order_col))
    ).scalars().first()

    if not q:
        try:
            q = s.execute(
                select(Survey)
                .where(getattr(Survey, "user_id") == int(user_id))
                .order_by(desc(order_col))
            ).scalars().first()
        except Exception:
            q = None

    if not q: return None

    for attr in ("answers", "responses", "response_vector", "vector"):
        v = _as_vec5(getattr(q, attr, None))
        if v is not None:
            return v
    return None


def _career_vec5_from_db(c: Career) -> np.ndarray | None:
    try:
        sv = c.skills_vector if isinstance(c.skills_vector, list) else json.loads(c.skills_vector)
        v = np.array(sv, dtype=np.float32)
        if v.shape[0] == 5: return v
    except Exception:
        pass
    return None


def _ensure_career_emb(s: Session, career_id: int, title: str, desc_text: str) -> np.ndarray:
    v = get_cached_embedding(s, "career", career_id)
    if v is not None: return v
    text = f"{title}. {desc_text}".strip()
    vec = embed_texts([text])[0]
    upsert_embedding("career", career_id, vec, s=s)
    return vec


_STOP = {"a", "an", "the", "and", "of", "to", "for", "with", "on", "by", "at", "from", "or", "in"}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+\-_.#]*")

def _skill_tokens(text: str) -> set[str]:
    toks = {t.lower() for t in _TOKEN_RE.findall(text or "")}
    return {t for t in toks if t not in _STOP and len(t) >= 2}

def _overlap_score(user_text: str, title: str, desc: str) -> float:
    A = _skill_tokens(user_text)
    B = _skill_tokens(f"{title}. {desc or ''}")
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


def _apply_sharpen(x: float, tau: float) -> float:
    return float(max(x, 0.0) ** max(tau, 1.0))


def _user_text(s: Session, user_id, headline: str, use_resume: bool) -> str:
    if use_resume:
        res = s.execute(select(Resume).where(Resume.user_id == str(user_id))).scalars().first()
        if res and res.raw_text and len(res.raw_text.strip()) >= 20:
            return res.raw_text
    return (headline or "").strip()


# ---------- Endpoint ----------
@router.post("/recommend")
def recommend(payload: RecommendIn):
    a = ALPHA if payload.alpha is None else float(payload.alpha)
    b = BETA  if payload.beta  is None else float(payload.beta)
    g = GAMMA if payload.gamma is None else float(payload.gamma)
    t = TAU   if payload.tau   is None else float(payload.tau)

    with SessionLocal() as s:
        text = _user_text(s, payload.user_id, payload.headline, payload.use_resume)
        user_vec = embed_texts([text])[0] if text else None
        survey_vec = _latest_survey_vec(s, payload.user_id)

        careers: List[Career] = s.execute(select(Career)).scalars().all()
        if not careers:
            raise HTTPException(status_code=404, detail="No careers found")

        results = []
        for c in careers:
            content_sim, survey_aff, overlap = 0.0, 0.0, 0.0

            if user_vec is not None:
                c_emb = _ensure_career_emb(s, c.id, c.title, c.description or "")
                content_sim = _cos(user_vec, c_emb)

            if survey_vec is not None:
                c_vec5 = _career_vec5_from_db(c)
                if c_vec5 is not None:
                    survey_aff = _cos(survey_vec, c_vec5)

            if text:
                overlap = _overlap_score(text, c.title, c.description or "")

            raw = a * content_sim + b * survey_aff + g * overlap
            final = _apply_sharpen(raw, t)

            results.append({
                "id": c.id,
                "title": c.title,
                "description": c.description,
                "score": round(final, 4),
                "content_sim": round(content_sim, 4),
                "survey_aff": round(survey_aff, 4),
                "overlap": round(overlap, 4),
                "weights": {"alpha": a, "beta": b, "gamma": g, "tau": t},
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "ok": True,
            "query": "resume" if payload.use_resume else "headline",
            "used_resume": bool(payload.use_resume),
            "items": results[:payload.top_k],
        }
