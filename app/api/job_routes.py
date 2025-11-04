# app/api/job_routes.py
import json
import os
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models_extra import Job, Embedding, Resume
from app.models.survey import SurveyResponse as Survey
from app.nlp.embeddings import embed_texts, upsert_embedding, get_cached_embedding

ALPHA = float(os.getenv("W_ALPHA", 0.55))  # content (resume/headline ↔ job text)
BETA  = float(os.getenv("W_BETA",  0.25))  # survey ↔ job vector (reserved for future if you add a 5-dim vec on Job)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# ---------- Schemas ----------
class JobIn(BaseModel):
    title: str
    company: Optional[str] = ""
    location: Optional[str] = ""
    description: str
    skills: Optional[List[str]] = []
    source: Optional[str] = "manual"

class BulkJobsIn(BaseModel):
    jobs: List[JobIn]

class RecommendIn(BaseModel):
    user_id: str | int
    headline: Optional[str] = ""
    top_k: int = 10
    use_resume: bool = True

# ---------- Helpers ----------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2: a = a[0]
    if b.ndim == 2: b = b[0]
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _latest_survey_vec(s: Session, user_id) -> np.ndarray | None:
    from sqlalchemy import desc
    # try both string and int user_id
    candidates = [str(user_id)]
    try:
        candidates.append(int(user_id))
    except Exception:
        pass

    for uid in candidates:
        q = s.execute(
            select(Survey)
            .where(Survey.user_id == uid)
            .order_by(desc(getattr(Survey, "created_at", getattr(Survey, "id"))))
        ).scalars().first()
        if not q:
            continue

        # tolerate multiple field names
        for attr in ("answers", "responses", "response_vector", "vector"):
            val = getattr(q, attr, None)
            if val is None:
                continue
            try:
                arr = val if isinstance(val, list) else json.loads(val)
                v = np.array(arr, dtype=np.float32)
                if v.shape[0] == 5:
                    return v
            except Exception:
                continue
    return None

def _user_text(s: Session, user_id, headline: str, use_resume: bool) -> str:
    if use_resume:
        res = s.execute(select(Resume).where(Resume.user_id == str(user_id))).scalars().first()
        if res and res.raw_text and len(res.raw_text.strip()) >= 20:
            return res.raw_text
    return (headline or "").strip()

def _ensure_job_emb(s: Session, job_id: int, title: str, desc: str) -> np.ndarray:
    vec = get_cached_embedding(s, "job", job_id)
    if vec is not None:
        return vec
    text = f"{title}. {desc}".strip()
    vec = embed_texts([text])[0]
    upsert_embedding("job", job_id, vec, s=s)  # use SAME session
    return vec

# ---------- Endpoints ----------
@router.post("/ingest")
def ingest_jobs(payload: BulkJobsIn):
    if not payload.jobs:
        raise HTTPException(status_code=400, detail="No jobs provided")

    with SessionLocal() as s:
        created = 0
        for j in payload.jobs:
            # Insert only columns that exist on your Job table
            row = {
                "title": j.title,
                "company": j.company or "",
                "location": j.location or "",
                "description": j.description,
            }
            if hasattr(Job, "skills"):
                row["skills"] = json.dumps(j.skills or [])
            if hasattr(Job, "source"):
                row["source"] = j.source or "manual"

            res = s.execute(insert(Job).values(**row))
            job_id = int(res.inserted_primary_key[0])
            s.flush()  # make the PK visible within txn

            # Precompute and store embedding using SAME session
            _ensure_job_emb(s, job_id, j.title, j.description)
            created += 1

        s.commit()
    return {"ok": True, "created": created}

@router.get("")
def list_jobs():
    with SessionLocal() as s:
        rows = s.execute(select(Job)).scalars().all()
        return [{"id": r.id, "title": r.title, "company": r.company, "location": r.location} for r in rows]

@router.post("/recommend")
def recommend_jobs(payload: RecommendIn):
    with SessionLocal() as s:
        txt = _user_text(s, payload.user_id, payload.headline, payload.use_resume)
        user_vec = embed_texts([txt])[0] if txt else None
        survey_vec = _latest_survey_vec(s, payload.user_id)

        jobs = s.execute(select(Job)).scalars().all()
        if not jobs:
            raise HTTPException(status_code=404, detail="No jobs found")

        scored = []
        for j in jobs:
            content_sim = 0.0
            if user_vec is not None:
                jv = _ensure_job_emb(s, j.id, j.title, j.description or "")
                content_sim = _cos(user_vec, jv)

            survey_aff = 0.0
            # (future) if you add a 5-dim job vector, compute cosine(survey_vec, job_vec) here

            score = ALPHA * content_sim + BETA * survey_aff
            scored.append({
                "id": j.id,
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "description": j.description,
                "score": round(float(score), 4),
                "content_sim": round(float(content_sim), 4),
                "survey_aff": round(float(survey_aff), 4),
                "skills": getattr(j, "skills", None),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        k = max(1, int(payload.top_k))
        return {"ok": True, "items": scored[:k]}
