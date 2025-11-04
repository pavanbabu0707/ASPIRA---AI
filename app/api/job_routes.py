# app/api/job_routes.py
import os, json
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select, insert, delete
from sqlalchemy.orm import sessionmaker

from app.db.models_extra import Job, Embedding, Resume
from app.models.survey import SurveyResponse as Survey
from app.nlp.embeddings import embed_texts, upsert_embedding

DB_URL = os.getenv("DB_URL", "sqlite+aiosqlite:///./aspira_ai_db.db").replace("aiosqlite","pysqlite")
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)

ALPHA = float(os.getenv("W_ALPHA", 0.55))  # content similarity (resume/headline ↔ text)
BETA  = float(os.getenv("W_BETA",  0.25))  # survey affinity (vector ↔ job.skills vector if present)

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
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _get_latest_survey_vec(s, user_id):
    from sqlalchemy import desc
    try_ids = [str(user_id)]
    try:
        try_ids.append(int(user_id))
    except:  # noqa: E722
        pass
    for uid in try_ids:
        q = s.execute(
            select(Survey).where(Survey.user_id == uid).order_by(
                desc(getattr(Survey, "created_at", getattr(Survey, "id")))
            )
        ).scalars().first()
        if q:
            # tolerate field names
            for attr in ("answers","responses","response_vector","vector"):
                val = getattr(q, attr, None)
                if val is not None:
                    try:
                        arr = val if isinstance(val, list) else json.loads(val)
                        v = np.array(arr, dtype=np.float32)
                        if v.shape[0] == 5: return v
                    except:  # noqa: E722
                        pass
    return None

def _user_text(s, user_id, headline, use_resume: bool):
    if use_resume:
        res = s.execute(select(Resume).where(Resume.user_id == str(user_id))).scalars().first()
        if res and res.raw_text and len(res.raw_text.strip()) >= 20:
            return res.raw_text
    return (headline or "").strip()

def _ensure_job_emb(s, job_id: int, title: str, desc: str) -> np.ndarray:
    e = s.execute(select(Embedding).where(Embedding.ref_type=="job", Embedding.ref_id==job_id)).scalars().first()
    if e:
        return np.frombuffer(e.vector, dtype=np.float32)
    txt = f"{title}. {desc}".strip()
    X = embed_texts([txt])  # (1,dim)
    upsert_embedding("job", job_id, X)
    return X[0]

# ---------- Endpoints ----------
@router.post("/ingest")
def ingest_jobs(payload: BulkJobsIn):
    if not payload.jobs:
        raise HTTPException(status_code=400, detail="No jobs provided")
    with SessionLocal() as s:
        created = 0
        for j in payload.jobs:
            res = s.execute(insert(Job).values(
                title=j.title,
                company=j.company or "",
                location=j.location or "",
                description=j.description,
                skills=json.dumps(j.skills or []),
                source=j.source or "manual",
            ))
            job_id = res.inserted_primary_key[0]
            # precompute embedding
            vec = embed_texts([f"{j.title}. {j.description}"])[0]
            upsert_embedding("job", job_id, vec)
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
        survey_vec = _get_latest_survey_vec(s, payload.user_id)

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
            if survey_vec is not None:
                # optional: if you later store a 5-dim vector for jobs, compute cosine here.
                pass

            score = ALPHA*content_sim + BETA*survey_aff
            scored.append({
                "id": j.id, "title": j.title, "company": j.company, "location": j.location,
                "description": j.description, "score": round(float(score),4),
                "content_sim": round(float(content_sim),4), "survey_aff": round(float(survey_aff),4),
                "skills": j.skills
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"ok": True, "items": scored[:max(1, int(payload.top_k))]}
