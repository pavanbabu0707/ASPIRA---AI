# app/api/job_routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy import select, delete
from app.db.session import get_db, SessionLocal
from app.db.models_extra import Job
from app.nlp.embeddings import embed_texts, upsert_embedding, get_cached_embedding
import numpy as np
import os, json, re

router = APIRouter(prefix="/jobs", tags=["jobs"])

# ---------------------------
# Schemas
# ---------------------------

class JobIn(BaseModel):
    title: str
    company: Optional[str] = ""
    location: Optional[str] = ""
    description: str
    skills: List[str] = []
    source: Optional[str] = "manual"

class JobsIn(BaseModel):
    jobs: List[JobIn]

class JobsRecommendIn(BaseModel):
    user_id: str | int
    headline: str = Field("", description="Used if no resume")
    top_k: int = 10
    use_resume: bool = True
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    tau: Optional[float] = None


# ---------------------------
# Helper functions
# ---------------------------

ALPHA = float(os.getenv("WJ_ALPHA", "0.70"))
BETA  = float(os.getenv("WJ_BETA",  "0.25"))
GAMMA = float(os.getenv("WJ_GAMMA", "0.05"))
TAU   = float(os.getenv("WJ_TAU",   "1.3"))
TEMP  = float(os.getenv("WJ_TEMP",  "1.2"))

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+\-_.#]*")
_STOP = {"a","an","and","the","of","in","on","to","for","with","by","at","from","or","as","is","are","be"}

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2: a = a[0]
    if b.ndim == 2: b = b[0]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _skill_tokens(txt: str) -> set[str]:
    toks = {t.lower() for t in _TOKEN_RE.findall(txt or "")}
    return {t for t in toks if t not in _STOP and len(t) >= 2}

def _overlap(a: str, b: str) -> float:
    A, B = _skill_tokens(a), _skill_tokens(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def _apply_sharpen(x: float, tau: float) -> float:
    return float(max(x, 0.0) ** max(tau, 1.0))


# ---------------------------
# Routes
# ---------------------------

@router.post("/ingest")
def ingest_jobs(payload: JobsIn):
    """Ingest or update job postings and cache embeddings."""
    with SessionLocal() as s:
        inserted = 0
        for j in payload.jobs:
            job = s.execute(select(Job).where(Job.title == j.title, Job.company == j.company)).scalars().first()
            if not job:
                job = Job(title=j.title, company=j.company, location=j.location,
                          description=j.description, skills=",".join(j.skills), source=j.source)
                s.add(job)
                s.flush()  # get job.id
                inserted += 1
            else:
                job.location = j.location
                job.description = j.description
                job.skills = ",".join(j.skills)
                job.source = j.source

            # Create or update embedding for this job
            text = f"{job.title}. {job.description or ''}".strip()
            vec = embed_texts([text])[0]
            upsert_embedding("job", job.id, vec, s=s)

        s.commit()
        return {"ok": True, "inserted_or_updated": inserted, "total": len(payload.jobs)}


# --- Hybrid job recommendations ---
@router.post("/recommend")
def recommend_jobs(payload: JobsRecommendIn):
    """Hybrid job recommender (resume/headline text + overlap)."""
    from app.db.models_extra import Resume

    a = ALPHA if payload.alpha is None else float(payload.alpha)
    b = BETA  if payload.beta  is None else float(payload.beta)
    g = GAMMA if payload.gamma is None else float(payload.gamma)
    t = TAU   if payload.tau   is None else float(payload.tau)

    with SessionLocal() as s:
        # --- choose user text ---
        user_text = ""
        if payload.use_resume:
            res = s.execute(select(Resume).where(Resume.user_id == str(payload.user_id))).scalars().first()
            if res and res.raw_text and len(res.raw_text.strip()) >= 20:
                user_text = res.raw_text
        if not user_text:
            user_text = (payload.headline or "").strip()

        user_vec = embed_texts([user_text])[0] if user_text else None

        jobs: List[Job] = s.execute(select(Job)).scalars().all()
        if not jobs:
            raise HTTPException(status_code=404, detail="No jobs ingested")

        out = []
        for j in jobs:
            cached = get_cached_embedding(s, "job", j.id)
            if cached is None:
                text = f"{j.title}. {j.description or ''}".strip()
                vec = embed_texts([text])[0]
                upsert_embedding("job", j.id, vec, s=s)
                cached = vec

            content_sim = _cos(user_vec, cached) if user_vec is not None else 0.0
            overlap = _overlap(user_text, f"{j.title}. {j.description or ''}") if user_text else 0.0

            raw = a * content_sim + g * overlap
            final = _apply_sharpen(raw, t)

            out.append({
                "id": j.id,
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "score": float(final),
                "content_sim": round(float(content_sim), 4),
                "overlap": round(float(overlap), 4),
                "source": j.source,
                "description": j.description,
            })

        # normalize + softmax
        scores = np.array([r["score"] for r in out], dtype=np.float32)
        mn, mx = float(scores.min()), float(scores.max())
        norm = (scores - mn) / (mx - mn) if mx > mn else np.ones_like(scores)
        exps = np.exp((scores - scores.max()) / max(TEMP, 1e-6))
        conf = exps / exps.sum()

        for i, r in enumerate(out):
            r["norm_score"] = float(round(norm[i], 4))
            r["confidence"] = float(round(conf[i], 4))
            r["score"] = float(round(r["score"], 4))

        out.sort(key=lambda x: x["score"], reverse=True)
        return {
            "ok": True,
            "used_resume": bool(payload.use_resume and user_text),
            "alpha": a,
            "beta": b,
            "gamma": g,
            "tau": t,
            "items": out[:max(1, payload.top_k)]
        }
