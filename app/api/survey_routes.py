# app/api/survey_routes.py
from typing import List
import json
import math
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from app.db.session import get_db
from app.models.survey import SurveyResponse
from app.models.career import Career
from app.schemas.survey import SurveySubmit, SurveyOut

router = APIRouter(prefix="/survey", tags=["Survey"])

def _to_vec(x) -> List[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    if isinstance(x, str):
        return [float(v) for v in json.loads(x)]
    raise ValueError("Answers/skills_vector must be list or JSON-encoded list")

def cosine(a: List[float], b: List[float]) -> float:
    # safe cosine
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

@router.post("/submit", response_model=SurveyOut, status_code=201)
def submit_survey(payload: SurveySubmit, db: Session = Depends(get_db)):
    row = SurveyResponse(user_id=payload.user_id, answers=json.dumps(payload.answers))
    db.add(row)
    db.commit()
    db.refresh(row)
    return SurveyOut(id=row.id, user_id=row.user_id, answers=row.answers)

@router.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = Query(3, ge=1, le=20), db: Session = Depends(get_db)):
    # 1) latest survey for user
    survey = db.execute(
        select(SurveyResponse).where(SurveyResponse.user_id == user_id).order_by(desc(SurveyResponse.created_at))
    ).scalars().first()
    if not survey:
        raise HTTPException(status_code=404, detail="No survey found for user")

    user_vec = _to_vec(survey.answers)

    # 2) all careers
    careers = db.execute(select(Career)).scalars().all()
    if not careers:
        raise HTTPException(status_code=404, detail="No careers available")

    # 3) cosine similarity ranking
    scored = []
    for c in careers:
        c_vec = _to_vec(c.skills_vector)
        scored.append({
            "career": c.title,
            "onet_code": c.onet_code,
            "score": round(cosine(user_vec, c_vec), 4),
        })

    # 4) sort by score desc and return top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": scored[:top_k],
    }

@router.get("/user/{user_id}", summary="List surveys for a user (newest first)")
def list_user_surveys(user_id: int, db: Session = Depends(get_db)):
    rows = db.execute(
        select(SurveyResponse)
        .where(SurveyResponse.user_id == user_id)
        .order_by(desc(SurveyResponse.created_at))
    ).scalars().all()
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "answers": r.answers,
            "created_at": r.created_at,
        }
        for r in rows
    ]
