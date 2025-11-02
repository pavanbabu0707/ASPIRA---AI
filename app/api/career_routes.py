# app/api/career_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.session import get_db
from app.models.career import Career

router = APIRouter(prefix="/careers", tags=["Careers"])

@router.get("", summary="List all careers")
def list_careers(db: Session = Depends(get_db)):
    rows = db.execute(select(Career)).scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No careers found")
    return [
        {
            "id": c.id,
            "onet_code": c.onet_code,
            "title": c.title,
            "description": c.description,
            "skills_vector": c.skills_vector,
        }
        for c in rows
    ]
