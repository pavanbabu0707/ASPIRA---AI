# app/api/nlp_routes.py
import json, os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, insert, select
from app.db.models_extra import Resume
from app.nlp.extractors import sniff_and_extract_text, extract_resume_entities
from app.nlp.embeddings import embed_texts, upsert_embedding

DB_URL = os.getenv("DB_URL", "sqlite+aiosqlite:///./aspira_ai_db.db").replace("aiosqlite","pysqlite")
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(engine)

router = APIRouter(prefix="/nlp", tags=["nlp"])

class ResumeIn(BaseModel):
    user_id: str
    raw_text: str

@router.post("/resume/upload")
async def upload_resume(user_id: str = Form(...), file: UploadFile = File(None), raw_text: str = Form(None)):
    if not file and not raw_text:
        raise HTTPException(status_code=400, detail="Provide a file or raw_text")

    if file:
        data = await file.read()
        text = sniff_and_extract_text(file.filename, data)
    else:
        text = raw_text or ""

    text = (text or "").strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Resume text too short or unreadable")

    entities = extract_resume_entities(text)
    skills_json = json.dumps(entities.get("skills", []), ensure_ascii=False)
    entities_json = json.dumps({k:v for k,v in entities.items() if k!="skills"}, ensure_ascii=False)

    with SessionLocal() as s:
        # upsert by user_id
        existing = s.execute(select(Resume.id).where(Resume.user_id==user_id)).scalar()
        if existing:
            s.execute(
                Resume.__table__.update()
                .where(Resume.id==existing)
                .values(raw_text=text, skills=skills_json, entities=entities_json)
            )
            resume_id = existing
        else:
            res = s.execute(insert(Resume).values(user_id=user_id, raw_text=text, skills=skills_json, entities=entities_json))
            resume_id = res.inserted_primary_key[0]
        s.commit()

    # embed and store
    vec = embed_texts([text])
    upsert_embedding(ref_type="resume", ref_id=resume_id, vec=vec)

    return {
        "ok": True,
        "resume_id": resume_id,
        "user_id": user_id,
        "preview": {
            "chars": len(text),
            "skills": entities.get("skills", []),
            "edu": entities.get("edu", []),
            "orgs": entities.get("orgs", []),
            "exp_years": entities.get("exp_years", 0.0),
        }
    }
