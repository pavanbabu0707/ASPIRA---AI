# app/db/seed.py
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.career import Career

SEED_ROWS = [
    ("15-1252.00", "Software Developer",
     "Develops and maintains software applications across platforms.",
     [0.9, 0.8, 0.2, 0.3, 0.1]),
    ("15-2051.00", "Data Scientist",
     "Builds models, analyzes data, and communicates insights to stakeholders.",
     [0.6, 0.9, 0.7, 0.4, 0.2]),
    ("13-2099.01", "Financial Quant Analyst",
     "Designs quantitative strategies, risk models, and conducts backtests.",
     [0.4, 0.7, 0.9, 0.6, 0.8]),
    ("15-1299.02", "Machine Learning Engineer",
     "Deploys and optimizes ML systems on cloud and edge devices.",
     [0.85, 0.85, 0.5, 0.4, 0.3]),
]

def seed_careers(db: Session) -> int:
    # if already seeded, skip
    existing = db.execute(select(Career)).scalars().first()
    if existing:
        return 0
    for onet, title, desc, vec in SEED_ROWS:
        db.add(Career(onet_code=onet, title=title, description=desc, skills_vector=str(vec)))
    db.commit()
    return len(SEED_ROWS)
