# app/db/models_extra.py
from sqlalchemy import Column, Integer, String, Text, LargeBinary
from app.db.base import Base

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, unique=True, nullable=False)
    raw_text = Column(Text, nullable=False)
    skills = Column(Text)
    entities = Column(Text)

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    title = Column(String, index=True, nullable=False)
    company = Column(String, index=True)
    location = Column(String)
    description = Column(Text, nullable=False)
    skills = Column(Text)
    source = Column(String)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    ref_type = Column(String, index=True, nullable=False)  # 'career' | 'job' | 'resume'
    ref_id = Column(Integer, index=True, nullable=False)
    model = Column(String, nullable=False)
    dim = Column(Integer, nullable=False)
    vector = Column(LargeBinary, nullable=False)
