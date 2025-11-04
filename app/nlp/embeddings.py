# app/nlp/embeddings.py
import os, numpy as np
from sqlalchemy import insert, delete
from sentence_transformers import SentenceTransformer
from app.db.base import Base
from app.db.models_extra import Embedding
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

DB_URL = os.getenv("DB_URL", "sqlite+aiosqlite:///./aspira_ai_db.db").replace("aiosqlite","pysqlite")
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(engine)

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts: list[str], normalize=True) -> np.ndarray:
    X = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize)
    return X.astype(np.float32)

def upsert_embedding(ref_type: str, ref_id: int, vec: np.ndarray, model: str = MODEL_NAME):
    if vec.ndim == 2 and vec.shape[0] == 1:
        vec = vec[0]
    data = vec.tobytes()
    with SessionLocal() as s:
        s.execute(delete(Embedding).where(Embedding.ref_type==ref_type, Embedding.ref_id==ref_id))
        s.execute(insert(Embedding).values(ref_type=ref_type, ref_id=ref_id, model=model, dim=int(len(vec)), vector=data))
        s.commit()
