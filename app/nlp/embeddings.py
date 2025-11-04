# app/nlp/embeddings.py
import numpy as np
from sqlalchemy import delete, insert, select
from sentence_transformers import SentenceTransformer

from app.db.session import SessionLocal
from app.db.models_extra import Embedding

# ---------- Model cache ----------
_model = None

def get_model():
    global _model
    if _model is None:
        # Small + fast + good: all-MiniLM-L6-v2
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

# ---------- Public helpers ----------
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Returns np.float32 array of shape (N, D), L2-normalized.
    """
    M = get_model()
    X = M.encode(texts, normalize_embeddings=True)
    return np.asarray(X, dtype=np.float32)

def upsert_embedding(ref_type: str, ref_id: int, vec: np.ndarray, s=None) -> None:
    """
    Upsert an embedding row. IMPORTANT: pass the **caller session** `s`
    when writing inside a request/transaction to avoid SQLite write locks.
    Falls back to its own short-lived session only if `s` is None.
    """
    def _do(sess):
        sess.execute(
            delete(Embedding).where(
                Embedding.ref_type == ref_type,
                Embedding.ref_id == ref_id,
            )
        )
        sess.execute(
            insert(Embedding).values(
                ref_type=ref_type,
                ref_id=ref_id,
                vector=np.asarray(vec, dtype=np.float32).tobytes(),
            )
        )

    if s is not None:
        _do(s)
    else:
        with SessionLocal() as s2:
            _do(s2)
            s2.commit()

def get_cached_embedding(s, ref_type: str, ref_id: int) -> np.ndarray | None:
    """
    Returns np.float32 vector if present, else None.
    """
    row = s.execute(
        select(Embedding).where(
            Embedding.ref_type == ref_type,
            Embedding.ref_id == ref_id,
        )
    ).scalars().first()
    if not row:
        return None
    return np.frombuffer(row.vector, dtype=np.float32)
