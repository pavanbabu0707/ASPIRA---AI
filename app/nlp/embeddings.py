# app/nlp/embeddings.py
import numpy as np
from sqlalchemy import delete, insert, select
from sentence_transformers import SentenceTransformer

from app.db.session import SessionLocal
from app.db.models_extra import Embedding

# ---------- Model cache ----------
_model = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

# Column discovery so we only insert what exists
_EMBED_COLS = {c.name for c in Embedding.__table__.columns}

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
        # Delete any existing rows for this ref (all models)
        sess.execute(
            delete(Embedding).where(
                Embedding.ref_type == ref_type,
                Embedding.ref_id == ref_id,
            )
        )
        # Prepare row with optional columns if they exist
        row = {
            "ref_type": ref_type,
            "ref_id": ref_id,
            "vector": np.asarray(vec, dtype=np.float32).tobytes(),
            # Optional columns commonly present in schemas
            "model": _MODEL_NAME,
            "dim": int(vec.shape[-1]) if hasattr(vec, "shape") else None,
            "normalized": True,
        }
        row = {k: v for k, v in row.items() if k in _EMBED_COLS}

        sess.execute(insert(Embedding).values(**row))

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
