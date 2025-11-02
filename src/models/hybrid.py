# src/models/hybrid.py
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

P_TEXT = Path("models/role_centroids.parquet")            # cols: [canonical_role, centroid]
P_TRAIT = Path("models/role_trait_centroids.parquet")     # cols: [canonical_role, trait_vec]

# ---------- Loading ----------
def _to_unit_rows(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype("float32", copy=False)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return mat / n

def load_text_centroids():
    df = pd.read_parquet(P_TEXT)
    M = np.stack(df["centroid"].values).astype("float32")
    M = _to_unit_rows(M)
    return df.reset_index(drop=True), M

def load_trait_centroids():
    if not P_TRAIT.exists():
        return None, None
    df = pd.read_parquet(P_TRAIT)
    # Bulletproof clean (empty/None/NaN → zeros)
    def clean(v):
        a = np.asarray(v, dtype="float32").ravel()
        if a.size == 0:
            a = np.zeros(2, dtype="float32")
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.linalg.norm(a)
        return (a / n).astype("float32") if n > 0 else a
    df["trait_vec"] = df["trait_vec"].apply(clean)
    M = np.stack(df["trait_vec"].values).astype("float32")
    M = _to_unit_rows(M)
    return df.reset_index(drop=True), M

# ---------- Encoders ----------
_model_singleton = None
def get_text_model():
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_singleton

def encode_text(query: str) -> np.ndarray:
    m = get_text_model()
    q = m.encode([query or ""], normalize_embeddings=True)
    return q.astype("float32")[0]

# q_trait is expected to be already normalized (or zero). For now we allow None/zero → fallback.

# ---------- Scoring ----------
def cosine(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    # M and v are assumed unit-normalized; still defend against zeros.
    v = v.astype("float32", copy=False)
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.zeros(M.shape[0], dtype="float32")
    return (M @ v).astype("float32")

def hybrid_score(M_text, M_trait, q_text, q_trait=None, alpha: float = 0.7) -> np.ndarray:
    """
    alpha in [0,1]: weight for text. (1-alpha) is weight for traits.
    If traits are missing or zero, fall back to text-only.
    """
    sims_text = cosine(M_text, q_text)
    use_traits = (
        (M_trait is not None)
        and (q_trait is not None)
        and np.linalg.norm(q_trait) > 0
        and M_trait.shape[0] == M_text.shape[0]
    )
    if not use_traits:
        return sims_text  # fallback

    q_trait = q_trait.astype("float32", copy=False)
    nq = np.linalg.norm(q_trait)
    if nq > 0:
        q_trait = q_trait / nq
    sims_trait = (M_trait @ q_trait).astype("float32")

    return alpha * sims_text + (1.0 - alpha) * sims_trait
