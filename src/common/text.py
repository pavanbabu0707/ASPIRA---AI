from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None

def get_sbert(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL

def embed_text(texts: List[str]) -> np.ndarray:
    model = get_sbert()
    return np.array(model.encode(texts, normalize_embeddings=True))
