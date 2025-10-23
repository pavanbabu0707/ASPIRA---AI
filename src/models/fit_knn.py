# src/models/fit_knn.py
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def to_vec(x):
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32)
    return x

def main():
    df = pd.read_parquet(FEATURES / "training_view.parquet")
    # keep rows with embeddings
    df = df.dropna(subset=["text_vec"])
    df["text_vec"] = df["text_vec"].apply(to_vec)

    X = np.stack(df["text_vec"].values).astype(np.float32)
    # sklearn cosine metric expects raw vectors; we already normalized in reembed_text.py, but safe:
    # (Optional) X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    roles = df["canonical_role"].fillna("Unknown").astype(str).tolist()

    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(X)

    out = MODELS / "knn_index.pkl"
    with open(out, "wb") as f:
        pickle.dump({"nn": nn, "roles": roles}, f)

    print(f"Saved knn_index.pkl for {len(roles)} roles")

if __name__ == "__main__":
    main()
