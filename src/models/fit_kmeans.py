# src/models/fit_kmeans.py
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def to_vec(x):
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32)
    return x

def main(n_clusters: int = 5, random_state: int = 42):
    df = pd.read_parquet(FEATURES / "training_view.parquet")
    df = df.dropna(subset=["text_vec"])
    df["text_vec"] = df["text_vec"].apply(to_vec)

    X = np.stack(df["text_vec"].values).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    km.fit(X)

    out = MODELS / "kmeans.pkl"
    with open(out, "wb") as f:
        pickle.dump({"kmeans": km}, f)

    print(f"Saved kmeans.pkl with {n_clusters} clusters")

if __name__ == "__main__":
    main()
