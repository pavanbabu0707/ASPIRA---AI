# scripts/build_role_centroids.py
from pathlib import Path
import numpy as np
import pandas as pd

INP = Path("data/features/training_view.parquet")
OUT = Path("models/role_centroids.parquet")

def to_vec(x):
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32)
    return x

def main():
    df = pd.read_parquet(INP)
    df = df.dropna(subset=["canonical_role", "text_vec"])
    df["text_vec"] = df["text_vec"].apply(to_vec)

    rows = []
    for role, grp in df.groupby("canonical_role"):
        vecs = np.stack(grp["text_vec"].values, axis=0)
        centroid = vecs.mean(axis=0)
        centroid = (centroid / (np.linalg.norm(centroid) + 1e-9)).astype(np.float32)
        rows.append({"canonical_role": role, "centroid": centroid})

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(out_df)} roles")

if __name__ == "__main__":
    main()
