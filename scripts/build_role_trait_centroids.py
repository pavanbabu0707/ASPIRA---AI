# scripts/build_role_trait_centroids.py  (safe version)
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = Path("data/features/training_view.parquet")
OUT = Path("models/role_trait_centroids.parquet")

def row_trait_vec(df, trait_cols):
    A = df[trait_cols].astype("float32").fillna(0.0).to_numpy(copy=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return (A / n).astype("float32")

def main():
    tv = pd.read_parquet(FEATURES)
    trait_cols = [c for c in tv.columns if c.startswith("trait_")]
    if not trait_cols:
        raise RuntimeError("No trait_* columns found in training_view.parquet")

    tv = tv.copy()
    tv["trait_vec"] = list(row_trait_vec(tv, trait_cols))

    def mean_vec(g):
        arrs = np.stack(g["trait_vec"].values)
        m = arrs.mean(axis=0)
        n = np.linalg.norm(m)
        return (m / n).astype("float32") if n > 0 else m

    out = (tv.groupby("canonical_role", as_index=False)
             .apply(lambda g: pd.Series({"trait_vec": mean_vec(g)}))
             .reset_index(drop=True))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Saved: {OUT} (rows={len(out)}, dim={len(out['trait_vec'].iloc[0])})")

if __name__ == "__main__":
    main()
