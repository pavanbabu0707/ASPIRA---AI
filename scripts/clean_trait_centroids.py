# scripts/clean_trait_centroids.py
from pathlib import Path
import numpy as np
import pandas as pd

P = Path("models/role_trait_centroids.parquet")

def clean_vec(v):
    a = np.asarray(v, dtype="float32")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)  # purge bad values
    n = np.linalg.norm(a)
    return (a / n).astype("float32") if n > 0 else a

def main():
    if not P.exists():
        raise FileNotFoundError(P)
    df = pd.read_parquet(P)
    df["trait_vec"] = df["trait_vec"].apply(clean_vec)
    df.to_parquet(P, index=False)
    print(f"Cleaned and saved: {P} (rows={len(df)})")

if __name__ == "__main__":
    main()
