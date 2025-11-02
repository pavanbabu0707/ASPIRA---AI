# scripts/rebuild_centroids_strict.py
from pathlib import Path
import ast
import numpy as np
import pandas as pd

FEATURES = Path("data/features/training_view.parquet")
OUT = Path("models/role_centroids.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

EMB_DIM = 384
MIN_ROW_NORM = 1e-6   # drop rows with ~zero vectors

def coerce_vec(v):
    """
    Convert a value (list/np.ndarray/str) to a clean float32 vector.
    - If it's a string like "[0.1, 0.2, ...]", parse it.
    - If dim mismatches, return zeros.
    - Replace NaNs/inf with 0.
    """
    if v is None:
        return np.zeros(EMB_DIM, dtype="float32")

    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return np.zeros(EMB_DIM, dtype="float32")

    arr = np.asarray(v, dtype="float32").ravel()
    if arr.size != EMB_DIM:
        # try to pad/truncate defensively
        if arr.size == 0:
            return np.zeros(EMB_DIM, dtype="float32")
        if arr.size > EMB_DIM:
            arr = arr[:EMB_DIM]
        else:
            pad = np.zeros(EMB_DIM - arr.size, dtype="float32")
            arr = np.concatenate([arr, pad])
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def l2norm(a):
    n = np.linalg.norm(a)
    return a / n if n > 0 else a

def main():
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing {FEATURES}")

    df = pd.read_parquet(FEATURES)
    needed = {"canonical_role", "text_vec"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{FEATURES} must contain columns: {sorted(needed)}")

    # Coerce all rows to numeric vectors
    df = df[["canonical_role", "text_vec"]].copy()
    df["text_vec"] = df["text_vec"].apply(coerce_vec)

    # Drop rows with ~zero vectors
    norms = df["text_vec"].apply(lambda x: float(np.linalg.norm(x)))
    kept = norms > MIN_ROW_NORM
    dropped = int((~kept).sum())
    df = df.loc[kept].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("All rows were dropped (no valid embeddings). Check your training_view.parquet.")

    # Group and mean per role
    grp = (
        df.groupby("canonical_role", as_index=False)["text_vec"]
          .apply(lambda s: np.mean(np.stack(s.values), axis=0))
    )
    # Normalize each centroid
    grp["centroid"] = grp["text_vec"].apply(lambda v: l2norm(np.asarray(v, dtype="float32")))
    grp = grp[["canonical_role", "centroid"]].sort_values("canonical_role").reset_index(drop=True)

    # Diagnostics
    c_norms = np.array([np.linalg.norm(x) for x in grp["centroid"].values], dtype="float32")
    zero_roles = (c_norms <= MIN_ROW_NORM).sum()
    print(f"[rebuild] from file: {FEATURES}")
    print(f"[rebuild] dropped rows with ~zero vectors: {dropped}")
    print(f"[rebuild] roles: {len(grp)}, centroid norm min/max/mean: "
          f"{c_norms.min():.6f}/{c_norms.max():.6f}/{c_norms.mean():.6f}")
    if zero_roles:
        bad = grp.loc[c_norms <= MIN_ROW_NORM, "canonical_role"].tolist()
        print(f"[rebuild] WARNING: {zero_roles} roles have ~zero centroids: {bad[:5]}{' ...' if len(bad)>5 else ''}")

    grp.to_parquet(OUT, index=False)
    print(f"Saved: {OUT} (rows={len(grp)})")

if __name__ == "__main__":
    main()
