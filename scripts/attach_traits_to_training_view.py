# scripts/attach_traits_to_training_view.py
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = Path("data/features")
TV_PATH  = FEATURES / "training_view.parquet"

# your canonical trait names (adjust if you use different ones)
TRAIT_NAMES = [
    "analytical","creativity","leadership","resilience",
    "impact","growth","collaboration","execution",
]

def _is_trait_col(name: str) -> bool:
    n = name.strip().lower()
    return (n in TRAIT_NAMES) or n.startswith("trait_") or n.endswith("_trait")

def ensure_trait_vec(df: pd.DataFrame) -> pd.DataFrame:
    # If trait_vec looks array-like already, keep it
    if "trait_vec" in df.columns:
        try:
            _ = np.asarray(df["trait_vec"].iloc[0])
            return df
        except Exception:
            pass  # rebuild below

    # Detect numeric trait columns
    cand = [c for c in df.columns if _is_trait_col(str(c))]
    trait_cols = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]
    if trait_cols:
        # Ordered: preferred names first, then the rest
        ordered = [c for c in TRAIT_NAMES if c in trait_cols] + [c for c in trait_cols if c not in TRAIT_NAMES]
        df["trait_vec"] = df[ordered].apply(lambda r: r.values.astype("float32"), axis=1)
        print("[traits-fix] using trait columns:", ordered)
    else:
        # Make zeros
        for n in TRAIT_NAMES:
            if n not in df.columns:
                df[n] = 0.0
        df["trait_vec"] = df[TRAIT_NAMES].apply(lambda r: r.values.astype("float32"), axis=1)
        print("[traits-fix] no numeric trait cols found; created zero-vectors.")

    return df

def main():
    if not TV_PATH.exists():
        raise FileNotFoundError(f"Missing {TV_PATH} — build features first.")
    df = pd.read_parquet(TV_PATH)
    print(f"[traits-fix] loaded training_view: {df.shape}")
    df = ensure_trait_vec(df)
    df.to_parquet(TV_PATH, index=False)
    print(f"[traits-fix] wrote {TV_PATH} with trait_vec. Rows={len(df)}")

if __name__ == "__main__":
    main()
