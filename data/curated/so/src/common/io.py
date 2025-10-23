from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE = Path(".")
DATA = BASE / "data"
RAW = DATA / "raw"
STAGED = DATA / "staged"
CURATED = DATA / "curated"
FEATURES = DATA / "features"
MODELS = Path("models")

for p in [DATA, RAW, STAGED, CURATED, FEATURES, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
