# scripts/materialize.py
"""
Materialize the starter CareerPath AI repo structure and code.

Usage:
    python scripts/materialize.py
"""

from pathlib import Path

FILES = {
    "Makefile": r"""
.PHONY: setup ingest build_features fit_knn fit_kmeans clean

setup:
\tpython -m venv .venv && \\
\t./.venv/bin/pip install --upgrade pip && \\
\t./.venv/bin/pip install -e .

ingest:
\tpython src/ingest/dummy_check.py  # placeholder for your loaders

build_features:
\tpython src/features/build_training_view.py

fit_knn:
\tpython src/models/fit_knn.py

fit_kmeans:
\tpython src/models/fit_kmeans.py

clean:
\trm -rf dist build *.egg-info
""",
    "pyproject.toml": r"""
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "careerpath-ai"
version = "0.0.1"
description = "CareerPath AI training pipeline"
authors = [{name="Team CareerPath AI"}]
requires-python = ">=3.10"
dependencies = [
  "pandas>=2.2.0",
  "pyarrow>=16.0.0",
  "numpy>=1.26.0",
  "scikit-learn>=1.5.0",
  "tqdm>=4.66.0",
  "sentence-transformers>=3.0.0",
  "faiss-cpu>=1.8.0",
  "matplotlib>=3.8.0"
]

[tool.setuptools.packages.find]
where = ["src"]
""",
    "src/__init__.py": "",
    "src/common/io.py": r"""
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
""",
    "src/common/text.py": r"""
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
""",
    "src/ingest/dummy_check.py": r"""
# Placeholder to verify Makefile wiring.
print("Ingest step placeholder — replace with real loaders (SO/O*NET/CareerCon)")
""",
    "src/features/build_training_view.py": r"""
# Simplified placeholder for building a training view
# Extend this to join SO, O*NET, and CareerCon curated datasets

from pathlib import Path
import pandas as pd
from src.common.io import FEATURES, write_parquet

def main():
    # For now, create a dummy frame
    df = pd.DataFrame({
        "profile_id": [1, 2],
        "canonical_role": ["Data Engineer", "ML Engineer"],
        "satisfaction": [4, 5],
        "text_vec": [[0.1]*384, [0.2]*384]
    })
    write_parquet(df, FEATURES / "training_view.parquet")
    print(f"Saved dummy training_view.parquet with {len(df)} rows")

if __name__ == "__main__":
    main()
""",
    "src/models/fit_knn.py": r"""
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.common.io import FEATURES, MODELS

def main():
    df = pd.read_parquet(FEATURES / "training_view.parquet")
    X = np.vstack(df["text_vec"].to_list())
    roles = df["canonical_role"].tolist()

    nn = NearestNeighbors(metric="cosine", n_neighbors=min(5, len(roles)))
    nn.fit(X)

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "knn_index.pkl", "wb") as f:
        pickle.dump({"nn": nn, "roles": roles}, f)
    print(f"Saved knn_index.pkl for {len(roles)} roles")

if __name__ == "__main__":
    main()
""",
    "src/models/fit_kmeans.py": r"""
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.common.io import FEATURES, MODELS

def main():
    df = pd.read_parquet(FEATURES / "training_view.parquet")
    X = np.vstack(df["text_vec"].to_list())

    k = min(5, len(df))  # simple heuristic
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)

    with open(MODELS / "kmeans.pkl", "wb") as f:
        pickle.dump(km, f)
    print(f"Saved kmeans.pkl with {k} clusters")

if __name__ == "__main__":
    main()
""",
}

def main():
    for path, content in FILES.items():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Wrote: {p}")

if __name__ == "__main__":
    main()
