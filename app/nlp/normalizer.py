# app/nlp/normalizer.py
from pathlib import Path
import csv

# Optional ontology at artifacts/ontologies/skills.csv
# Columns: alias,canonical
_DEFAULT_ALIASES = {
    "py": "python", "python3": "python", "pytorch": "pytorch", "tf": "tensorflow",
    "js": "javascript", "nodejs": "node.js", "ts": "typescript",
    "sql": "sql", "mysql": "mysql", "postgres": "postgresql", "postgresql": "postgresql",
    "spark": "spark", "airflow": "airflow",
    "aws": "aws", "azure": "azure", "gcp": "gcp",
    "mlops": "mlops", "ml": "machine learning", "machine learning": "machine learning",
    "nlp": "nlp", "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
    "docker": "docker", "kubernetes": "kubernetes",
}

def _load_alias_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    m: dict[str,str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alias = (row.get("alias") or "").strip().lower()
            canon = (row.get("canonical") or "").strip().lower()
            if alias and canon:
                m[alias] = canon
    return m

def normalize_skills(raw_tokens: list[str]) -> list[str]:
    """Map aliases → canonical skills; lowercase; de-dup."""
    path = Path("artifacts/ontologies/skills.csv")
    alias_map = {**_DEFAULT_ALIASES, **_load_alias_csv(path)}
    out = []
    seen = set()
    for t in raw_tokens:
        a = t.strip().lower()
        if not a: 
            continue
        canon = alias_map.get(a, a)
        if canon not in seen and len(canon) >= 2:
            seen.add(canon)
            out.append(canon)
    return out
