from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INP = Path("data/features/training_view.parquet")
OUT = INP  # overwrite same file

SO_SKILL_COLS = [
    "LanguageHaveWorkedWith","DatabaseHaveWorkedWith","PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith","MiscTechHaveWorkedWith","ToolsTechHaveWorkedWith",
    "LanguageWantToWorkWith","DatabaseWantToWorkWith","PlatformWantToWorkWith",
    "WebframeWantToWorkWith","MiscTechWantToWorkWith","ToolsTechWantToWorkWith",
]

def build_free_text(row: pd.Series) -> str:
    bits = []
    # strong identity
    val = str(row.get("canonical_role") or "").strip()
    if val: bits.append(val)
    else:
        raw = str(row.get("raw_title") or "").strip()
        if raw: bits.append(raw)

    # questionnaire
    edu = str(row.get("education_level") or "").strip()
    exp = str(row.get("exp_years") or "").strip()
    cty = str(row.get("country") or "").strip()
    if edu: bits.append(f"education: {edu}")
    if exp: bits.append(f"experience: {exp} yrs")
    if cty: bits.append(f"country: {cty}")

    # skills (only if present)
    for col in SO_SKILL_COLS:
        if col in row.index:
            v = str(row.get(col) or "").strip()
            if v:
                bits.append(f"{col}: {v}")

    text = " | ".join(bits).strip()
    return text if text else "unknown role"

def main():
    df = pd.read_parquet(INP)
    if "raw_title" not in df.columns:
        df["raw_title"] = pd.NA

    df["free_text"] = df.apply(build_free_text, axis=1)
    uniq = df["free_text"].nunique()
    print(f"Distinct free_text values: {uniq}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(df["free_text"].tolist(), normalize_embeddings=True)
    df["text_vec"] = [v.astype(np.float32) for v in vecs]

    df.to_parquet(OUT, index=False)
    print(f"✅ Rewrote {OUT} with real text embeddings")

if __name__ == "__main__":
    main()
