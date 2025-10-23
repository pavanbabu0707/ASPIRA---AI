# src/features/build_training_view.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "data" / "curated"
FEATURES = ROOT / "data" / "features"
REF = ROOT / "data" / "reference"
FEATURES.mkdir(parents=True, exist_ok=True)

# ---------- IO helpers ----------
def _load(path: Path) -> pd.DataFrame:
    """Read parquet/csv (forcing csv to strings) or return empty df if missing."""
    if path.suffix.lower() == ".parquet" and path.exists():
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv" and path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame()

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _norm_code_series(s: pd.Series) -> pd.Series:
    """Normalize O*NET codes to clean strings (strip; empty/NaN -> <NA>) with pandas 'string' dtype."""
    s = s.astype("string")
    s = s.str.strip()
    return s.mask(s.isin([None, "", "nan", "None"]))

def _norm_title_series(s: pd.Series) -> pd.Series:
    """Trim/clean raw titles; keep dtype=str to preserve values."""
    s = s.fillna("").astype(str).str.strip()
    return s

# ---------- Build pipeline ----------
def build():
    # Inputs
    so = _load(CURATED / "so" / "profiles.parquet")
    traits = _load(CURATED / "onet" / "traits.parquet")
    crosswalk = _load(REF / "role_crosswalk.csv")

    if so.empty:
        raise SystemExit("Missing data/curated/so/profiles.parquet")
    if crosswalk.empty:
        raise SystemExit("Missing data/reference/role_crosswalk.csv")
    if traits.empty:
        print("⚠️  traits.parquet not found or empty — proceeding without O*NET joins.")
        traits = pd.DataFrame(columns=["onet_code"])

    # Standardize SO columns
    so = so.rename(columns={
        "Respondent": "profile_id",
        "ResponseId": "profile_id",
        "DevType": "raw_title",
    })
    # Common questionnaire-ish fields if present:
    # country / Country, EdLevel / education_level, YearsCodePro / exp_years, JobSat / satisfaction
    for cand, canon in [
        ("Country", "country"),
        ("EdLevel", "education_level"),
        ("YearsCodePro", "exp_years"),
        ("JobSat", "satisfaction"),
    ]:
        if cand in so.columns and canon not in so.columns:
            so[canon] = so[cand]

    so = _ensure_cols(
        so,
        ["profile_id", "raw_title", "country", "education_level", "exp_years", "satisfaction"]
    )
    so["profile_id"] = so["profile_id"].astype(str)

    # Explode multi-valued DevType like "A; B; C"
    so["raw_title"] = _norm_title_series(so["raw_title"])
    exploded = so.assign(raw_title=so["raw_title"].str.split(";")).explode("raw_title")
    exploded["raw_title"] = exploded["raw_title"].str.strip()
    exploded = exploded[exploded["raw_title"] != ""]

    # Crosswalk merge (normalize titles & codes)
    cw = crosswalk.copy()
    cw["raw_title"] = _norm_title_series(cw.get("raw_title", pd.Series([], dtype=str)))
    cw["canonical_role"] = _norm_title_series(cw.get("canonical_role", pd.Series([], dtype=str)))
    cw["onet_code"] = _norm_code_series(cw.get("onet_code", pd.Series([], dtype="string")))

    merged = exploded.merge(cw, on="raw_title", how="left")
    # If a profile has multiple matched titles, keep the first non-null canonical mapping
    merged = merged.sort_values(["profile_id", "canonical_role"], na_position="last")
    dedup = merged.drop_duplicates("profile_id", keep="first")

    # Traits join — force BOTH sides to 'string' dtype and normalized format BEFORE merge
    if not traits.empty:
        traits = traits.copy()
        if "onet_code" not in traits.columns:
            traits["onet_code"] = pd.Series([], dtype="string")
        traits["onet_code"] = _norm_code_series(traits["onet_code"])
        dedup["onet_code"] = _norm_code_series(dedup.get("onet_code", pd.Series([], dtype="string")))
        trait_cols = ["onet_code"] + [c for c in traits.columns if c.startswith(("trait_", "skill_", "workstyle_"))]
        traits = traits[trait_cols].drop_duplicates("onet_code")
        dedup = dedup.merge(traits, on="onet_code", how="left")

    # Light feature cleanup
    if "satisfaction" in dedup.columns:
        sat = pd.to_numeric(dedup["satisfaction"], errors="coerce")
        dedup["satisfaction"] = (sat - 1.0) / 4.0  # map 1..5 -> 0..1

    # Placeholder text embeddings (overwritten by reembed_text.py)
    dedup["text_vec"] = [np.zeros(384, dtype=np.float32)] * len(dedup)

    # Splits
    usable = dedup.copy()
    usable["label_role"] = usable["canonical_role"].astype(str)
    strat = usable["label_role"].fillna("_")
    if strat.nunique() > 1:
        train, temp = train_test_split(usable, test_size=0.3, random_state=42, stratify=strat)
        valid, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label_role"].fillna("_"))
    else:
        train, temp = train_test_split(usable, test_size=0.3, random_state=42)
        valid, test = train_test_split(temp, test_size=0.5, random_state=42)

    train["split"], valid["split"], test["split"] = "train", "valid", "test"
    full = pd.concat([train, valid, test], ignore_index=True)

    keep = [
        "profile_id", "split", "label_role", "canonical_role", "onet_code",
        "country", "education_level", "exp_years", "satisfaction", "text_vec"
    ] + [c for c in full.columns if c.startswith(("trait_", "skill_", "workstyle_"))]

    full = full[keep]
    out = FEATURES / "training_view.parquet"
    full.to_parquet(out, index=False)
    print(f"Saved: {out} (rows={len(full)})")

def main():
    build()

if __name__ == "__main__":
    main()
