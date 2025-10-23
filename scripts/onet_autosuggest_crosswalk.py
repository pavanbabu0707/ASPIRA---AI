from pathlib import Path
import re
import pandas as pd

# Optional: better fuzzy matching (pip install rapidfuzz)
try:
    from rapidfuzz import process, fuzz
    HAVE_RF = True
except Exception:
    import difflib
    HAVE_RF = False

# --- YOUR EXACT FILE PATHS (keep the r'' raw strings) ---
OCC_PATH = Path(r"C:\Users\pavan\OneDrive\Desktop\DASC CAPSTONE PROJECT\New folder\db_29_3_excel\Occupation Data.xlsx")
ALT_PATH = Path(r"C:\Users\pavan\OneDrive\Desktop\DASC CAPSTONE PROJECT\New folder\db_29_3_excel\Alternate Titles.xlsx")

IN_CROSSWALK = Path("data/reference/role_crosswalk.csv")
OUT_SUGGESTED = Path("data/reference/role_crosswalk_suggested.csv")
OUT_MERGED = Path("data/reference/role_crosswalk_with_suggestions.csv")
# --------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, dtype=str)
    elif path.suffix.lower() in {".csv", ".txt"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def normalize(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def best_match(query: str, choices: pd.Series):
    if HAVE_RF:
        match, score, _ = process.extractOne(query, choices.tolist(), scorer=fuzz.WRatio)
        return match, score
    else:
        best = difflib.get_close_matches(query, choices.tolist(), n=1, cutoff=0.0)
        if best:
            # crude length-based score to keep API uniform
            score = int(100 * (1 - (abs(len(best[0]) - len(query)) / max(1, len(query)))))
            return best[0], score
        return None, 0

def load_onet_titles(occ_path: Path, alt_path: Path) -> pd.DataFrame:
    occ_df = read_table(occ_path)

    # Find typical columns
    code_col = next((c for c in occ_df.columns if "soc" in c.lower() and "code" in c.lower()), None)
    title_col = next((c for c in occ_df.columns if c.lower() == "title"), None)
    if not code_col or not title_col:
        raise ValueError(f"Unexpected columns in {occ_path.name}: {occ_df.columns.tolist()}")

    rows = occ_df[[code_col, title_col]].rename(columns={code_col: "onet_code", title_col: "title"})
    rows["source"] = "Occupation Data"

    if alt_path.exists():
        alt_df = read_table(alt_path)
        alt_title_col = next((c for c in alt_df.columns if c.lower() in {"alternate title","alternate_title","title"}), None)
        alt_code_col = next((c for c in alt_df.columns if "soc" in c.lower() and "code" in c.lower()), None)
        if alt_title_col and alt_code_col:
            alt_rows = alt_df[[alt_code_col, alt_title_col]].rename(columns={alt_code_col: "onet_code", alt_title_col: "title"})
            alt_rows["source"] = "Alternate Titles"
            rows = pd.concat([rows, alt_rows], ignore_index=True)

    rows["title_norm"] = rows["title"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    rows = rows.dropna(subset=["onet_code","title_norm"]).drop_duplicates(["onet_code","title_norm"])
    return rows

def main():
    onet = load_onet_titles(OCC_PATH, ALT_PATH)

    cw = pd.read_csv(IN_CROSSWALK, dtype=str).fillna("")
    if not {"raw_title","canonical_role","onet_code"}.issubset(cw.columns):
        raise ValueError("role_crosswalk.csv must have columns: raw_title, canonical_role, onet_code")

    suggestions = []
    for _, row in cw.iterrows():
        raw = row["raw_title"]
        # keep rows you already filled
        if row.get("onet_code","").strip():
            suggestions.append({**row, "suggested_onet_code":"", "suggested_onet_title":"", "match_score":""})
            continue

        q = normalize(raw)
        if not q:
            suggestions.append({**row, "suggested_onet_code":"", "suggested_onet_title":"", "match_score":""})
            continue

        match_title_norm, score = best_match(q, onet["title_norm"])
        if match_title_norm is None:
            suggestions.append({**row, "suggested_onet_code":"", "suggested_onet_title":"", "match_score":0})
            continue

        hit = onet.loc[onet["title_norm"] == match_title_norm].iloc[0]
        suggestions.append({
            **row,
            "suggested_onet_code": hit["onet_code"],
            "suggested_onet_title": hit["title"],
            "match_score": score,
            "onet_source": hit["source"],
        })

    out = pd.DataFrame(suggestions)
    OUT_SUGGESTED.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SUGGESTED, index=False)
    print(f"✅ Wrote suggestions → {OUT_SUGGESTED}")

    merged = out.copy()
    # Auto-accept high-confidence matches (score >= 85)
    mask = (merged["onet_code"].str.strip() == "") & (pd.to_numeric(merged["match_score"], errors="coerce").fillna(0) >= 85)
    merged.loc[mask, "onet_code"] = merged.loc[mask, "suggested_onet_code"]
    merged.to_csv(OUT_MERGED, index=False)
    print(f"✅ Wrote merged (auto-accepted >=85) → {OUT_MERGED}")

if __name__ == "__main__":
    main()
