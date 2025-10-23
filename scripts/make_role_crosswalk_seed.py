from pathlib import Path
import pandas as pd

PROFILES = Path("data/curated/so/profiles.parquet")
OUT = Path("data/reference/role_crosswalk.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PROFILES)

# Use 'raw_title' (SO DevType). Split multi-values like "Developer; Data Scientist"
titles = (
    df.get("raw_title", pd.Series([], dtype=str))
      .fillna("")
      .astype(str)
      .str.split(";", expand=True)
      .stack()
      .str.strip()
      .replace("", pd.NA)
      .dropna()
)

top = titles.value_counts().head(50).index.tolist()

# Auto-map a few common ones; leave the rest blank to fill later
auto = {
    "Data Scientist": ("Data Scientist", "15-2051.00"),
    "Data Engineer": ("Data Engineer", "15-2051.01"),
    "Machine Learning Engineer": ("ML Engineer", "15-2051.01"),
    "Software Developer": ("Software Engineer", "15-1252.00"),
    "DevOps Engineer": ("DevOps Engineer", "15-1243.00"),
    "Database Administrator": ("Database Administrator", "15-1242.00"),
    "Business Analyst": ("Business/Data Analyst", "13-1111.00"),
    "Cloud Engineer": ("Cloud Engineer", "15-1242.00"),
}

rows = []
for t in top:
    canon, code = auto.get(t, (t, ""))  # default: same title, empty code to fill later
    rows.append({"raw_title": t, "canonical_role": canon, "onet_code": code})

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"✅ Wrote {OUT} with {len(rows)} seed rows — fill missing onet_code values next.")
