from pathlib import Path
import pandas as pd

# Path to your downloaded Stack Overflow CSV
SO_CSV = r"C:\Users\pavan\OneDrive\Desktop\DASC CAPSTONE PROJECT\New folder\archive (1)\survey_results_public.csv"

# Output folder
OUT = Path("data/curated/so")
OUT.mkdir(parents=True, exist_ok=True)

print("📥 Reading Stack Overflow CSV...")
df = pd.read_csv(SO_CSV, low_memory=False)

# Map Stack Overflow columns to standardized names
colmap = {
    "ResponseId": "profile_id",
    "Respondent": "profile_id",  # fallback for older schema
    "Country": "country",
    "EdLevel": "education_level",
    "YearsCodePro": "exp_years",
    "JobSat": "satisfaction",
    "DevType": "raw_title",      # job role(s)
}
for src, dst in colmap.items():
    if src in df.columns:
        df = df.rename(columns={src: dst})

# Normalize numeric fields
if "exp_years" in df:
    df["exp_years"] = pd.to_numeric(df["exp_years"], errors="coerce")

# Map JobSat text → numeric scale if present
if "satisfaction" in df:
    sat_map = {
        "Very satisfied": 5,
        "Slightly satisfied": 4,
        "Neither satisfied nor dissatisfied": 3,
        "Slightly dissatisfied": 2,
        "Very dissatisfied": 1,
    }
    df["satisfaction"] = df["satisfaction"].map(sat_map).fillna(df["satisfaction"])

# Ensure required columns exist
keep = ["profile_id", "raw_title", "country", "education_level", "exp_years", "satisfaction"]
have = [c for c in keep if c in df.columns]
for c in keep:
    if c not in have:
        df[c] = None

df = df[keep].dropna(subset=["profile_id"]).drop_duplicates("profile_id")

# Save to Parquet
out_path = OUT / "profiles.parquet"
df.to_parquet(out_path, index=False)
print(f"✅ Wrote {out_path} with {len(df)} rows")
