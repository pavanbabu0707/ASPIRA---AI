from pathlib import Path
import pandas as pd

IN = Path("data/reference/role_crosswalk_with_suggestions.csv")  # or role_crosswalk.csv if you prefer
OUT = Path("data/reference/role_crosswalk_final.csv")

# Our reviewed overrides
FIX = {
    "Developer, back-end": "15-1252.00",
    "Developer, desktop or enterprise applications": "15-1252.00",
    "Developer, mobile": "15-1252.00",
    "Student": "",
    "Developer, embedded applications or devices": "15-1252.00",
    "DevOps specialist": "15-1244.00",  # alt: "15-1299.08" if available in your O*NET
    "Academic researcher": "",
    "Research & Development role": "",
    "Senior Executive (C-Suite, VP, etc.)": "11-1011.00",
    "Cloud infrastructure engineer": "15-1241.00",
    "Developer, game or graphics": "15-1252.00",
    "Data or business analyst": "15-1211.00",  # alt: "13-1111.00"
    "Developer, QA or test": "15-1253.00",
    "Security professional": "15-1212.00",
    "Product manager": "11-1021.00",
    "Engineer, site reliability": "15-1244.00",
    "Educator": "25-1199.00",
    "Developer Experience": "",
    "Developer Advocate": "",
}

KEEP_OK = {
    "Developer, full-stack": "15-1252.00",
    "Developer, front-end": "15-1254.00",
    "Engineering manager": "11-9041.00",
    "Data scientist or machine learning specialist": "15-2051.00",
    "Engineer, data": "15-1243.01",
    "System administrator": "15-1244.00",
    "Project manager": "15-1299.09",
    "Scientist": "15-1221.00",
    "Blockchain": "15-1299.07",
    "Hardware Engineer": "17-2061.00",
    "Designer": "15-1255.00",
    "Database administrator": "15-1242.00",
    "Marketing or sales professional": "11-2021.00",
}

def main():
    df = pd.read_csv(IN, dtype=str).fillna("")
    # Start from suggested or existing, then apply overrides
    # If onet_code is empty, take suggested_onet_code; then apply our FIX dict
    df["onet_code"] = df["onet_code"].where(df["onet_code"].str.strip() != "", df["suggested_onet_code"])

    # Apply overrides
    for raw, code in {**KEEP_OK, **FIX}.items():
        mask = df["raw_title"].str.strip().str.casefold() == raw.casefold()
        df.loc[mask, "onet_code"] = code

    # Write the final crosswalk
    out_cols = ["raw_title", "canonical_role", "onet_code"]
    df[out_cols].to_csv(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(df)} rows")

if __name__ == "__main__":
    main()
