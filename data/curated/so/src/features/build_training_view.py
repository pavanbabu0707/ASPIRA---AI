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
