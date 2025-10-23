from pathlib import Path
import pandas as pd

OUT = Path("data/curated/onet"); OUT.mkdir(parents=True, exist_ok=True)

rows = [
    {"onet_code": "15-2051.01", "trait_dependability": 4.5, "trait_analytical": 4.7},
    {"onet_code": "15-2051.00", "trait_dependability": 4.4, "trait_analytical": 4.6},
    {"onet_code": "15-1252.00", "trait_dependability": 4.3, "trait_analytical": 4.5},
    {"onet_code": "13-1111.00", "trait_dependability": 4.2, "trait_analytical": 4.1},
    {"onet_code": "15-1242.00", "trait_dependability": 4.3, "trait_analytical": 4.2},
    {"onet_code": "15-1243.00", "trait_dependability": 4.2, "trait_analytical": 4.3},
]
pd.DataFrame(rows).to_parquet(OUT / "traits.parquet", index=False)
print("✅ Wrote", OUT / "traits.parquet")
