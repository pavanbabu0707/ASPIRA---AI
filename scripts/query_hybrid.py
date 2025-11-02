# scripts/query_hybrid.py
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.hybrid import (
    load_text_centroids, load_trait_centroids,
    encode_text, hybrid_score
)

ALPHA = 0.8  # start text-heavy until questionnaire traits are wired

def main():
    df_text, M_text = load_text_centroids()
    df_trait, M_trait = load_trait_centroids()  # may be (None, None) if not present

    roles = df_text["canonical_role"].tolist()

    print("Type profile (or 'exit'):")
    while True:
        q = input("> ").strip()
        if not q or q.lower() == "exit":
            break

        q_text = encode_text(q)  # unit-norm
        # For now we have no user trait vector; set None to force text-only fallback.
        q_trait = None

        sims = hybrid_score(M_text, M_trait, q_text, q_trait, alpha=ALPHA)
        idx = np.argsort(-sims)[:7]
        print("\nTop roles (hybrid fallback-aware):")
        for i in idx:
            print(f"- {roles[i]}  (score ≈ {sims[i]:.3f})")
        print()

if __name__ == "__main__":
    main()
