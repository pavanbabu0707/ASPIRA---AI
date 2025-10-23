# scripts/query_roles_fix.py
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

TOPK_NEIGHBORS = 100
TOPK_ROLES = 10

def main():
    df = pd.read_parquet("data/features/training_view.parquet")

    mat = np.vstack(df["text_vec"].to_numpy()).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms  # ensure unit norm

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Type a short profile (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"): break
        if not q:
            print("Please enter a few skills/keywords."); continue

        qv = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        sims = mat @ qv  # cosine because both unit-normalized

        idx = np.argpartition(-sims, TOPK_NEIGHBORS)[:TOPK_NEIGHBORS]
        top = sorted([(int(i), float(sims[i])) for i in idx], key=lambda x: -x[1])

        seen = {}
        for i, s in top:
            role = (str(df.get("canonical_role", pd.Series([""])).iloc[i]) or
                    str(df.get("raw_title", pd.Series(["Unknown"])).iloc[i]) or "Unknown").strip()
            if role not in seen:
                seen[role] = s
            if len(seen) >= TOPK_ROLES:
                break

        print("Top roles:")
        for role, s in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"- {role}  (cos sim ≈ {s:.3f})")

if __name__ == "__main__":
    main()
