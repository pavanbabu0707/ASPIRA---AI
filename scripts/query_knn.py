# scripts/query_knn.py
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = Path("models/knn_index.pkl")
TOPK_NEIGHBORS = 100  # search deeper, then aggregate
TOPK_ROLES = 10        # show unique roles

def main():
    with open(MODEL, "rb") as f:
        obj = pickle.load(f)
    nn = obj["nn"]
    roles = obj["roles"]

    enc = SentenceTransformer("all-MiniLM-L6-v2")
    print("Type a short profile (or 'exit'):")

    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        qvec = enc.encode([q], normalize_embeddings=True).astype(np.float32)
        dist, idx = nn.kneighbors(qvec, n_neighbors=min(TOPK_NEIGHBORS, len(roles)), return_distance=True)
        dist, idx = dist[0], idx[0]

        role_best = defaultdict(lambda: -1.0)
        for d, i in zip(dist, idx):
            sim = 1.0 - float(d)  # cosine sim
            r = roles[i]
            if sim > role_best[r]:
                role_best[r] = sim

        top = sorted(role_best.items(), key=lambda x: x[1], reverse=True)[:TOPK_ROLES]
        print("Top roles:")
        for r, s in top:
            print(f"- {r}  (cos sim ≈ {s:.3f})")

if __name__ == "__main__":
    main()
