# scripts/query_roles_fix.py
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TOPK_NEIGHBORS = 300   # scan a wider pool for more stable per-role scoring
TOPK_ROLES = 10        # how many roles to display
MIN_SCORE = 0.20       # hide weak matches

# --- Simple query expansion for stronger embeddings ---
EXPAND_MAP = {
    "frontend": ["ui", "ux", "web", "react", "typescript", "css", "javascript"],
    "back-end": ["api", "microservices", "java", "spring", "python", "django", "flask"],
    "full-stack": ["frontend", "backend", "react", "node", "typescript"],
    "data engineer": ["etl", "spark", "airflow", "aws", "glue", "pipelines"],
    "mlops": ["kubernetes", "docker", "airflow", "mlflow", "terraform", "ci/cd"],
    "analyst": ["sql", "excel", "tableau", "power bi", "dashboards", "analytics"],
    "product manager": ["stakeholder", "roadmap", "metrics", "a/b testing", "user research"],
    "devops": ["docker", "kubernetes", "ci/cd", "terraform", "ansible"],
    "cloud": ["aws", "gcp", "azure", "iam", "lambda", "kinesis"],
}

def expand_query(q: str) -> str:
    q_low = q.lower()
    extras = []
    for k, vals in EXPAND_MAP.items():
        if k in q_low:
            extras.extend(vals)
    if not extras:
        return q
    # add unique, sorted extras to stabilize embedding text
    extras = sorted(set(extras))
    return q + " " + " ".join(extras)

# --- Small lexical boost if role name matches query keywords ---
def keyword_boost(role: str, q: str) -> float:
    words = set(re.findall(r"[a-zA-Z][a-zA-Z\-+_/]*", q.lower()))
    base = role.lower()
    hits = sum(1 for w in words if w in base)
    # tiny bounded boost so cosine still dominates
    return min(0.05, 0.01 * hits)

def main():
    df = pd.read_parquet("data/features/training_view.parquet")

    # Build role label per row (canonical if present, else raw_title, else Unknown)
    role_series = df.get("canonical_role")
    if role_series is None:
        role_series = pd.Series([""] * len(df))
    role_fallback = df.get("raw_title")
    if role_fallback is None:
        role_fallback = pd.Series(["Unknown"] * len(df))

    roles = role_series.fillna("").astype(str).str.strip()
    roles = roles.mask(roles.eq(""), role_fallback.fillna("Unknown").astype(str).str.strip())

    # Prepare normalized text embedding matrix
    vecs = df["text_vec"].to_numpy()
    mat = np.vstack([np.asarray(v, dtype="float32").ravel() for v in vecs]).astype("float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms  # L2 normalize -> cosine = dot

    # Load embedder once
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Type a short profile (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            print("Please enter a few skills/keywords.")
            continue

        q_expanded = expand_query(q)
        qv = model.encode([q_expanded], normalize_embeddings=True)[0].astype("float32")

        sims = mat @ qv  # cosine similarities

        # Take a larger candidate pool so each role can accumulate several strong rows
        cand_idx = np.argpartition(-sims, min(TOPK_NEIGHBORS, len(sims)-1))[:TOPK_NEIGHBORS]
        cand = pd.DataFrame({
            "idx": cand_idx,
            "role": roles.iloc[cand_idx].values,
            "sim": sims[cand_idx].astype("float32"),
        })

        # Aggregate per role:
        #   - take top few row-sims per role
        #   - score = 0.7 * max + 0.3 * mean(top3)
        grouped_scores = []
        for role, g in cand.groupby("role", sort=False):
            sims_sorted = np.sort(g["sim"].values)[::-1]
            top3 = sims_sorted[:3] if sims_sorted.size >= 3 else sims_sorted
            max_sim = float(sims_sorted[0])
            mean_top3 = float(np.mean(top3)) if top3.size else 0.0
            score = 0.7 * max_sim + 0.3 * mean_top3
            score += keyword_boost(role, q)  # tiny lexical nudge
            grouped_scores.append((role, score, max_sim))

        # Sort by composite score, filter weak, and take top roles
        ranked = sorted(grouped_scores, key=lambda x: -x[1])
        ranked = [(r, s, m) for (r, s, m) in ranked if s >= MIN_SCORE][:TOPK_ROLES]

        print("\nTop roles:")
        if not ranked:
            print("- (no strong matches — try adding a few more skills/tools)")
        else:
            for role, score, max_sim in ranked:
                print(f"- {role}  (cos sim ≈ {score:.3f})")
        print()  # blank line between queries

if __name__ == "__main__":
    main()
