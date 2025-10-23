import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.common.io import FEATURES, MODELS

def main():
    df = pd.read_parquet(FEATURES / "training_view.parquet")
    X = np.vstack(df["text_vec"].to_list())

    k = min(5, len(df))  # simple heuristic
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)

    with open(MODELS / "kmeans.pkl", "wb") as f:
        pickle.dump(km, f)
    print(f"Saved kmeans.pkl with {k} clusters")

if __name__ == "__main__":
    main()
