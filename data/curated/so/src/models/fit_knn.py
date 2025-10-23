import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.common.io import FEATURES, MODELS

def main():
    df = pd.read_parquet(FEATURES / "training_view.parquet")
    X = np.vstack(df["text_vec"].to_list())
    roles = df["canonical_role"].tolist()

    nn = NearestNeighbors(metric="cosine", n_neighbors=min(5, len(roles)))
    nn.fit(X)

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "knn_index.pkl", "wb") as f:
        pickle.dump({"nn": nn, "roles": roles}, f)
    print(f"Saved knn_index.pkl for {len(roles)} roles")

if __name__ == "__main__":
    main()
