.PHONY: setup ingest build_features fit_knn fit_kmeans clean

setup:
\tpython -m venv .venv && \\
\t./.venv/bin/pip install --upgrade pip && \\
\t./.venv/bin/pip install -e .

ingest:
\tpython src/ingest/dummy_check.py  # placeholder for your loaders

build_features:
\tpython src/features/build_training_view.py

fit_knn:
\tpython src/models/fit_knn.py

fit_kmeans:
\tpython src/models/fit_kmeans.py

clean:
\trm -rf dist build *.egg-info
