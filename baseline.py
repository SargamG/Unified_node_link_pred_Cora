import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
NODES_PATH = f"{DATA_DIR}/nodes.csv"

OUTPUT_PATH = "sample_submission.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------
# Load data
# -------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
nodes = pd.read_csv(NODES_PATH)

# Node features matrix
X_nodes = nodes.set_index("node_id")

# -------------------------------
# Helper functions
# -------------------------------
def get_node_features(node_ids):
    return X_nodes.loc[node_ids].values


def get_edge_features(src, dst):
    """
    Very simple edge representation:
    element-wise product of node features
    """
    src_feat = get_node_features(src)
    dst_feat = get_node_features(dst)
    return src_feat * dst_feat


# -------------------------------
# Split training data by task
# -------------------------------
node_train = train[train.task_type == "node"]
edge_train = train[train.task_type == "link"]

# -------------------------------
# NODE CLASSIFICATION BASELINE
# -------------------------------
X_node_train = get_node_features(node_train.src)
y_node_train = node_train.label.values

node_clf = LogisticRegression(
    max_iter=2000,
    multi_class="auto",
    n_jobs=-1
)
node_clf.fit(X_node_train, y_node_train)

# -------------------------------
# LINK PREDICTION BASELINE
# -------------------------------
X_edge_train = get_edge_features(
    edge_train.src.values,
    edge_train.dst.values
)
y_edge_train = edge_train.label.values

edge_clf = LogisticRegression(
    max_iter=2000,
    n_jobs=-1
)
edge_clf.fit(X_edge_train, y_edge_train)

# -------------------------------
# Generate predictions on test set
# -------------------------------
predictions = []

for _, row in test.iterrows():
    if row.task_type == "node":
        node_id = row.src
        x = get_node_features([node_id])
        pred = node_clf.predict(x)[0]
    else:
        src, dst = row.src, row.dst
        x = get_edge_features([src], [dst])
        pred = edge_clf.predict_proba(x)[0, 1]
    predictions.append(pred)

# -------------------------------
# Save submission
# -------------------------------
submission = pd.DataFrame({
    "id": test.id,
    "prediction": predictions
})

submission.to_csv(OUTPUT_PATH, index=False)

print("Baseline submission written to:", OUTPUT_PATH)
