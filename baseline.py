import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score

# -------------------------------
# Config
# -------------------------------
# Public data location (align with template)
DATA_DIR = "data/public"
EMBED_DIM = 64
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load data
# -------------------------------
nodes = pd.read_csv(f"{DATA_DIR}/nodes.csv")
train_nodes = pd.read_csv(f"{DATA_DIR}/train_nodes.csv")
test_nodes = pd.read_csv(f"{DATA_DIR}/test_nodes.csv")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

num_nodes = len(nodes)
num_feats = nodes.shape[1] - 1
num_classes = train_nodes["label"].nunique()

# Node feature tensor
X = torch.tensor(
    nodes.drop(columns=["node_id"]).values,
    dtype=torch.float32,
    device=DEVICE
)

# -------------------------------
# Build adjacency list
# -------------------------------
edge_rows = train[train["task_type"] == "link"]
edges = edge_rows[edge_rows["label"] == 1][["src", "dst"]].values

adj = [[] for _ in range(num_nodes)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# -------------------------------
# Simple GraphSAGE layer
# -------------------------------
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, adj):
        agg = []
        for i in range(len(adj)):
            if len(adj[i]) == 0:
                agg.append(x[i])
            else:
                agg.append(x[adj[i]].mean(dim=0))
        agg = torch.stack(agg)
        h = torch.cat([x, agg], dim=1)
        return torch.relu(self.lin(h))

# -------------------------------
# Model
# -------------------------------
class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.sage = GraphSAGELayer(in_dim, hidden_dim)
        self.node_head = nn.Linear(hidden_dim, num_classes)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, adj):
        return self.sage(x, adj)

# -------------------------------
# Train
# -------------------------------
model = GNN(num_feats, EMBED_DIM, num_classes).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion_node = nn.CrossEntropyLoss()
criterion_edge = nn.BCEWithLogitsLoss()

train_node_ids = torch.tensor(train_nodes["node_id"].values, device=DEVICE)
train_node_labels = torch.tensor(train_nodes["label"].values, device=DEVICE)

edge_train = train[train["task_type"] == "link"]
edge_src = torch.tensor(edge_train["src"].values, device=DEVICE)
edge_dst = torch.tensor(edge_train["dst"].values, device=DEVICE)
edge_labels = torch.tensor(edge_train["label"].values, device=DEVICE).float()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    h = model(X, adj)

    # Node loss
    node_logits = model.node_head(h[train_node_ids])
    loss_node = criterion_node(node_logits, train_node_labels)

    # Edge loss
    edge_repr = torch.cat([h[edge_src], h[edge_dst]], dim=1)
    edge_logits = model.edge_head(edge_repr).squeeze()
    loss_edge = criterion_edge(edge_logits, edge_labels)

    loss = loss_node + loss_edge
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

# -------------------------------
# Inference
# -------------------------------
model.eval()
with torch.no_grad():
    h = model(X, adj)

preds = []

for _, row in test.iterrows():
    if row.task_type == "node":
        node_id = int(row.src)
        logits = model.node_head(h[node_id])
        pred = logits.argmax().item()
    else:
        u, v = int(row.src), int(row.dst)
        edge_vec = torch.cat([h[u], h[v]])
        logit = model.edge_head(edge_vec).item()
        pred = torch.sigmoid(torch.tensor(logit)).item()

    preds.append({"id": row.id, "prediction": pred})

submission = pd.DataFrame(preds)
submission_out = f"{DATA_DIR}/sample_submission.csv"
submission.to_csv(submission_out, index=False)

print(f"{submission_out} generated")
