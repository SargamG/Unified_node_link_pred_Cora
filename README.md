# Unified_node_link_pred_Cora
# 🧠 Multi-Task Graph Representation Challenge

This challenge explores how well a **single graph neural network (GNN)** can learn **shared node representations** that generalize across **multiple graph tasks** on the same dataset.

Participants must train a model that performs well on **both node classification and link prediction**, using only the provided graph data and within strict constraints.

---

## 1️⃣ Problem Statement

Given a citation graph where:
- nodes represent research papers,
- edges represent citation relationships, and
- nodes have high-dimensional feature vectors,

your goal is to learn **node embeddings** that can simultaneously:

1. **Classify unseen nodes** into research areas  
2. **Predict unseen edges** (citation links) between nodes  

The challenge is intentionally designed so that **optimizing one task alone is insufficient**—successful solutions must learn **general-purpose representations** useful for multiple objectives.

---

## 2️⃣ Dataset Description

The dataset is derived from a citation network and consists of the following files (located in `data/`):

### 🔹 Node Features
- `nodes.csv`
- 2708 nodes
- 1433 features per node (`x0` to `x1432`)
- Each row corresponds to a unique node ID

### 🔹 Node Labels
- Training labels: nodes `0–639`
- Test labels: nodes `1708–2707`
- Nodes `640–1707` are **unlabeled** and must not be used for node supervision

### 🔹 Graph Structure
- Directed edges represent citations
- The source cites the destination

---

## 3️⃣ Task Definition

This is a **multi-task learning challenge** with two tasks:

### 🔹 Task 1: Node Classification
**Objective:**  
Predict the research category of unseen nodes.

- Input: node features + graph structure
- Output: class label (integer) from 0 to 6

---

### 🔹 Task 2: Link Prediction
**Objective:**  
Predict whether a citation link exists between two nodes.

- Input: pair of node embeddings
- Output: probability ∈ [0, 1]

---

Both tasks must be solved using **a shared node embedding space**.

---

## 4️⃣ Evaluation Metric

Each submission is evaluated on **both tasks**, and a single final score is computed.

### 🔹 Metrics
- **Node Classification:** Macro F1-score  
- **Link Prediction:** ROC-AUC  

### 🔹 Final Score
- Final Score = 0.5 × Node Macro-F1 + 0.5 × Link ROC-AUC
- Equal weighting ensures that neither task can be ignored.

---

## 5️⃣ Rules & Constraints

To keep the challenge fair and focused:

- ❌ No external datasets or pretrained models are allowed
- ❌ No manual label engineering  
- ✅ Any GNN architecture allowed (GCN, GraphSAGE, etc.)
- ❌ Solutions should not use different embeddings for both tasks

Submissions that violate these rules may be disqualified.

---

## 6️⃣ How to Submit

1. Fork this repository  
2. Generate predictions for **all rows** in `data/test.csv`
3. Create a CSV file in the following format:
    ```csv
    id,prediction
    node_1708,3
    edge_12_45,0.82
4. Node rows → prediction is a class label (integer)
5. Edge rows → prediction is a probability in [0,1]
6. Place the file in: submissions/
7. Open a Pull Request to this repository
8. Your submission will be scored automatically using GitHub Actions.

**Note**: If your submission is not scored automatically, it is likely because your GitHub account is considered a first-time or new contributor. In this case, make any prior public contribution on GitHub (e.g., open a PR anywhere, even a typo fix), then re-submit.

---

## 7️⃣ Leaderboard

🏆 The live leaderboard is maintained automatically:
- Only the best score per participant is retained
- Scores update instantly after PR submission

---

## 📌 Getting Started

A simple baseline using a GraphSAGE-style model is provided in baseline.py
It demonstrates:
- shared node embeddings
- joint optimization of node + link tasks
- correct submission format

Participants are encouraged to improve upon it.
