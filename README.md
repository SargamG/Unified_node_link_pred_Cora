# Unified_node_link_pred_Cora
# 🧠 Multi-Task Graph Representation Challenge

This challenge explores how well a **single graph neural network (GNN)** can learn **shared node representations** that generalize across **multiple graph tasks** on the same dataset.

Participants must train a model that performs well on **both node classification and link prediction**, using only the provided graph data and within strict constraints.

### [View Live Leaderboard](https://SargamG.github.io/Unified_node_link_pred_Cora)
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

The dataset is derived from a citation network and consists of the following files (located in `data/public/`):

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

1. **Fork** this repository.

2. Generate predictions for **all rows** in the public test file:  
   `data/public/test.csv`

3. Create a folder in your fork:

   ```
   submissions/inbox/<team_name>/<run_id>/
   ```

4. Inside that folder, create:
   - `predictions.csv`
   - `metadata.json` (optional but recommended)

5. Your `predictions.csv` must look like:

   ```csv
   id,y_pred
   node_1708,3
   edge_12_45,0.82
   ```

   - **Node rows** → `y_pred` must be a class label (integer)  
   - **Edge rows** → `y_pred` must be a probability in `[0,1]`

6. Encrypt Your Predictions (Required)

For security and fairness, you **must encrypt your submission before uploading**.

  - **Install dependency (if not already installed)**:

      ```bash
      pip install cryptography
      ```
      
  - **Encrypt your file from the root of the repository**:
      
      ```bash
      python encryption/encrypt.py submissions/inbox/<team_name>/<run_id>/predictions.csv
      ```
      
      This will generate:
      
      ```
      predictions.csv.enc
      ```
      
  - **Delete the plaintext file**:
      
      ```bash
      rm submissions/inbox/<team_name>/<run_id>/predictions.csv
      ```
      
      ⚠️ Only the encrypted `.enc` file should be submitted.

7. Your folder should now contain:

   ```
   submissions/inbox/<team_name>/<run_id>/
       predictions.csv.enc
       metadata.json
   ```
   
8.  Submit via Pull Request

   Commit and push **only the encrypted file**:
   
   ```bash
   git add submissions/inbox/<team_name>/<run_id>/predictions.csv.enc
   git commit -m "Submission: <team_name>"
   git push
   ```
   
   Then open a **Pull Request** to this repository.

**Important Rules**

- Only **one submission per GitHub user** is allowed.
- If you submit more than once, your PR will not update the leaderboard.
- If your PR does **not close automatically**, it most likely means:
  - You have already submitted once.
  - Please check the PR comments for details.

**Note**: If your submission is not scored automatically, it is likely because your GitHub account is considered a first-time or new contributor. In this case, make any prior public contribution on GitHub (e.g., open a PR anywhere, even a typo fix), then re-submit. If the scoring fails, check to make sure your predictions.csv has the correct format, number of rows and columns.

---

## 7️⃣ Leaderboard

🏆 The live leaderboard is maintained automatically:
- Only the best score per participant is retained
- Scores update instantly after PR submission

### [View Live Leaderboard](https://SargamG.github.io/Unified_node_link_pred_Cora)
---

## 📌 Getting Started

A simple baseline using a GraphSAGE-style model is provided in baseline.py
It demonstrates:
- shared node embeddings
- joint optimization of node + link tasks
- correct submission format

Participants are encouraged to improve upon it. Focus on improving the GNN's learnt features rather than modifying the complete model architecture. A GNN with two MLP heads for prediction as in the baseline should suffice.

---

## References

- [Deep Graph Learning Playlist](https://www.youtube.com/playlist?list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)
- [One for All: Towards Training One Graph Model for All Classification Tasks](https://arxiv.org/html/2310.00149v3)

