## Submissions

Submit by opening a Pull Request that adds a folder:

`submissions/inbox/<team_name>/<run_id>/`

Required files:
- `predictions.csv.enc`
- `metadata.json`

### `predictions.csv` format (later encrypted to predictions.csv.enc)

Preferred columns:

```csv
id,y_pred
node_1708,3
edge_12_45,0.82
```

Notes:
- `id` must match exactly the IDs in the public test file (expected: `data/public/test.csv`).
- Node rows must be **integer** class labels.
- Link rows must be **probabilities in [0,1]**.

### `metadata.json` format

Example:

```json
{
  "team": "example_team",
  "model": "llm-only",
  "llm_name": "gpt-x",
  "notes": "GraphSAGE + tuned heads"
}
```


