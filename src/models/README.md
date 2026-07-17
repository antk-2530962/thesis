# Model registry — best runs per dataset × architecture

One row per (dataset, architecture): the current best-scoring run and where it
came from. Every run lives at `models/<dataset>/<architecture>/<timestamp>/`
with its own `README.md` (training conditions + metrics), `data.md` (exact
data/subset/split), `metadata.json` (machine-readable run record), `assets/`
(plots) and `cache/` (eval artifacts). Weights are gitignored; the docs are
the record.

## Queryable index — [index.csv](index.csv)

Every run's `metadata.json` is flattened into one table, [index.csv](index.csv),
so "best 3 gru runs on gislr" or "all ME_126 runs" is a filter/sort instead of
a folder crawl. Rebuild + query it (runs from anywhere):

```bash
.venv/Scripts/python.exe scripts/build_model_index.py                       # rebuild + full leaderboard
.venv/Scripts/python.exe scripts/build_model_index.py --dataset gislr --architecture gru --top 3
.venv/Scripts/python.exe scripts/build_model_index.py --subset ME_126
```

`metadata.json` lifecycle: written by the training notebook's run-docs cell
(`gislr.1.model.gru.ipynb` §7) with `eval_status: "pending"` and the
training-loop accuracy (`train_val_acc`); `scripts/eval_gru.py` promotes it to
`eval_status: "canonical"` by filling `overall_accuracy` / `macro_accuracy` /
`median_class_accuracy` / `n_classes_below_50pct`. The index's `val_acc`
column uses the canonical number when present, else `train_val_acc` — check
`eval_status` before treating a row as leaderboard-comparable. Rebuild
`index.csv` after each training run or canonical eval.

## Leaderboard

| dataset | architecture | best run | input | params | val acc (overall) | macro | notes |
|---|---|---|---|---|---|---|---|
| gislr | gru | [20260715-190729](gislr/gru/20260715-190729/README.md) | **ME-126** subset × xyz (378) | 0.95M | **73.73%** | 73.49% | landmark-subset ablation winner; +3.14 over full-543 |
| gislr | gru *(prev best)* | [20260713-213000](gislr/gru/20260713-213000/README.md) | all 543 × xyz (1,629) | 1.91M | 70.59% | 70.36% | full-input baseline |

All runs above share the canonical evaluation: stratified 90/10 split
(`random_state=42`), 9,448-video val set, per-class accuracy from raw parquet.
A new run displaces the leader only on the same split and metric.

## Context

- **ME-126** = Kaggle-1st-place 118 landmarks (lips, hands, nose, eyes) ∪
  upper-body pose {11–16, 23, 24}. Derivation, motion-energy evidence and the
  1st-place cross-check: `docs/2026-07-15.md`.
- Planned entries: exact 1st-place-118 GRU, ME-126 xy-only, lag-feature GRU
  (TODO §3.1); 1D-CNN + Transformer port and other architectures (TODO §4).
