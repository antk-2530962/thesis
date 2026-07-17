# Data — GISLR · ME_126 subset

- **dataset**: Kaggle `asl-signs` (GISLR), 94,477 videos, 250 classes
- **landmark subset**: **ME_126** (126 landmarks) from the canonical
  registry `src/modules/dataset/landmark/subsets.py`; exact indices in
  [`cache/landmarks.npy`](cache/landmarks.npy)
- **channels**: xyz -> feature_dim 378
- **preprocessing**: NaN->0, uniform subsample to <= 128 frames
  (identical to the 20260713-213000 baseline)
- **split**: stratified 90/10, `random_state=42` -> 85,029 train /
  9,448 val (the canonical leaderboard split)
- **feature caches**: `src/cache/{train,val}_me126_{data,offsets}.npy`
- **subset provenance**: docs/2026-07-15.md §4; GRU run 20260715-190729 (73.73% val acc); docs/2026-07-16.md verdict.
