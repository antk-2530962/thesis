# Data — GISLR · FP_118 subset

- **dataset**: Kaggle `asl-signs` (GISLR), 94,477 videos, 250 classes
- **landmark subset**: **FP_118** (118 landmarks) from the canonical
  registry `src/modules/dataset/landmark/subsets.py`; exact indices in
  [`cache/landmarks.npy`](cache/landmarks.npy)
- **channels**: xyz -> feature_dim 354
- **preprocessing**: NaN->0, uniform subsample to <= 128 frames
  (identical to the 20260713-213000 baseline)
- **split**: stratified 90/10, `random_state=42` -> 85,029 train /
  9,448 val (the canonical leaderboard split)
- **feature caches**: `src/cache/{train,val}_fp118_{data,offsets}.npy`
- **subset provenance**: gislr.0.competition.entry.1st.ipynb POINT_LANDMARKS (hoyso48).
