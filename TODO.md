# TODO — sign2speech

Living project TODO, organized by workstream so new tasks can be filed under an
existing section or a new one added without restructuring. Mirrors the sections in
`Project_Structure` / `ASL_Recognition_Pipeline_Report`.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[?]` open question / decision needed

**How to add a task:** file it under the matching workstream section below. If it
doesn't fit an existing one, add a new `## N. <Workstream Name>` section at the end
(before "Backlog / Someday") rather than bolting it onto an unrelated section.

---

## 1. Landmark Motion-Over-Time Analysis (GISLR)

**Goal:** One robust, resumable notebook measuring how much each landmark moves
over time, at three scopes: per-video, per-category, global. Builds on existing
motion-energy pipeline findings (RMS speed, `["type","landmark_index"]` grouping,
Savitzky-Golay filtering) rather than re-deriving them.

**Suggested location:** `notebooks/gislr/00_landmark_motion_analysis.ipynb`
(pre-`01_preprocess` — diagnostic/exploratory, not part of the train pipeline)

### 1.0 Decision to lock in

- [?] **Confirm DuckDB as the loading layer.** Querying parquet directly via
  `CREATE VIEW ... glob` avoids materializing all 30K videos in memory; aggregation
  happens before pulling into pandas, so peak memory stays bounded by one query's
  result, not dataset size. Resolves the previously "exploratory, not finalized"
  DuckDB decision from the report — accept going forward unless a blocker turns up.

### 1.1 Reusable core (build once, use in all three scopes)

- [ ] `get_duckdb_conn()` — in-memory DuckDB connection, `CREATE VIEW meta_holistic`
  joining `train.csv` metadata to parquet landmarks via glob pattern (path
  normalization: strip `islr_str`, normalize `\\` → `/`).
- [ ] `load_landmarks_for_paths(paths: list[str]) -> pd.DataFrame` — single
  parameterized query; the one function all three scopes call.
- [ ] `compute_motion_energy(df: pd.DataFrame) -> pd.DataFrame` — group by
  `["type", "landmark_index"]` → Savitzky-Golay (window=7, polyorder=2) → RMS speed
  (not mean-squared) → tidy long format: `type, landmark_index, rms_speed, video_id[, sign]`.
- [ ] `plot_motion_gridspec(df, title)` — reuse gridspec layout (combined overview
  ordered pose → left_hand → right_hand → face, plus per-type panels). Parameterize
  for single-video, category-aggregate, and global use without three separate
  plotting functions.

### 1.2 Resumable caching / state management

- [ ] Manifest per scope: `cache/motion_analysis/<scope>_manifest.json` — tracks
  item id, status (`pending`/`done`/`failed`), timestamp, output artifact path.
- [ ] Idempotent write pattern: write per-unit result file
  (`cache/motion_analysis/<scope>/<id>.parquet`) **before** marking `done` in the
  manifest — a crash mid-write never leaves a `done` entry pointing at a corrupt file.
- [ ] Resume check at notebook start: skip ids already `done`, retry `failed`.
- [ ] Final aggregation reads only cached per-unit files, never recomputes from raw
  parquet.
- [ ] Wrap per-unit processing in `tqdm` + try/except; log failures to manifest
  instead of raising.

### 1.3 Scope 1 — Per-video (50 random samples)

- [ ] Sample 50 video paths (fixed seed, recorded in output for reproducibility).
- [ ] Per video: load → compute → cache → plot (save PNG, don't just display inline).
- [ ] Output: `cache/motion_analysis/per_video/summary.parquet`
  (`video_id, sign, type, landmark_index, rms_speed`).

### 1.4 Scope 2 — Per-category (10 sampled sign categories)

- [ ] Sample 10 sign labels (fixed seed).
- [ ] Per category: batched load for all videos in that sign → aggregate RMS speed
  per landmark (mean + std — feeds the future within-class consistency analysis).
- [ ] Output: `cache/motion_analysis/per_category/summary.parquet`
  (`sign, type, landmark_index, rms_speed_mean, rms_speed_std, n_videos`).

### 1.5 Scope 3 — Global (entire dataset)

- [ ] Prefer in-SQL aggregation via DuckDB where possible (memory-bounded — this is
  where DuckDB's advantage over pandas matters most).
- [ ] If full in-SQL aggregation isn't feasible (Savitzky-Golay needs per-frame
  ordering, which SQL can't do cleanly), fall back to chunked batches (e.g. 500
  videos/chunk) through the same load → compute → cache path, manifest tracking
  chunk completion instead of per-video.
- [ ] Output: `cache/motion_analysis/global/summary.parquet` (same schema as
  per-category, one row set for the whole dataset).

### 1.6 Cross-scope comparison

- [ ] Overlay/side-by-side plot: per-video sample vs per-category vs global RMS
  speed per landmark — sanity check whether the samples represent the global
  pattern before trusting them for downstream landmark-importance decisions.
- [ ] Note in notebook: findings are **not conclusive** until cross-checked against
  the competition-suggested landmark subset (per report §3.1).

### 1.7 Explicitly out of scope here

- Within-class / cross-class ANOVA-style discriminability analysis (separate
  future task — this notebook only produces its motion-energy inputs)
- Gradient saliency / SHAP (needs a trained model; this is pre-training analysis)
- Spectrogram-format conversion

---

## 2. Bulk Landmark Extraction (POPSIGN)

- [ ] Resumable bulk extraction with QC manifests (interruption-safe over ~30K videos)
- [ ] `landmark_worker.py` / `run_extraction.py` split already in place — extend
  with manifest-based resume once extraction begins in earnest

---

## 3. Data-Driven Landmark Importance

- [ ] Motion energy (feeds from §1) + within-class consistency + cross-class
  discriminability (ANOVA-style between/within variance ratio)
- [ ] Position as complementary to gradient saliency and SHAP from trained models
  (not a replacement)

---

## 4. Architecture Benchmarking

- [ ] ST-GCN, TCN, Transformer, Conformer — evaluate against BiLSTM/GRU baselines
- [ ] Caution: 1st-place GISLR Kaggle solution found hand-crafted angle/distance
  features didn't help and GCNs underperformed simpler sequence models — keep this
  in mind when scoping the ST-GCN evaluation

---

## 5. Spectrogram-Format Checkpoint (CNN/ViT arm)

- [ ] xyz-as-RGB channels, landmarks on y-axis, frames on x-axis
- [ ] Linear interpolation to fixed frame count
- [ ] Scoped only to this benchmarking arm — image quantization deferred to
  spectrogram-build time, not baked into the shared checkpoint format

---

## Backlog / Someday

- [ ] (add unscoped ideas here as they come up, promote to a numbered section once
  they have a concrete plan)

---

*Last updated: July 14, 2026*