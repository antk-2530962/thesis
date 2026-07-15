# TODO — sign2speech

Living project TODO, organized by workstream so new tasks can be filed under an
existing section or a new one added without restructuring.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[?]` open question / decision needed

**How to add a task:** file it under the matching workstream section below. If it
doesn't fit an existing one, add a new `## N. <Workstream Name>` section at the end
(before "Backlog / Someday") rather than bolting it onto an unrelated section.

---

## 0. Repo Restructure Follow-ups & Tooling

Cleanup left over from the move to the flat `src/` layout (notebooks renamed to
`<dataset>.<stage>.<topic>.ipynb`, `modules/` slimmed to `paths.py` + `data/`,
checkpoints+metrics merged into one per-run folder).

### 0.1 Stale imports / references broken by the restructure

- [ ] `src/gislr.1.model.gru.ipynb` imports `from modules.dataset import ...` —
  actual module is now `modules.data.dataset`.
- [ ] `src/popsign.1.mediapipe.ipynb` imports `from modules.datasets import DATASETS`
  and uses `DATASETS["ISLR"]` — `datasets/` was deleted; it's now
  `modules.paths.DATASETS` with key `"GISLR"`.
- [ ] `src/modules/` has no `__init__.py` files (the old ones were deleted) — either
  restore them or confirm namespace-package imports work from the `src/` kernel CWD.
- [ ] `src/popsign.1.mediapipe.ipynb` currently contains early **GISLR** motion-energy
  exploration code, not POPSIGN extraction — retire that content (superseded by
  `gislr.0.dataset.motion-energy.ipynb`) and rebuild the notebook as the extraction
  driver (§2).

### 0.2 Packaging / config

- [x] `pyproject.toml`: torch/torchvision/torchaudio sat under an invalid top-level
  `[dependencies]` table. **Fixed 2026-07-15:** `torch>=2.13.0` declared in
  `[project].dependencies` with the cu130 index pinned via `[tool.uv.sources]`;
  torchvision/torchaudio dropped (nothing imports them, and torchaudio has no
  cu130 build that resolves for the full `requires-python` range). `uv sync`
  verified: torch 2.13.0+cu130, CUDA available.
- [x] `pyproject.toml`: placeholder description replaced (2026-07-15).
- [ ] Type checker: `pyrefly.toml` was deleted but `pyrefly` is still a runtime
  dependency, and the dev group also pulls in `ty` — pick one, drop the other, and
  move it to the dev group.
- [ ] `.gitignore`: the bare `data/` pattern also ignores `src/data/external/`
  (the MediaPipe `holistic_landmarker.task`) and `src/data/cache/dataframes/`
  (POPSIGN manifests) — decide whether to narrow the ignore and commit those, or
  document them as download/generate-on-setup. (Still open after the 2026-07-15
  rewrite — the pattern was kept as-is pending this decision.)
- [x] `.gitignore`: rewritten 2026-07-15 for the `src/` layout — stale root-level
  `cache/*.npy` lines and the self-ignoring `.gitignore` line removed; now covers
  `src/cache/`, model weights (`src/models/**/*.pt`, bare `gru_best.pt` /
  `gru_latest.pt` from notebook runs), export artifacts (`*.onnx`, `*.tflite`,
  `src/saved_model_dir/`, `src/final_saved_model/`) and `.ipynb_checkpoints/`.
- [x] Repo size: `src/gislr.0.competition.entry.1st.ipynb` was 17 MB — 15 MB of
  landmark-animation cell outputs (cells 10–13) stripped 2026-07-15 → 247 KB.
  Code, markdown and the training-log output are intact. Full copy with outputs:
  `src/cache/gislr.0.competition.entry.1st.with-outputs.ipynb` (gitignored) or
  Kaggle discussion 406978.

---

## 1. Landmark Motion-Over-Time Analysis (GISLR)

**Goal:** One robust, resumable notebook measuring how much each landmark moves
over time, at three scopes: per-video, per-category, global. Builds on existing
motion-energy pipeline findings (RMS speed, `["type","landmark_index"]` grouping,
Savitzky-Golay filtering) rather than re-deriving them.

**Location:** `src/gislr.0.dataset.motion-energy.ipynb`

**Status: ✅ executed end-to-end 2026-07-15 — all three scopes complete, 0 failed
units. Findings, stats, figures and the landmark keep/discard recommendation are
written up in `docs/2026-07-15.md`.** Remaining work moved to §1.8.

### 1.0 Decision to lock in

- [x] **Confirm DuckDB as the loading layer.** Adopted — `gislr.0` builds on
  `get_duckdb_conn()` querying parquet via `CREATE VIEW ... glob`, so aggregation
  happens before pulling into pandas and peak memory stays bounded by one query's
  result, not dataset size. Revisit only if a blocker turns up.

### 1.1 Reusable core (build once, use in all three scopes)

- [x] `get_duckdb_conn()` — validated end-to-end (94,477 videos · 250 signs · 21
  participants resolved from the meta view).
- [x] `load_landmarks_for_paths(paths: list[str]) -> pd.DataFrame` — validated; the
  one function all three scopes call (explicit parquet file list, not glob filter).
- [x] `compute_motion_energy(df: pd.DataFrame) -> pd.DataFrame` — validated:
  Savitzky-Golay (window=7, polyorder=2) → RMS speed over raw-valid transitions
  only (NaN policy) → tidy long format incl. `n_valid_transitions`.
- [x] `plot_motion_gridspec(df, title)` — validated across all three scopes
  (auto-detects per-video vs aggregate frames, std as error bars).
- [x] Run the reusable-core cells against real data — done, all scopes ran clean.

### 1.2 Resumable caching / state management

- [x] Manifest per scope: `cache/motion_analysis/<scope>_manifest.json` — with
  atomic saves (temp file + `os.replace`).
- [x] Idempotent write pattern: per-unit parquet written **before** marking `done`
  in the manifest — validated over 50 + 10 + 189 units, 0 failures.
- [x] Resume check (skip `done`, retry `failed`) + per-unit try/except — validated
  by the executed runs (skip path exercised on re-runs; a deliberate
  mid-run-interrupt drill wasn't needed given the invariants held over 249 units).
- [x] Final aggregation reads only cached per-unit files (`load_cached_units` /
  in-SQL over chunk parquets), never raw parquet — validated for all three scopes.

### 1.3 Scope 1 — Per-video (50 random samples)

- [x] Sample 50 video paths (seed 42, recorded to `per_video_sample.json`).
- [x] Per video: load → compute → cache → plot (50 PNGs in `per_video/plots/`).
- [x] Output: `cache/motion_analysis/per_video/summary.parquet` (27,150 rows =
  50 videos × 543 landmarks).

### 1.4 Scope 2 — Per-category (10 sampled sign categories)

- [x] Sample 10 sign labels (seed 42, recorded to `per_category_sample.json`).
- [x] Per category: batched load (100 videos/read) → aggregate RMS speed per
  landmark (mean + std — feeds the future within-class consistency analysis).
- [x] Output: `cache/motion_analysis/per_category/summary.parquet`
  (`sign, type, landmark_index, rms_speed_mean, rms_speed_std, n_videos`).

### 1.5 Scope 3 — Global (entire dataset)

- [x] Decision: full in-SQL aggregation rejected (Savitzky-Golay needs ordered
  per-frame series) → chunked Python compute, then **in-SQL aggregation over the
  cached chunk parquets** (the memory-bounded part still happens in DuckDB).
- [x] 189 chunks of ≤500 videos through load → compute → cache, manifest tracking
  chunk completion — full run ≈25 min at ~65 videos/s, 0 failures.
- [x] Output: `cache/motion_analysis/global/summary.parquet` + `global_overview.png`.

### 1.6 Cross-scope comparison

- [x] Overlay plot + rank correlation: per-video sample rho **0.954**, per-category
  sample rho **0.996** vs global (n=543) — the seeded samples reproduce the global
  per-landmark pattern, so sample-based landmark analyses can be trusted.
- [x] Cross-check against the competition 1st-place landmark subset — done in
  `docs/2026-07-15.md` §5: agrees on hands / face-mass-discard / legs / z-drop;
  diverges on upper-body pose (ME-126 keeps, 1st place drops) and lips (1st
  place keeps, on linguistic grounds motion energy can't see).

### 1.8 Follow-ups from the 2026-07-15 findings (report §7)

- [x] xy-vs-xyz decomposition on the 50-video sample — **~92% of pose "motion" is
  z-axis noise** (pose-head 99%, legs 95%; hands only 24%, face 14%). Cached at
  `cache/motion_analysis/xy_vs_xyz_sample50.parquet`, chart in the report.
- [ ] Re-run the **global** scope with xy-only RMS (store `rms_speed_xy` alongside
  `rms_speed` in the chunk schema) so landmark-importance numbers at full-dataset
  scale aren't z-contaminated.
- [ ] Consider adding the xy/xyz split to `compute_motion_energy` itself (cheap —
  same smoothed array, second reduction) before any re-run.

### 1.7 Explicitly out of scope here

- Within-class / cross-class ANOVA-style discriminability analysis (separate
  future task — this notebook only produces its motion-energy inputs)
- Gradient saliency / SHAP (needs a trained model; this is pre-training analysis)
- Spectrogram-format conversion

---

## 2. Bulk Landmark Extraction (POPSIGN)

**Decision (resolved):** extracted landmarks go to a **separate drive** configured
via `POPSIGN_LANDMARKS_DRIVE` in `.env` — too large to live next to the code.

### 2.1 Fix `modules/data/landmark_worker.py` before any bulk run

- [ ] **Bug: landmarks are never saved.** `process_video()` collects
  `pose_frames` / `lh_frames` / `rh_frames` but `np.savez_compressed()` writes only
  `fps` and `num_frames` — a full extraction run would produce empty archives.
- [ ] `MODEL_PATH_STRING` points at `tools/mediapipe/tasks/holistic_landmarker.task`,
  but the task file lives at `src/data/external/mediapipe/tasks/` — resolve via
  `modules.paths` instead of a hardcoded stale string.
- [ ] `OUTPUT_DIR` is hardcoded to `./data/landmarks/` — wire it to
  `POPSIGN_LANDMARKS_DRIVE` from `.env` (and actually populate `.env`, currently empty).

### 2.2 Bulk run

- [ ] Rebuild `src/popsign.1.mediapipe.ipynb` as the extraction driver (its current
  contents are stale GISLR exploration — see §0.1). The old `run_extraction.py`
  driver was dropped in the restructure.
- [ ] Resumable bulk extraction with QC manifests (interruption-safe over ~30K
  videos) — reuse the manifest pattern from §1.2 rather than inventing a second one.
- [ ] Regenerate/verify the video manifests (`src/data/cache/dataframes/train.csv`,
  `test.csv`) from `popsign.0.dataset.ipynb` — currently only 1 of 4 POPSIGN train
  datasets is enabled in `modules/paths.py` (the other three downloads are
  commented out).

---

## 3. Data-Driven Landmark Importance

- [x] Motion energy (feeds from §1) — delivered; keep/discard recommendation in
  `docs/2026-07-15.md` §4 (keep: hands 42 + upper-body pose 8 + lips 40 + eyes/
  nose 36 = **ME-126**; discard: 392 face, pose head/legs, z channel).
- [ ] Within-class consistency + cross-class discriminability (ANOVA-style
  between/within variance ratio) — the instrument that can actually rank the
  face landmarks (motion energy sees a flat 0.003 band there) and adjudicate
  the pose-vs-no-pose divergence with the 1st-place subset.
- [ ] Position as complementary to gradient saliency and SHAP from trained models
  (not a replacement)

### 3.1 Landmark-subset training ablations (GRU, all-else-identical)

Controlled runs that change ONLY the input subset vs the full-543 baseline
(`src/models/gislr/gru/20260713-213000`, val acc 70.59%):

- [x] **ME-126, xyz** — done 2026-07-15 (`src/models/gislr/gru/20260715-190729`):
  **73.73% val vs 70.59% baseline (+3.14) with 50.4% fewer params** (0.95M),
  failing classes 22→9. Leaderboard updated in `src/models/README.md`.
- [ ] Exact 1st-place 118 (ME-126 minus the 8 pose landmarks) — isolates whether
  upper-body pose helps a *streaming* model (hand-dropout fallback hypothesis).
- [ ] ME-126 with **xy only** (drop z; input 252) — tests the z-noise finding
  in-model rather than only in the motion statistics.
- [ ] ME-126 + lag-1/lag-2 difference features (the 1st-place motion features) —
  note these are causal, so streaming-safe.

---

## 4. Architecture Benchmarking

- [ ] LSTM / BiLSTM baselines vs the existing GRU (BiLSTM offline-only — accuracy
  reference, never a deployment candidate)
- [ ] ST-GCN, TCN, Transformer, Conformer — evaluate against the recurrent baselines
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

*Last updated: July 15, 2026*
