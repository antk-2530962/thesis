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
`<dataset>.<stage>.<topic>.ipynb`), plus the **2026-07-18 restructure** (§0.4:
unified `modules/model` training stack, flat epoch-seconds registry with
meta.json schema v2, `data/{raw,cache,temp,models}` placement policy,
docs daily/weekly/reports split).

### 0.1 Stale imports / references broken by the restructure

- [x] **`src/modules/data/` no longer exists on disk** (2026-07-16): `dataset.py`
  (`GISLRRawDataset`) and `landmark_worker.py` are gone. `landmark_worker.py` is
  superseded by `modules/dataset/landmark/extraction.py` (§2). **Resolved
  2026-07-16:** `GISLRRawDataset` no longer needs restoring — the rebuilt
  `gislr.1.model.gru.ipynb` defines its dataset in-notebook (in-RAM arrays,
  `num_workers=0`, the pattern proven by the ME-126 training script).
- [x] `src/gislr.1.model.gru.ipynb` imports `from modules.dataset import ...` —
  **resolved 2026-07-16** by the notebook overhaul (§3.1): it now imports only
  the subset registry (`modules.dataset.landmark.subsets`) and defines the
  dataset class itself.
- [ ] `src/popsign.1.mediapipe.ipynb` imports `from modules.datasets import DATASETS`
  and uses `DATASETS["ISLR"]` — `datasets/` was deleted; it's now
  `modules.paths.DATASETS` with key `"GISLR"`.
- [x] `src/modules/` has no `__init__.py` files — **resolved 2026-07-18**: every
  package level ships one (`modules/`, `modules/model/`, `modules/scripts/`,
  `modules/dataset/`, `modules/dataset/landmark/`).
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

### 0.3 Model-run metadata & queryable index (2026-07-17, superseded by §0.4's schema v2)

Structured run records so "best 3 gru runs on gislr" / "all runs on subset X"
is a query, not a folder crawl:

- [x] Per-run **`metadata.json`** schema (dataset, architecture, subset, coords,
  n_params, hyperparameters, train-loop vs canonical accuracies, `eval_status`
  pending/canonical) — backfilled for all 5 existing gislr/gru runs 2026-07-17.
- [x] **`scripts/build_model_index.py`** — flattens every run's `metadata.json`
  into `src/models/index.csv` (committed) and answers filter queries
  (`--dataset/--architecture/--subset/--top`); warns on runs missing metadata.
- [x] `gislr.1.model.gru.ipynb` §7 (run-docs cell) now also writes
  `metadata.json` (preserving canonical-eval fields on re-runs);
  `scripts/eval_gru.py` promotes `eval_status` to `canonical` after the
  per-class eval.
- [x] Run folders are always fresh (2026-07-17): `gislr.1.model.gru.ipynb`'s
  `resolve_run_dir` no longer reuses/skips a **completed** run — every new
  training gets a new `<timestamp>` folder (timestamp = training start), even
  under identical conditions. Auto-resume still continues an *interrupted*
  run in its own folder.
- [~] Future training notebooks must write the same `metadata.json` schema —
  **done for GISLR** (the lstm/bilstm/cnn1d siblings inherit the GRU
  notebook's §7 emission, 2026-07-17); still applies to future POPSIGN
  training notebooks.
- [x] ~~Rebuild `index.csv` after the canonical evals of the six pending runs~~ —
  **voided 2026-07-18**: those runs' weights were deleted with the old
  `src/models/` tree during the restructure, so their canonical evals can
  never run. Their train-loop numbers stand as historical references
  (git history ≤ `3668dae` + docs/logs/daily/); the registry restarted empty (§0.4).

### 0.4 Restructure 2026-07-18 — unified stack, flat registry, data-tree policy

Executed 2026-07-18 (full write-up: `docs/logs/daily/2026-07-18.md`):

- [x] **`src/modules/model/`** — unified training stack (`architectures.py`
  single model-class definition shared with eval, `data.py`, `registry.py`,
  `train.py`, `report.py`); the four `gislr.1.model.*.ipynb` notebooks
  regenerated as thin drivers (identity block + `modules.model` calls).
- [x] **Registry v2**: flat `src/data/models/<epoch-seconds>/` folders holding
  only `meta.json` + `best.pt`/`last.pt` (gitignored) + `assets/`;
  schema v2 documented in README.md § "meta.json schema" (machine check:
  `modules/model/registry.py::REQUIRED_KEYS`); `meta.json` rewritten every
  epoch by the driver; **registry reset to empty** (header-only `index.csv`).
- [x] **Single progress bar per training run** (batch progress, metrics, LR,
  plateau counter in one bar) — replaces the nested-bar + per-epoch-print spam.
- [x] **`modules/paths.py`**: absolute CWD-independent tree constants
  (`RAW/CACHE/TEMP/EXTERNAL/MODELS`), lazy dataset resolution (import no
  longer downloads), `cleanup_temp()`.
- [x] **Data placement policy** applied: POPSIGN pilot npz → `data/temp/popsign_pilot/`
  with cleanup cell (stale `data/raw/popsign/_pilot` deleted); diagnostic
  caches → `data/cache/gislr/{motion_analysis,subset_comparison}`;
  manifests → `data/cache/popsign/dataframes/`; `.gitignore` reworked
  (all of `src/data/` ignored except `data/models/` minus weights/exports).
- [x] **Scripts split**: project CLIs in `src/modules/scripts/` (`eval_gru.py`
  takes a run folder, `build_model_index.py` flat-layout; both run from any
  CWD); root `scripts/` = housekeeping only.
- [x] **Docs split**: `docs/logs/daily/` + `docs/logs/weekly/<YYYY>-<WW>.md` +
  `docs/reports/<topic>.md` (convention in `docs/README.md`).
- [x] Regenerate the POPSIGN video manifests at
  `data/cache/popsign/dataframes/{train,test}.csv` — **done 2026-07-19**:
  §1 of `popsign.0.dataset.extraction.ipynb` now generates them from the raw
  video tree (30,867 train / 33,600 test, ids unique, labels cross-checked
  against the filename). Unblocks the §2 pilot/bulk runs. Still only 1 of 4
  train parts (§2.2).
- [ ] First v2-regime training runs (user) to seed the fresh registry —
  re-establishes the FULL_543 baseline and ME_126 leader under the new
  schema before any new ablation conclusions.
- [ ] `popsign.2.model.ipynb` / `popsign.3.pipeline.ipynb` still predate the
  restructure (old paths, TF-era code) — modernize or retire alongside
  `popsign.1` (§0.1).

### 0.5 Docs tree: `logs/` vs `reports/` (2026-07-19)

The docs tree grew a third top-level sibling (`daily/`, `weekly/`, `reports/`)
where only two *kinds* of document exist: time-ordered logs and standalone
test/analysis reports. Collapse the first two under one `logs/` parent:

```
docs/
├── logs/
│   ├── daily/<YYYY-MM-DD>.md
│   └── weekly/<YYYY>-<WW>.md
└── reports/<topic>.md        # motion-energy, subset-comparison, confidence-tuning, …
```

- [x] Move `docs/logs/daily/` → `docs/logs/daily/`, `docs/logs/weekly/` → `docs/logs/weekly/`
  (assets subfolders move with them).
- [x] Update every reference: `docs/README.md`, `README.md` (report table +
  project-structure block + conventions), `CLAUDE.md`, this file.
- [x] Backfill the standalone reports that currently live only as daily entries
  (motion-energy, subset-comparison) into `docs/reports/<topic>.md`, leaving the
  daily logs as the dated narrative that links to them. **Done 2026-07-22**:
  `docs/reports/motion-energy.md` and `docs/reports/subset-comparison.md`.
- [x] **Weekly logs started 2026-07-19**: `docs/logs/weekly/2026-29.md` (Jul 12–18)
  and `2026-30.md` (Jul 19–25, running). **Week numbering corrected**: weeks run
  **Sunday → Saturday**, week 1 = the week containing Jan 1 — *not* ISO, which
  `docs/README.md`, `README.md` and `CLAUDE.md` all previously said and which would
  number 2026-07-19 as the tail of week 29 rather than the start of week 30. All
  three updated; weekly titles now state the date range explicitly.
- [ ] Close out `2026-30.md` on Sat 2026-07-25 (drop the "in progress" marker) and
  open `2026-31.md`.

### 0.6 Stray notebook/module state to clean up (found 2026-07-19)

- [x] `src/popsign.0.dataset.extraction.ipynb` had a bare
  `DATASETS = resolve_datasets()` scratch cell in §1 (an unguarded call that
  would hit kagglehub on every top-to-bottom run). **Resolved 2026-07-19**: it
  became the proper manifest-generation cell — guarded by `FORCE_REGENERATE`
  and skipped entirely when both CSVs exist, so a plain re-run never reaches
  `resolve_datasets()` at all. (`force_download=True` is not in the current
  `modules/paths.py`; the ~870GB re-download hazard the original note described
  no longer exists.)
- [ ] `modules/paths.py::resolve_datasets` (uncommitted working-tree change)
  hardcodes `D:/`/`E:/` `PATH_PARTS` and calls `p.unlink()` on directories
  (raises on a real directory) — reconcile with the `.env`-driven policy.
  **Status 2026-07-22**: this is the actual change that downloaded the
  remaining 3 of 4 POPSIGN train dataset parts (§2.2) — `train-n-s-signs` to
  `D:/datasets/…` (complete 07-20) and `train-t-z-signs` to `E:/datasets/…`
  (complete 07-21) — so it did its job, but it's still uncommitted and still
  hardcoded rather than `.env`-driven. `src/temp.py` (untracked) is a scratch
  copy of an earlier version of this file, used to manually trigger the
  a-e/f-m downloads by hand — clean up both once the real fix is committed.
- [ ] `.env` is empty/missing at the repo root, so `POPSIGN_LANDMARKS_DRIVE`
  is unset and extraction output falls back into `src/data/raw/popsign`.

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

**Decision (resolved):** extracted landmarks go to **`data/raw/popsign/{train,test}`**,
rooted at the separate drive configured via `POPSIGN_LANDMARKS_DRIVE` in `.env`
when set (fallback: `src/data/`, gitignored) — too large to live next to the code.

### 2.1 Extraction module — `modules/dataset/landmark/extraction.py` (2026-07-16)

Replaces the deleted `modules/data/landmark_worker.py` (whose known bugs —
landmarks never written to the npz, stale hardcoded model path and output dir —
must not be reproduced):

- [x] Per-video MediaPipe Holistic extraction saving **all** landmark groups
  (face + pose + both hands → one `(T, 543, 3)` float16 npz in GISLR holistic
  row order + fps/num_frames metadata), atomic writes (temp file + `os.replace`).
  Smoke-validated 2026-07-16: 211-frame video → npz with 7% NaN, hand/pose/face
  blocks populated.
- [x] Multiprocess pool **capped at ≤70% of CPU / RAM** (worker count from
  `cpu_count × 0.70`, workers pinned to 1 math thread, RAM backpressure via
  `psutil`); MediaPipe is CPU-only on Windows so GPU is not touched.
  Pool + Windows-spawn path and resume-skip validated on a 4-video run.
- [x] Output layout `data/raw/popsign/{train,test}/<label>/<id>.npz`, root
  resolved via `POPSIGN_LANDMARKS_DRIVE` (fallback `src/data/`, gitignored).
- [x] **BUG (2026-07-19, found by the first pilot): leaked MediaPipe graphs
  deadlocked the pool.** POPSIGN mixes resolutions (1944×2592 and 1080×1920);
  the persistent per-worker landmarker raises `RET_CHECK ... current_mat->rows
  == previous_mat->rows` on a resolution change, and the retry handler rebuilt
  it by **rebinding `_LANDMARKER` without `.close()`** — leaking a native graph
  and ~70 threads each time. The wedged worker reached **219 threads / 1.3 GB**
  (vs 76 / 614 MB for its siblings), then stopped at **0% CPU**; the run stalled
  at 18/20 with no error, because `imap_unordered` cannot tell a live-but-hung
  worker from a slow one. An overnight 30K-video run would have hung silently.
  Fixed three ways: `_reset_landmarker()` closes before rebuilding; the
  landmarker is rebuilt **proactively on a resolution change** (checked once per
  video, so the exception path isn't used at all); `maxtasksperchild=64`
  recycles workers as a safety net. Verified on the exact failure sequence
  (2592 → 1920 → 2592 ending on the two hung videos): 4/4 in 21.7 s, 0 failed.
- [x] `extract_popsign.py pilot` **clears the temp tree before benchmarking** —
  `extract_dataset` skips done videos, so leftover npz from an interrupted pilot
  would time an empty trial and report a meaningless throughput.
- [ ] Consider a per-video **watchdog timeout** in the driver. The three fixes
  above address the known cause, but nothing yet bounds an unknown one: a worker
  that stops returning still hangs the whole run indefinitely.

### 2.2 Extraction driver — `src/popsign.0.dataset.extraction.ipynb` (2026-07-16)

Replaces the deleted `popsign.0.dataset.ipynb` stub as the extraction driver
(`popsign.1.mediapipe.ipynb` stays stale pending retirement, §0.1):

- [x] Manifest verification (`data/cache/dataframes/{train,test}.csv`) — both
  verified 2026-07-16 (30,867 train / 33,600 test rows, unique ids, spot-checked
  paths exist; train covers 72 labels = the 1 enabled dataset).
- [x] **Pilot batch (≤100 videos) cells built**: seeded sample, worker-count
  sweep (6/10/14/19) on disjoint 20-video slices under the 70% cap, measured
  videos/s + frames/s + CPU%, ETA + disk projection (`cache/popsign_extraction/`).
- [x] Resumable bulk-extraction cells with progress bars + QC section
  (interruption-safe over ~30K videos) — §1.2 manifest pattern (per-unit artifact
  before `done`, atomic saves, `failed` retried, batched manifest rewrites).
- [x] Pilot output moved to the temp tree (2026-07-18): npz →
  `data/temp/popsign_pilot/w<N>/`, deleted by the notebook's cleanup cell once
  `eta.json` is recorded in `data/cache/popsign/extraction/`.
- [x] **Extraction runs in the notebook (2026-07-19).** §2/§3/§4 call
  `extract_dataset` directly — the "pool can't run in Jupyter" guard was
  disproved and removed (§2.3). Two supporting fixes made it safe:
  the leaked-graph deadlock (§2.1), and **`_init_worker` redirecting each
  worker's fd 2** to `<out_dir>/<split>/_worker_stderr.log`. That second one is
  not cosmetic: MediaPipe logs from C++ straight to fd 2 and ipykernel captures
  fd-level output into cell output by default (`IPKernelApp.capture_fd_output`),
  so a 30K-video run would have written hundreds of lines *per worker* into the
  `.ipynb` — the 17 MB-notebook failure mode of §0.2.
  Verified by executing the notebook's own cells in a real ipykernel
  (setup → manifests → verify → pilot → ETA → cleanup): 0 failed,
  **0 noise lines, ~2 KB of cell output total**.
  `modules/scripts/extract_popsign.py` stays as an **optional** CLI for
  unattended runs; both share the manifests and are resumable, so they can be
  used interchangeably.
- [x] **Manifests regenerated 2026-07-19** by the notebook's new §1 cell —
  30,867 train / 33,600 test, verified against the raw tree (ids unique, every
  label agreeing with the sign encoded in its filename, spot-checked paths all
  present). The pilot is **no longer blocked**.
- [x] All 4 POPSIGN train dataset parts are now downloaded (2026-07-21) — see
  §0.6: `train-n-s-signs` → `D:/datasets/…` (complete 07-20 17:19),
  `train-t-z-signs` → `E:/datasets/…` (complete 07-21 16:29), `a-e`/`f-m`
  already in the default kagglehub cache. ~650GB that was outstanding all last
  week is now on disk.
- [ ] **Regenerate the train manifest** — `data/cache/popsign/dataframes/train.csv`
  was last written 2026-07-19 17:51, *before* the 3 new parts finished, so it
  still reflects only the old 72-label / 30,867-video single-part state. Consequence
  today: **train still covers only 72 labels while test covers all 250**, so 178
  test labels have no train videos yet — purely because the manifest hasn't been
  regenerated, not because the video is missing. Re-run §1 of
  `popsign.0.dataset.extraction.ipynb` with `FORCE_REGENERATE = True` before
  starting the train bulk run.
- [x] Run the pilot (user), review videos/s + resource headroom, then run the
  bulk extraction for train + test:
  ```
  .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py pilot
  .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py run train --confidence default
  ```
  **Test split: done 2026-07-20 02:06** — 33,599/33,600, **1 failed** (unchanged
  from initial report — investigate below), 1.461 videos/s wall, 8 workers,
  6.4 h total wall time. Output in `src/data/raw/popsign/test/`
  (`POPSIGN_LANDMARKS_DRIVE` still unset, §0.6); confirmed the resolution-change
  fix (§2.1) holds at scale across the full 33.6K videos, zero deadlocks.
  **Train: not started** — 0/30,867 done, blocked on the manifest regeneration
  above. A second pilot run on 2026-07-20 (post test-split, pre-manifest-regen)
  measured only 0.246–0.365 videos/s at 4–8 workers — **roughly a quarter of**
  what the test bulk run sustained at 8 workers. Likely explanation: the pilot's
  window overlaps the `train-n-s-signs` download finishing on the same drive
  that day, but this hasn't been confirmed — **re-run the pilot with no
  concurrent downloads before picking a worker count for the train run.**
  Full write-up: `docs/logs/weekly/2026-30.md` §3.
- [ ] Investigate the one failed test video —
  `gtsignstudy4a.8035-into-2023_01_30_12_00_12.563-0`, `cv2` cannot open the source
  mp4. Likely a truncated/corrupt download rather than an extraction bug; the
  manifest retries `failed` on the next run, so confirm the source file first.

### 2.4 Output inspection — `src/popsign.0.dataset.output-inspection.ipynb` (2026-07-19)

- [x] Standalone read-only diagnostic showing how an extracted sign is stored:
  archive keys/shapes/dtypes, the 543-row holistic group layout, per-group
  detection rates, one frame + a presence timeline, and the reference
  npz→model-input loader (NaN→0, subset gather, uniform subsample to
  `MAX_SEQ_LEN`) mirroring `modules/model/data.py`.
  **Safe to run against a live extraction by construction**: no writes anywhere in
  the landmarks tree, no worker pool, no MediaPipe import, and `.tmp.npz` staging
  files are excluded from sampling so a half-written video can never be opened.
  Format recorded in `docs/logs/daily/2026-07-19.md` §4c.

### 2.3 Extraction-quality / confidence tuning — `src/popsign.0.dataset.confidence-tuning.ipynb` (2026-07-19)

**Goal:** before committing ~30K videos of CPU time to bulk extraction, find out
which `HolisticLandmarker` confidence thresholds actually produce good landmarks
on POPSIGN video — with both a numeric score and a visual check, because no
ground-truth landmarks exist for this dataset.

**Sample:** 50 videos = 5 classes × 10 videos, seeded, recorded to the cache so
every re-run scores the same clips. Videos are globbed straight off the raw
video drives — this notebook deliberately does **not** depend on the missing
`data/cache/popsign/dataframes/` manifests (§2.2), so it is unblocked today.

- [x] **Finding (2026-07-19, measured): `min_hand_landmarks_confidence` is inert
  and the *pose* thresholds are what gate the hands.** Driving each field to
  0.01 vs 0.99 on one clip: the hand threshold produces **bit-identical**
  landmarks, while `min_pose_{detection,landmarks}_confidence` swing hand
  detection rate from **0.52 → 0.09**, and the face thresholds at 0.99 drop the
  face entirely (91% NaN). Cause: holistic derives hand ROIs from the pose
  landmarks. The naive grid (sweep the hand threshold) would have produced a
  table of identical rows and read as "tuning doesn't matter".
  Note also: the task API has **no** `min_tracking_confidence` and no separate
  hand *detection* threshold — the real field list is
  `extraction.CONFIDENCE_FIELDS`.
- [x] Threshold grid over the fields that actually move the output — pose
  detection/landmarks confidence (jointly and one at a time) plus a face-off
  arm — each config extracted over the 50-video sample, resumable per
  (config, video) via the §1.2 manifest pattern.
- [x] **Numeric quality proxies** (no ground truth, so these are proxies and are
  documented as such): per-group detection rate (fraction of frames with a
  non-NaN block), hand-presence rate, temporal jitter (median frame-to-frame
  landmark displacement — high = flicker), longest detection gap, and
  bone-length coefficient of variation (a rigid bone such as shoulder-elbow
  should keep constant length; variance = detection instability).
- [x] Composite `quality_score` per (config, video) combining those proxies, with
  the weighting exposed as a tunable so the ranking can be re-derived without
  re-extracting.
- [x] **Visual test**: 100 rendered frames with landmarks overlaid — 10 from the
  best-scoring frames and 80 from the worst-scoring (plus a 10-frame
  median-quality strip for reference), written to
  `data/cache/popsign/confidence_tuning/overlays/`, contact sheets shown inline.
- [x] Supporting module work: `extraction.py` gained `CONFIDENCE_FIELDS` /
  `DEFAULT_CONFIDENCE` and a `confidence=` parameter threaded through
  `extract_dataset` → pool initializer → landmarker (unknown fields assert);
  new `modules/dataset/landmark/quality.py` (proxies + composite score) and
  `overlay.py` (landmark drawing, frame rendering, contact sheets).
  Smoke-validated end to end 2026-07-19 on 2 configs × 2 videos.
- [x] **RESOLVED 2026-07-19: "`multiprocessing.Pool` cannot run in a Jupyter
  kernel" was wrong, and the guard has been removed.** Two measurements
  retired it:
  1. `multiprocessing.spawn.get_preparation_data` only sets `main_path` when
     `__main__` has a `__file__`. A kernel has none, so the child **never
     re-imports `__main__`**; `sys_path` *is* propagated and the worker
     functions live in an importable module. (The classic Jupyter+spawn failure
     is about workers defined *in the notebook* — ours are not.)
  2. A real ipykernel driven over ZMQ ran the pool end to end: 4 videos,
     2 workers, 0 failed, **21.6 s** — the same wall time as the CLI, through
     the resolution sequence that used to deadlock.

  The hang that motivated the guard was real but is far better explained by the
  **leaked-graph deadlock in §2.1**, which wedges a worker at 0% CPU with no
  error and which `imap_unordered` cannot detect. That is now fixed.
  The other objection — MediaPipe flooding cell output — is handled by
  `_init_worker`'s `stderr_log` fd-2 redirect (see §2.2). Extraction therefore
  runs **in the notebook**; `extract_popsign.py` remains as an optional CLI for
  unattended runs that should outlive the kernel.
  Superseded sub-items (kept for history):
  - `extraction._assert_pool_usable` refuses the case up front with
    instructions, so it fails in a second instead of hanging indefinitely.
  - `n_workers=1` now runs genuinely **in-process** (no Pool at all) — the only
    mode usable in-notebook, fine for a smoke test.
  - `extract_dataset(bar=...)` accepts a caller-owned tqdm so a multi-config
    sweep still shows ONE progress bar (the old per-config bar also meant the
    display could not move until an entire 50-video config finished).
  - **`modules/scripts/tune_confidence.py`** is the supported way to run the
    sweep; notebook §4 is now a handoff cell that prints the command and
    reports per-config progress from the npz on disk. Same work: ~100 videos in
    ~10 min as a script vs **0 files in 30 min** in-notebook.
- [x] **`popsign.0.dataset.extraction.ipynb` had the same defect** — its pilot
  (§2) and bulk (§3/§4) cells called `extract_dataset` in-notebook and would now
  raise instead of hanging. **Resolved 2026-07-19**: new
  `modules/scripts/extract_popsign.py` (`pilot` / `run <split>` subcommands,
  resumable, `--confidence` naming a tuning arm, `--limit` for staged runs);
  the three notebook cells are handoff cells that print the exact command and
  read progress back off the split manifest. `CONFIDENCE_CONFIG` in the setup
  cell is threaded through, so the bulk output records which thresholds
  produced it.
- [x] **Report written: `docs/reports/confidence-tuning.md`** (2026-07-19) —
  covers the inert-hand-threshold finding, the default-vs-`pose_strict` paired
  comparison, and the padding finding below.

**Sweep results so far (2026-07-19): 2 of 7 arms measured** (`default` 50/50
videos, `pose_strict` 49/50). Full write-up in the report; the operating config
is **not** yet chosen, and the sweep should not simply be finished as-is:

- [x] **Thresholds barely move the output, and `default` leads.** Paired over the
  49 videos both arms extracted, `pose_strict` costs 0.022 of any-hand detection
  rate (worse on 31 videos, better on 5) and buys 0.002 less hand jitter.
  Composite score 0.160 vs −0.163 — but with two arms the z-scored score is ±1
  by construction, so it is directional only.
- [ ] **BLOCKER — the proxies are measuring clip padding, not extraction quality.**
  Hand presence peaks at **0.86** mid-clip and sits at 0.12–0.19 across the first
  and last fifths; the median clip's first hand detection is at 27% of its
  duration and its last at 72%. Restricted to that span the same extraction
  scores **0.85 mean / 0.94 median** rather than 0.427. `longest_gap_frames`
  correlates with `n_frames` at **rho 0.84** — that proxy is very largely a
  measurement of the lead-in/lead-out. The effect being ranked (~0.02) is an
  order of magnitude below the artifact (~0.4). **Restrict the proxies to the
  signing span (or add `*_span` variants) before scoring anything else**, then
  re-derive the comparison above.
- [ ] Follow-up: `pose_rate` was 1.0 in every arm tested, including
  `min_pose_*_confidence = 0.99` — the pose block appears to be emitted
  whenever *any* pose is found, so the proxy can't discriminate pose quality.
  Either find a per-landmark visibility signal or drop `pose_rate` from the
  composite score's weighting (it currently contributes a constant offset).
- [ ] Then run the remaining five arms (`pose_permissive`, `pose_very_permissive`,
  `pose_det_only`, `pose_lm_only`, `face_off`) — ~250 extractions, ~25 min at 19
  workers — and record the chosen config as `CONFIDENCE_CONFIG` in
  `popsign.0.dataset.extraction.ipynb` (and as the default in `extraction.py`).
- [ ] **Separate deficiency, bigger than any threshold: only 1.1% of frames carry
  both hands** (left 9.5%, right 34.3%), and `hand_rate` tops out at exactly 0.50
  across the sample — the signature of "exactly one hand, always". Several
  sampled signs (`car`, `bath`) are two-handed in ASL. This is about *which*
  landmarks holistic returns and no confidence threshold addresses it. Inspect
  the `default` overlay frames before accepting any config.
- [ ] **Downstream consequence** (not a §2.3 item, filed here so it isn't lost):
  ~50% of every POPSIGN clip is non-signing lead-in/lead-out. Trimming, or a
  learned attention over the signing span, belongs in the POPSIGN
  feature-building stage.

---

## 3. Data-Driven Landmark Importance

- [x] Motion energy (feeds from §1) — delivered; keep/discard recommendation in
  `docs/2026-07-15.md` §4 (keep: hands 42 + upper-body pose 8 + lips 40 + eyes/
  nose 36 = **ME-126**; discard: 392 face, pose head/legs, z channel).
- [x] Within-class consistency + cross-class discriminability — **delivered
  2026-07-16** (`docs/2026-07-16.md`): per-landmark **ANOVA F-ratio** +
  **mutual information** on per-video descriptors, **probe classifier** as the
  subset score. Verdict: **ME-126 wins** (49.9% global probe; FP-118 48.6%,
  FULL-543 last at 40.6%); pose divergence adjudicated in ME's favor;
  discriminability is ~uncorrelated with motion energy (rho −0.12). Caveat
  discovered: marginal F cannot rank face landmarks either (rigid-head
  redundancy) — the subset-level probe is the instrument that prices it.
- [ ] Position as complementary to gradient saliency and SHAP from trained models
  (not a replacement)

### 3.0 Landmark-subset registry + comparison notebook (2026-07-16)

- [x] `src/modules/dataset/landmark/subsets.py` — canonical registry of every
  landmark subset in play (FULL_543, FP_118 = 1st-place, ME_126, ME_132,
  HANDS_42, HANDS_POSE_50, plus component groups) with holistic row indices —
  `ME_126.array` verified equal to the trained run's `landmarks.npy`.
- [x] `src/gislr.0.dataset.subset-comparison.ipynb` — **executed end-to-end
  2026-07-16** (scope A 10 videos / scope B 10 classes / scope C global 189
  chunks, 0 failures; global descriptors ≈50 min, probes ≈7 min). All 6
  registered subsets scored; `probe_acc_global` written back into
  `subsets.py`; report `docs/2026-07-16.md`.
- [~] Feed the winning subset + per-landmark rankings into the §3.1 training
  ablations (probe predicts: pose helps, pose-wrist points {17-22} don't) —
  top-3 probe subsets queued in the rebuilt `gislr.1.model.gru.ipynb`
  (2026-07-16), awaiting user run.
- [ ] Candidate new subset: **face-anchor reduction** (eyes/nose 36 → ~8 rigid
  anchors) — the face's discriminative signal is one rigid transform; needs a
  trained ablation before admission to the registry (report §4).
- [ ] Note for feature work: `x_std` is the most discriminative descriptor
  (median F 25.1 vs 17.3 for speed) — causal running-std features are
  streamable and worth a §3.1-style ablation.

### 3.1 Landmark-subset training ablations (GRU, all-else-identical)

Controlled runs that change ONLY the input subset vs the full-543 baseline
(historical run `20260713-213000`, val acc 70.59%).

**NOTE (2026-07-18):** all runs below predate the registry reset (§0.4) — their
weights are gone, so "canonical evals pending" can no longer be satisfied for
them. Their train-loop numbers stand as historical references; the ablation grid
should be re-run under regime v2 into the fresh registry before drawing final
subset conclusions (the run-to-run-variance caveat below makes this doubly true):

**2026-07-16:** `src/gislr.1.model.gru.ipynb` overhauled into the subset-ablation
driver: trains the top-3 probe subsets (`ME_126`, `ME_132`, `FP_118`) as
all-else-identical runs (per-subset caches + auto-resume + auto-generated run
docs), with `TRAIN_SUBSETS`/`COORDS` as the only knobs — the xy ablation below
is now a one-line config change. Awaiting user run.

- [x] **ME-126, xyz** — done 2026-07-15 (`src/models/gislr/gru/20260715-190729`):
  **73.73% val vs 70.59% baseline (+3.14) with 50.4% fewer params** (0.95M),
  failing classes 22→9. Leaderboard updated in `src/models/README.md`.
  (A reproducibility re-run is included in the rebuilt notebook's default
  `TRAIN_SUBSETS`; drop it there to save ~25 GPU-min.)
- [~] Exact 1st-place 118 (ME-126 minus the 8 pose landmarks) — isolates whether
  upper-body pose helps a *streaming* model (hand-dropout fallback hypothesis).
  Queued as `FP_118` in the rebuilt `gislr.1` notebook — awaiting user run.
- [~] **ME-132** (`ME_126` + pose wrist points {17-22}) — #2 by probe score;
  tests the probe's prediction that the extra 6 landmarks add nothing. Queued
  in the rebuilt `gislr.1` notebook — awaiting user run.
- [~] **xy only** (drop z) — tests the z-noise finding in-model. Trained
  2026-07-17 for all three subsets (v1 regime, `COORDS="xy"`): train-loop val
  acc **ME_126-xy 74.92 / ME_132-xy 74.95 / FP_118-xy 74.54 — each beats its
  xyz counterpart** (73.73 / 72.47–74.95 / 74.60), consistent with the z-noise
  finding. Canonical evals pending. `scripts/eval_gru.py` xy mode added
  2026-07-17 (reads the checkpoint's `coords` key). **Caveat**: ME_132's two
  same-config xyz runs differ by ~2.5 pts (72.47 vs 74.95) — run-to-run
  variance is on the order of the subset deltas, so ablation conclusions need
  the canonical evals (and ideally repeat runs).
- [ ] ME-126 + lag-1/lag-2 difference features (the 1st-place motion features) —
  note these are causal, so streaming-safe.

### 3.2 GRU training-regime update (batch ↑, epochs 300, early stopping) — implemented 2026-07-17

`src/gislr.1.model.gru.ipynb` now trains under regime **`v2-plateau-300`**
(no run executed yet — awaiting user):

- [x] **Batch size 192 → 512**, lr scaled 3e-4 → 1e-3 (16 GB GPU has ample
  headroom for the ~1M-param models; per-epoch time at 512 untested).
- [x] **Epochs 60 → 300 (cap) with early stopping** on val-accuracy plateau:
  no gain > `es_min_delta=1e-3` for `es_patience=15` epochs (both HYP tunables).
- [x] **Scheduler decision: ReduceLROnPlateau** (factor 0.5, patience 5, on
  val acc) — watches the same plateau signal as the stopper with shorter
  patience, so the LR gets a chance to rescue a plateau before the run ends.
  OneCycleLR dropped (fixed-epoch anneal would be truncated by early stops).
- [x] Comparability: `metadata.json`/`hyp.json`/checkpoints carry
  `training_regime` (`v1-onecycle-60` backfilled for all 8 prior runs;
  `index.csv` has the column). v1 vs v2 runs are not
  hyperparameter-comparable; canonical eval split/metric unchanged.
- [x] Resume-safety: plateau counter + scheduler state persist in the
  checkpoint (`epochs_since_gain`, `finished`); resolve_run_dir resumes only
  unfinished runs (finished = early-stopped or cap reached).
- [ ] Run the v2 regime (user) — start with the best subset — and compare
  against its v1 counterpart on the canonical eval before adopting v2 as the
  default family.

---

## 4. Architecture Benchmarking

GISLR architecture-benchmark notebooks (filed 2026-07-17): one flat notebook
per architecture (`gislr.1.model.<arch>.ipynb`), reusing the `gislr.1.model.gru.ipynb`
pattern — per-subset feature caches, canonical split, auto-resume, one
timestamped run folder per training start, auto-generated
`data.md`/`README.md`/`metadata.json` so every run lands in `src/models/index.csv`.
Train on the best-known subset for comparability with the GRU runs.

- [~] **`gislr.1.model.lstm.ipynb`** — unidirectional `StreamingLSTM`
  (streaming-viable, direct cell-vs-cell GRU comparison; 1.24M params at
  hidden 256×2). **Built 2026-07-17** — awaiting user run.
- [~] **`gislr.1.model.bilstm.ipynb`** — `BiLSTM`, offline-only **accuracy
  reference, never a deployment candidate** (prices the causality gap;
  fwd-last + bwd-first readout, 3.0M params). **Built 2026-07-17** (replaced
  the empty stub) — awaiting user run.
- [~] **`gislr.1.model.cnn1d.ipynb`** — `CausalConv1D`: 5 dilated causal
  Conv1d blocks (kernel 5, dilations 1..16, per-frame LayerNorm, 125-frame
  receptive field ≈ MAX_SEQ_LEN; 1.86M params). Streaming-viable; first step
  toward the 1st-place port. **Built 2026-07-17** — awaiting user run.

All three verified 2026-07-17 by CPU smoke test: forward shapes correct,
future-frame corruption provably doesn't change logits for the two causal
models, and notebook state_dicts load into `scripts/eval_gru.py`'s classes
with identical logits (the script now dispatches on the checkpoint's `arch`
key and handles xy/xyz via its `coords` key).
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

## 5.5 Training consolidation — one notebook, one config (2026-07-19)

Four `gislr.1.model.<arch>.ipynb` notebooks each carried their own `HYP` dict, so
"all else identical" — the premise of both the architecture comparison (§4) and
the subset ablations (§3.1) — was a manual chore across four files.

- [x] **`src/gislr.1.models.training.ipynb`** replaces all four: shared setup /
  config / split / feature-cache sections, then one markdown+code section per
  architecture, then cross-architecture comparison and the eval handoff. Each
  architecture section re-reads the config from disk, so it is independently
  re-runnable. The four old notebooks are removed (git history ≤ `2d7f668`).
- [x] **`src/config/gislr.training.json`** is the source of truth for every
  hyperparameter, read at run time by **`modules/model/config.py`**.
  Architectures inherit `shared`; a deviation must be an explicit `overrides`
  entry, surfaced by §2 of the notebook and by `TrainingConfig.overrides_for`.
  Validation rejects: unknown architectures, unknown per-arch keys, an override
  naming a key absent from `shared` (a typo can't become a silent no-op),
  missing required HYP keys, bad `coords`, wrong `schema_version`. All six
  failure modes tested.
- [x] `modules/model/train.py::train_from_config(arch)` — what each section
  calls; trains every subset for that architecture and prints the resolved
  hyperparameters plus any overrides.
- [x] Feature caches are built **once** for every (subset, coords) pair the
  config needs, instead of once per notebook.
- [x] `report.comparison_row` now includes `finished`, and the comparison
  section separates interrupted runs from finished ones — an interrupted run
  was otherwise indistinguishable from a bad architecture.

### 5.5.1 Bug this consolidation exposed — cnn1d's receptive field

- [x] **The hand-sync that made all four `HYP` dicts identical also flattened
  `cnn1d`'s `num_layers` from 5 to 2.** For `CausalConv1D` that parameter is the
  number of dilated conv *blocks*, i.e. the receptive field:
  `1 + (kernel-1) * sum(2^i)` = **125 frames at 5 blocks** (dilations 1,2,4,8,16,
  matching `MAX_SEQ_LEN=128`) but only **13 frames at 2**. A 13-frame window
  cannot see a whole sign.
  **This is almost certainly the whole explanation for cnn1d's ~0.54 vs ~0.75**,
  and it means the "1D-CNN is 20 points behind" reading in the 2026-07-19 log is
  an artifact, not an architecture result. Restored as an explicit
  `overrides: {"num_layers": 5}` in the config, with the reasoning in its
  `notes` field.
- [ ] **Re-run cnn1d** with the corrected receptive field before drawing any
  conclusion about the architecture (all three subsets; ~1.7M params now).
- [ ] Once re-run, revisit `docs/logs/daily/2026-07-19.md` §1.3, which currently
  reports the crippled numbers.
- [ ] Consider making the receptive field an explicit, asserted quantity in
  `CausalConv1D.__init__` (e.g. warn when it is much shorter than
  `MAX_SEQ_LEN`), so a future misconfiguration fails loudly rather than
  training quietly at a fraction of the intended context.

---

## 6. Evaluation, Export & Kaggle Submission (GISLR)

**Location:** `src/gislr.2.models.evaluation.ipynb` — the single place where
**all** GISLR model evaluation and submission happens. The `gislr.1.model.*`
notebooks are training drivers only; they no longer carry export code.

### 6.1 Evaluation notebook (2026-07-19)

- [x] **Export code removed from `gislr.1.model.gru.ipynb`** (§7, cells 13–16 —
  the only training notebook that had it) and rehomed, arch-generic, in
  `modules/model/export.py` + the evaluation notebook.
- [x] **DuckDB leaderboard**: glob every `data/models/*/meta.json`, filter
  `dataset = 'gislr'`, ordered by accuracy (canonical `overall_accuracy` first,
  falling back to the training-loop `train_val_acc` for un-evaluated runs).
  DuckDB reads the meta.json files directly — `index.csv` stays the committed
  snapshot, not the query path.
- [x] **Learning curves**: the best 5 models overlaid on one loss+accuracy figure
  (train and val), sourced from each run's `assets/history.json` so the figure
  never needs the gitignored checkpoints.
- [x] **Confusion matrices**: the top 5 runs individually (250×250), then one
  row-normalized confusion matrix aggregated over **all** evaluated gislr runs
  (where the models agree on a mistake, the confusion is a property of the
  data/labels, not the architecture — this is the Phase-1 §7.4 instrument).
- [x] **Most-confused pairs** table extracted from the aggregate matrix (feeds
  §7 Phase 1.4).
- [ ] Run the notebook (user) once the fresh-registry runs have canonical evals.

### 6.2 Supporting module work (2026-07-19)

- [x] `modules/model/architectures.py`: every arch gains `forward_full(x)` —
  a batch-1, unpacked, ONNX-friendly forward used only by export. Parity against
  the packed training forward is asserted at export time.
- [x] `modules/model/export.py`: arch-generic ONNX → TF SavedModel → TFLite chain
  + `submission.zip` packaging + a validation pass using the grader's exact
  calling convention (raw `(T, 543, 3)` with NaNs in, `(250,)` out). NaN→0 and
  the landmark-subset gather stay **inside** the exported graph.
- [x] **The ONNX route was abandoned; export now goes through a native Keras
  rebuild** (`modules/model/keras_export.py`, 2026-07-19). **All 4
  architectures export**, all under the 40 MB cap:

  | arch | tflite | keras parity | tflite parity |
  |---|---|---|---|
  | gru | 3.44 MB | 4.1e-06 | 3.8e-06 |
  | lstm | 4.48 MB | 3.5e-06 | 3.8e-06 |
  | bilstm | 11.06 MB | 3.2e-06 | 2.9e-06 |
  | cnn1d | 2.89 MB | 6.1e-06 | 5.4e-06 |

  Why the ONNX route was dropped — five distinct failures, in order:
  1. `onnx2tf` declares **no dependencies of its own** (needs `tf-keras`,
     `onnx-graphsurgeon`, `sng4onnx`, `ai-edge-litert`).
  2. `torch.nan_to_num`'s infinity handling emits an ONNX **`IsInf`** op it
     cannot convert.
  3. It permutes 3D input layouts (emitted a model wanting `(T, 3, 543)`);
     needs `-kat inputs`.
  4. **Op coverage**: only **lstm** converted at all — **gru** died inside
     onnx2tf's own GRU handler (`tf.split(tR, 3)` on a 256-dim, expects 768),
     **bilstm** on "mixed types to Tensor", **cnn1d** on a squeeze.
  5. The one arch that did convert then failed **TFLite** conversion on a
     malformed `Squeeze` onnx2tf generated (`squeeze_dims` size > 8).

  `onnx`, `onnx2tf`, `onnxruntime`, `onnx-graphsurgeon`, `sng4onnx`,
  `ai-edge-litert` and `tf-keras` are all **removed from `pyproject.toml`**.
- [x] **Two parity gates** guard the weight transfer (a gate-order or bias
  mistake would convert cleanly and predict garbage): `keras_parity` (rebuild vs
  `forward_full`) and `tflite_parity` (the **final .tflite** vs PyTorch through
  the whole deployed path, raw `(T, 543, 3)` with NaNs). Conventions handled:
  GRU gate reorder `[r,z,n]`→`[z,r,h]` with `reset_after=True`; LSTM's two
  biases summed into Keras' one; LayerNorm epsilon forced to PyTorch's 1e-5
  (Keras defaults to 1e-3); Conv1d `(out,in,k)`→`(k,in,out)`.
- [x] Two bugs the gates caught, both fixed:
  - **int8 quantization**: `tf.lite.Optimize.DEFAULT` shifted logits by ~1e-1.
    Now off by default — these models are 3–11 MB against a 40 MB cap, so it
    bought nothing (`export_tflite(quantize=...)` if that ever changes).
  - **Uninitialized resource variables**: saving the Keras-backed module
    directly produced a TFLite model that died at invoke with
    `READ_VARIABLE ... variable != nullptr` inside the RNN's WHILE loop.
    Weights are now frozen to constants — and since freezing via
    `from_concrete_functions` loses the output *name* the grader indexes by
    (`output["outputs"]`), the frozen function is re-wrapped in a module that
    re-declares the exact signature.
- [x] `modules/scripts/eval_gru.py` refactored so `evaluate_run(run_dir)` is
  importable (the CLI is a thin wrapper), and it now also writes
  `assets/val_predictions.npz` (labels + preds) — that file is what makes the
  confusion matrices cheap and reproducible.
- [x] `modules/model/train.py` writes `assets/history.json` every epoch.

### 6.3 Submission tracking — meta.json schema v3 (2026-07-19)

Kaggle allows **100 submissions/day**, and every trained model should be
evaluated on the real test set, so submission state has to be part of the run
record rather than something remembered by hand.

- [x] **Schema v3** adds a `submission` object, deliberately dataset-agnostic
  (`tested` means "scored on the held-out/official test set", whatever that
  means for the dataset — no Kaggle vocabulary in the required keys):

  ```json
  "submission": {
    "tested": false,        // scored on the official/held-out test set yet?
    "platform": null,       // "kaggle" | "local" | … (null until tested)
    "submitted_at": null,   // ISO-8601
    "public_score": null,   // official metric (Kaggle public LB accuracy)
    "private_score": null,
    "reference": null,      // kernel slug + version, run id, … — free-form
    "notes": ""
  }
  ```

- [x] `registry.py`: `SCHEMA_VERSION = 3`, `submission` in `REQUIRED_KEYS`,
  `SUBMISSION_DEFAULT`, and `write_meta` preserves an existing submission block
  across the training loop's per-epoch rewrites (same protection the canonical
  metrics already had). `mark_tested()` records a result.
- [x] `build_model_index.py` exposes `submission_tested` / `submission_platform`
  / `public_score` columns and a `--untested` filter.
- [x] Backfill: every pre-v3 `meta.json` in the registry gets the default
  submission block (idempotent migration in `registry.migrate_meta`).
- [x] **Submission queue**: DuckDB globs all meta.json, filters
  `dataset = 'gislr' AND submission.tested = false`, `LIMIT 100` — so each run of
  the cell submits only untested models and respects the daily cap by
  construction. Exports each to `submission.zip`, submits, marks `tested`.
- [x] Declare `kaggle` in `pyproject.toml` and `uv sync` — done 2026-07-19.
- [~] Submission mechanics. The `kaggle` **CLI** path submits through a Kaggle
  kernel (`-k <owner>/<notebook> -v <version>`), so each zip must be attached to
  a kernel version first — and there are currently **no credentials on this
  machine** (`~/.kaggle/kaggle.json` absent, `KAGGLE_USERNAME` unset), so a
  non-dry-run submit can only fail or hang.
- [~] **Kaggle MCP server** (offered 2026-07-19) — likely the better path: it
  exposes `mcp_kaggle_start_competition_submission_upload` +
  `kaggle_mcp_submit_to_competition`, i.e. **upload a file and submit it
  directly**, with no kernel-version dance. `.mcp.json` added at the repo root
  using the **OAuth variant** (`npx mcp-remote https://www.kaggle.com/mcp`, then
  call the server's `authorize` tool) so that **no API token is stored at rest**
  — `.mcp.json` is committed and is not gitignored. Token auth is the fallback:
  add `--header "Authorization: Bearer ${KAGGLE_MCP_TOKEN}"` to the args and set
  that variable in the environment, never inline.
  - [ ] Authorize the server from an **interactive** session (OAuth cannot run
    in a non-interactive one), then submit one model by hand end to end.
  - [ ] Once proven, decide whether `modules/model/submission.py::submit_run`
    keeps shelling out to the CLI or the notebook drives the MCP tools instead;
    the queue query and `mark_tested` bookkeeping are unaffected either way.
- [ ] **Security**: an API token was pasted in plaintext into a chat transcript
  on 2026-07-19 and must be treated as compromised — rotate it (Kaggle
  Settings → Generate New Token) and never commit one.

---

## 7. Breaking the ~73% Accuracy Plateau

**Context:** GRU, 1D-CNN, LSTM and BiLSTM all converge to ~70–74% on the FP_118
subset. Architecture-independent ⇒ the ceiling is upstream, in
features/normalization/data, not the model. Ordered by priority: diagnose first,
then fix the highest-leverage causes, then ablate to confirm what actually
helped.

### 7.1 Phase 1 — Diagnose before changing anything

Figure out whether this is overfitting, underfitting or a data/label ceiling
*before* spending compute on new features.

- [ ] Run the canonical eval on the current best checkpoint and fill in the
  pending metrics: overall / macro / median class accuracy, `n_classes_below_50pct`
  (§6.1 leaderboard surfaces all four).
- [ ] Train/val accuracy gap at the best epoch. **train ≫ val ⇒ overfitting**,
  prioritize §7.4 augmentation + regularization; **train ≈ val, both ~73% ⇒
  underfitting the real signal**, prioritize §7.2 normalization and §7.3 motion.
- [ ] Full 250×250 confusion matrix on the val set (§6.1 produces it).
- [ ] Top 20–30 most-confused class pairs by off-diagonal mass (§6.1).
- [ ] Manually inspect a few sequences per confused pair (landmark-trajectory
  visualization) and classify each pair as distinguished by: **handshape only**
  (hand landmark resolution/features insufficient) · **motion trajectory only**
  (velocity features should help most) · **location on/near the body** (absolute
  position must be preserved, not normalized away — note the tension with §7.2).
- [ ] Per-class sample count vs per-class accuracy. If low accuracy correlates
  with low sample count this is **class imbalance**, and the fix is
  oversampling/class weighting, *not* feature engineering — record this
  separately.
- [ ] Write the verdict up (overfitting / underfitting / imbalance / specific
  confusable pairs) — it decides which phase below runs next.

### 7.2 Phase 2 — Fix normalization (remove signer-appearance bias)

**Goal:** make all relative geometry invariant to a signer's physical proportions.

- [ ] Pick the reference landmark by **semantic identity** (shoulder-center =
  midpoint of left/right shoulder), never by "whatever index sits at position 0"
  in a reordered subset like FP_118.
- [ ] Check that reference's non-NaN rate across the dataset — needs to be ~100%;
  otherwise pick a more reliable landmark or define a fallback.
- [ ] Per-frame **translation** normalization: subtract the reference position
  from every landmark in that frame.
- [ ] Choose a **scale** reference that is also reliably detected (inter-shoulder
  distance, or a fixed bone such as shoulder→elbow).
- [ ] Per-frame **scale** normalization: divide by that distance, with an epsilon
  floor for frames where the scale landmarks are missing or coincident.
- [ ] [?] Optional log-compression on top of the scale-normalized values (not on
  raw distances) — empirical, not assumed to help.
- [ ] Re-verify the NaN policy after these transforms: missing landmarks must stay
  NaN-flagged, not silently become 0 through subtraction/division.
- [ ] Confirm the whole pipeline is **causal** — frame *t* uses only data from
  frame *t*. This feeds a streaming model; no lookahead, no whole-sequence stats.
- [ ] Re-run the motion-energy analysis (§1) on normalized coordinates and compare
  with the existing ME-126 findings — normalization may change which landmarks
  look important.

### 7.3 Phase 3 — Motion features

**Goal:** give the model velocity, which the 1st-place solution indicates was its
primary edge.

- [ ] Stacking order: normalize (§7.2) **first**, then compute motion features on
  the normalized coordinates — never on raw ones.
- [ ] Frame-to-frame velocity (first-order delta) as extra channels, causal
  (frame *t* uses *t* and *t−1* only).
- [ ] Decide and **document** the first-frame policy (zero-velocity vs repeating
  the first delta) — it changes the model's first observation.
- [ ] Concatenate position + velocity; record the new `feature_dim` (≈2×).
- [ ] Retrain **GRU** (fastest, ~15 min wall) on position+velocity vs the
  position-only baseline under identical hyperparameters.
- [ ] Only if position+velocity clearly wins: evaluate acceleration (delta-delta)
  as a third channel. Smooth positions first (reuse the Savitzky-Golay pipeline
  from §1) — raw double-differencing amplifies MediaPipe jitter. Track
  `feature_dim` growth and its inference-latency cost (TFLite, 100 ms/video budget).
- [ ] If acceleration doesn't measurably help, **drop it** rather than keeping it
  "just in case" — it costs training time and on-device latency.

### 7.4 Phase 4 — Augmentation (especially if Phase 1 showed overfitting)

- [ ] Jitter: random rotation / scale / translation **per sequence**, not per
  frame (per-frame destroys temporal coherence).
- [ ] Temporal resampling: randomly speed up / slow down a sequence to simulate
  different signing speeds.
- [ ] Random frame dropout + interpolation — simulates detection gaps, builds
  robustness to missing landmarks.
- [ ] Left/right mirroring for handedness (see §7.5 — decide augmentation vs
  canonicalization vs both).
- [ ] Retrain with augmentation and compare the **train/val gap** before vs after,
  to confirm it reduces overfitting rather than just adding noise.

### 7.5 Phase 5 — Handedness canonicalization

- [ ] Determine whether GISLR labels signer dominant hand, or whether it must be
  inferred (which hand has higher motion energy / detection rate per sequence).
- [ ] Decide the canonical handedness (e.g. always right-handed).
- [ ] Mirroring transform: flip x-coordinates **and** swap left/right landmark
  indices.
- [ ] [?] Deterministic canonicalization of all data, or random mirroring at train
  time? Different design choices with different generalization effects.
- [ ] Re-run the §7.1 confusion-matrix analysis afterwards to see whether
  handedness confusion specifically improved.

### 7.6 Phase 6 — Controlled ablations

**Goal:** isolate which change mattered instead of stacking everything and getting
an unattributable result.

- [ ] Fixed protocol for every arm: GRU, same split, same hyperparameters, same
  seed.
- [ ] **Arm A** — position-only, normalized (§7.2 alone vs the current baseline).
- [ ] **Arm B** — position + velocity, normalized (motion on top of fixed
  normalization).
- [ ] **Arm C** — velocity-only, normalized. Tests whether static pose/location is
  necessary at all. Prediction: underperforms B and possibly A (loss of
  static-hold and sign-location information) — run it to confirm, not to assume.
- [ ] **Arm D** (only if B > A) — position + velocity + acceleration, normalized.
- [ ] **Arm E** (only if §7.4 is implemented) — best arm from A–D + augmentation.
- [ ] Tabulate overall accuracy, macro accuracy and train/val gap for all arms
  side by side.
- [ ] Re-validate the winning feature combination on **one other architecture**
  (1D-CNN) to confirm the gain is feature-driven, not GRU-specific.
- [ ] Record the normalization + feature configuration **per run** in the meta.json
  schema, so future comparisons stay attributable and nobody has to re-litigate
  "was it the normalization or the velocity?" later. (Schema change — coordinate
  with §6.3; likely a `features` object alongside `subset`/`coords`.)

### 7.7 Phase 7 — Validate against prior work

- [ ] Cross-check the §7.2 scheme against the 1st-place solution's specific
  single-reference-point normalization — match the validated approach rather than
  a variant of it.
- [ ] Revisit ME-126 / motion-energy using normalized coordinates: motion energy
  computed on un-normalized data may have been biased by signer scale.
- [ ] Resolve the long-open ME-126 vs Kaggle-suggested-subset cross-validation
  (§1.6, §3) — landmark importance rankings may shift once normalization is fixed.
- [ ] Plain-language write-up for supervisor progress reporting in
  `docs/reports/plateau-breakout.md`: what Phase 1 diagnosed, what changed, what
  moved accuracy and by how much.

---

## Backlog / Someday

- [ ] (add unscoped ideas here as they come up, promote to a numbered section once
  they have a concrete plan)

---

*Last updated: July 22, 2026 (POPSIGN test-split extraction finished · all 4 train dataset parts downloaded, train manifest regeneration still pending · motion-energy and subset-comparison reports backfilled)*
