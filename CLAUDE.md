# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Notebook-driven ML research: streaming (frame-by-frame, causal) sign language recognition on MediaPipe landmark sequences. Two datasets — **GISLR** (Kaggle `asl-signs`, landmarks pre-extracted, fast iteration) and **POPSIGN** (~870GB raw video, landmark extraction still in progress). There is no app, no test suite, and no CI; notebooks in `src/` are the primary dev surface, with `README.md` / `TODO.md` / `docs/` as the committed record of results.

**Never run model training yourself.** When a task calls for training (or any long GPU run), build the appropriate notebook (following the conventions below) and hand it to the user to execute — they run it, you analyze the results afterwards.

**Bookend every task with `README.md` and `TODO.md`.** Before starting, read both to orient — TODO.md is the source of truth for workstreams and open items. After finishing, update both: mark TODO items done / in-progress, file follow-ups under the matching numbered workstream section (add a new `## N.` section only if none fits), and reflect any change to results, structure, or plans in the README.

## Environment & commands

```bash
uv sync                     # install deps (Python >= 3.12; torch cu130 via [tool.uv.sources])
```

- **Never `uv pip install` ad-hoc** — `uv sync` removes anything not declared in `pyproject.toml` (torch was once lost this way). Declare new deps in `pyproject.toml` instead.
- Run project Python via `.venv/Scripts/python.exe` (Windows). **CWD must be `src/`** for anything importing `modules` or touching data — all paths (`cache/`, `data/`, `checkpoints/`, `models/`) are relative to `src/`, and the Jupyter kernel must also use this venv with `src/` as its working directory.
- Exception: `scripts/eval_gru.py` is self-contained (reads raw parquet via kagglehub) and runs from anywhere:
  ```bash
  .venv/Scripts/python.exe scripts/eval_gru.py <checkpoint.pt> <out_run_dir> [--landmarks <indices.npy>]
  ```
- `import modules.paths` **triggers kagglehub downloads/resolution at import time** — requires an authenticated Kaggle account that has accepted the `asl-signs` competition rules.
- `.env` at repo root (not committed) holds `POPSIGN_LANDMARKS_DRIVE` — POPSIGN's extracted landmarks go to a separate drive, never into the repo.
- Type checking: both `pyrefly` and `ty` are installed (`[tool.ty.environment]` points at `./.venv`); TODO §0.2 says pick one — neither is canonical yet.
- No `jq` on this machine — parse notebook JSON with `.venv/Scripts/python.exe -c "import json; ..."`.

## Windows constraints (shape architecture decisions)

- **Training is PyTorch + CUDA.** TensorFlow GPU doesn't work on native Windows; TFLite export happens post-hoc via ONNX → TF SavedModel → TFLite (in `gislr.1.model.gru.ipynb`).
- **MediaPipe extraction is CPU-only** here (GPU delegate is Ubuntu-only), parallelized across worker processes.
- DataLoader multiprocessing (spawn) is fragile from ad-hoc scripts — in-RAM arrays with `num_workers=0` train GISLR at ~0.3 min/epoch, which is plenty. `GISLRRawDataset` (`src/modules/data/dataset.py`) is memmap-backed and strips its memmap handle in `__getstate__` specifically to survive Windows spawn.

## Architecture & conventions

- **Streaming viability drives everything.** The deployment path is the unidirectional `StreamingGRU`; bidirectional/offline models (BiLSTM etc.) are only ever accuracy references, never deployment candidates. New features (e.g. lag differences) must be causal.
- **Notebooks are flat in `src/`, named `<dataset>.<stage>.<topic>.ipynb`** — dataset, then a stage number ordering the pipeline, then the topic. No nested notebook folders. Pipeline order per dataset is described in `README.md` §"Running the pipeline".
- **One folder per training run**: `src/models/<dataset>/<architecture>/<timestamp>/` (timestamp = run start, `YYYYMMDD-HHMMSS`) containing `README.md` (training conditions + metrics), `data.md` (exact dataset/subset/split), `assets/` (PNGs), `cache/` (per-class CSV, `landmarks.npy`, eval summary, training script). Weights (`*.pt`) are gitignored — the docs are the record. `src/models/README.md` is the leaderboard of best runs per dataset × architecture.
- **Canonical evaluation** (all GISLR runs must match to be comparable): stratified 90/10 split, `random_state=42`, 9,448-video val set, per-class accuracy from raw parquet — exactly what `scripts/eval_gru.py` reproduces. A new run displaces a leaderboard entry only on this same split/metric.
- **Dated reports**: every substantial analysis gets `docs/<YYYY-MM-DD>.md` with figures under `docs/assets/<YYYY-MM-DD>/`.
- Raw data never enters the repo: `data/raw/` doesn't exist; `modules/paths.py` resolves kagglehub cache paths at runtime. `src/data/` and `src/cache/` are gitignored caches.

### How to build a notebook (and any big task)

Three core rules:

1. **Every cell is independently re-runnable.** Tweaking a parameter and re-running *one* cell must be enough to redo that subtask — never a series of cells. Put a subtask's tunables at the top of its own cell; have cells load their inputs from disk/cache rather than from live memory produced by other subtask cells. The only allowed dependencies are the setup cell (imports, shared constants, paths) and reusable-core function definitions — cells that are run once and don't change during param iteration.
2. **Well documented with markdown cells.** Title cell first: what the notebook does, pipeline stage vs standalone diagnostic, a table of the artifacts it produces (path per output), how resumability works, and design decisions vs the TODO spec it implements. Then a numbered `## N.` markdown cell per section, cross-referencing TODO items (e.g. `## 3. Scope 1 — per-video (TODO §1.3)`).
3. **Long tasks (extraction, training, …) save state as they go** — where that's sensible policy, an error/interrupt must never force a complete rerun. Use the existing manifest-driven resumable pattern (per-unit artifact written *before* marking `done` in a `cache/.../<scope>_manifest.json`; atomic saves via temp file + `os.replace`; `done` skipped, `failed` retried) rather than inventing a second one (TODO §2.2); training uses auto-resume checkpointing. Record seeded samples to JSON in the cache so re-runs are stable.

Supporting conventions (`src/gislr.0.dataset.motion-energy.ipynb` is the exemplar; the older `gislr.1` / `popsign.*` notebooks predate these):

- **Code cells open with a banner comment** (`# ===== / # <what this cell does> / # =====`).
- **One setup cell** right after the title: all imports together, then every shared tunable as an UPPERCASE constant with a short inline comment (including `SEED = 42`); end by printing the resolved data/cache paths.
- **Download only what's needed**: call `kagglehub.competition_download("asl-signs")` directly rather than `import modules.paths`, which resolves/downloads *every* dataset (incl. POPSIGN) at import time.
- **Heavy outputs go to `cache/`, not into cell outputs**: write per-unit parquets/PNGs to disk and display only a couple of representative figures inline (a notebook once hit 17MB from animation outputs and had to be stripped).
- Define reusable core functions once in an early section; every later section calls them rather than re-deriving logic.

## Key domain facts

- GISLR frames have **543 landmarks** (`ROWS_PER_FRAME`), xyz each; sequences uniformly subsampled to `MAX_SEQ_LEN=128`; NaN → 0.
- **ME-126** landmark subset (hands + upper-body pose {11–16, 23, 24} + lips + eyes/nose) beats the full-543 GRU baseline 73.73% vs 70.59% val acc at half the parameters. The z channel is mostly noise for pose landmarks (~92% of pose "motion"). Evidence and derivation: `docs/2026-07-15.md`; run history: `src/models/README.md`.

## Known broken / stale (verify before relying on)

Tracked in TODO §0.1 and §2.1 — highlights:
- `gislr.1.model.gru.ipynb` imports `modules.dataset`; the real module is `modules.data.dataset`. `popsign.1.mediapipe.ipynb` imports the deleted `modules.datasets`; use `modules.paths.DATASETS` (key `"GISLR"`, not `"ISLR"`).
- `modules/data/landmark_worker.py` has a critical bug: `np.savez_compressed` writes only `fps`/`num_frames`, **never the landmarks** — plus a stale hardcoded model path and output dir. Fix before any bulk POPSIGN extraction.
- `popsign.1.mediapipe.ipynb` currently contains stale GISLR exploration, not POPSIGN extraction; `popsign.0` and `popsign.3` are stubs.
- Only 1 of 4 POPSIGN train dataset downloads is enabled in `modules/paths.py` (others commented out).
