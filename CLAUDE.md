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
- Run project Python via `.venv/Scripts/python.exe` (Windows). Notebook kernels use this venv with **CWD = `src/`** (that's what makes `import modules...` resolve). Path constants themselves are CWD-independent: `modules/paths.py` anchors everything to the `src/` directory (`RAW_DIR`, `CACHE_DIR`, `TEMP_DIR`, `EXTERNAL_DIR`, `MODELS_DIR`, `MODEL_INDEX`).
- The project CLIs live in `src/modules/scripts/` and run from **any** CWD (they bootstrap `sys.path` themselves); the root `scripts/` folder is housekeeping-only (PowerShell etc., no project Python):
  ```bash
  .venv/Scripts/python.exe src/modules/scripts/eval_gru.py <run_dir>        # canonical per-class eval (all archs)
  .venv/Scripts/python.exe src/modules/scripts/build_model_index.py [...]   # rebuild data/models/index.csv + query
  ```
- Dataset resolution is **lazy**: importing `modules.paths` downloads nothing; call `modules.paths.gislr_dir()` for GISLR only, `resolve_datasets()` for everything (POPSIGN included — huge). Requires an authenticated Kaggle account that has accepted the `asl-signs` competition rules.
- `.env` at repo root (gitignored) holds secrets — currently `KAGGLE_MCP_TOKEN` (Kaggle MCP auth, TODO §6.3). `POPSIGN_LANDMARKS_DRIVE` (meant to send extracted POPSIGN landmarks to a separate drive, never into the repo) is **not currently set**, so extraction falls back to `src/data/raw/popsign` — set it before a bulk POPSIGN run if that's not where you want ~hundreds of GB to land.
- Type checking: `ty` is canonical (`[tool.ty.environment]` points at `./.venv`) — run `.venv/Scripts/ty.exe check` from `src/`. `pyrefly` was dropped 2026-07-22 (TODO §0.2).
- No `jq` on this machine — parse notebook JSON with `.venv/Scripts/python.exe -c "import json; ..."`.

## Git workflow

- **Create a Git commit after completing each logical unit of work** — don't batch unrelated changes into one commit.
- **Automatically stage modified files** with `git add` as part of committing.
- **Always format messages as [conventional commits](https://www.conventionalcommits.org/)** (e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).

## Windows constraints (shape architecture decisions)

- **Training is PyTorch + CUDA.** TensorFlow GPU doesn't work on native Windows; TFLite export happens post-hoc by rebuilding the model in native Keras and transferring weights (`modules/model/keras_export.py`, driven by `gislr.2.models.evaluation.ipynb`) — the ONNX/onnx2tf route was tried and abandoned (TODO §6.2).
- **MediaPipe extraction is CPU-only** here (GPU delegate is Ubuntu-only), parallelized across worker processes.
- DataLoader multiprocessing (spawn) is fragile from ad-hoc scripts — in-RAM arrays with `num_workers=0` train GISLR at ~0.3 min/epoch, which is plenty (`modules/model/data.py::SubsetArrayDataset`).

## Architecture & conventions

- **Streaming viability drives everything.** The deployment path is the unidirectional `StreamingGRU`; bidirectional/offline models (BiLSTM etc.) are only ever accuracy references, never deployment candidates. New features (e.g. lag differences) must be causal.
- **Shared logic lives in `src/modules/`, notebooks stay thin.** The unified training stack is `modules/model/` (`architectures.py` = the *single* definition of every model class, imported by both training and `eval_gru.py` — never redefine a model class elsewhere; `data.py`, `registry.py`, `train.py`, `report.py`). **All GISLR training is one notebook**, `gislr.1.models.training.ipynb`, with a markdown section per architecture. **Hyperparameters live in `src/config/gislr.training.json`, never in a cell** — architectures inherit the `shared` block and may deviate only via an explicit `overrides` entry (`modules/model/config.py` validates this and rejects an override naming a key that isn't in `shared`). This is what keeps the architecture comparison and subset ablations all-else-identical; four separate notebooks made that a manual chore and it silently broke (cnn1d's `num_layers=5` got flattened to 2, cutting its receptive field from 125 frames to 13).
- **Notebooks are flat in `src/`, named `<dataset>.<stage>.<topic>.ipynb`** — dataset, then a stage number ordering the pipeline, then the topic. No nested notebook folders. Pipeline order per dataset is described in `README.md` §"Running the pipeline".
- **Model registry** (reset empty 2026-07-18, repopulated since — 36 runs as of 2026-07-22; pre-reset runs live in git history ≤ `3668dae`, their weights are gone): one **flat** folder per run at `src/data/models/<run_id>/` with `run_id` = **epoch seconds** at training start, containing only `meta.json` + `best.pt`/`last.pt` (gitignored) + `assets/` (everything else, each linked from `meta.json["assets"]`). Dataset/architecture/subset are meta.json fields, not directory levels. **The meta.json schema's source of truth is `README.md` § "meta.json schema"** (v3; machine check `modules/model/registry.py::REQUIRED_KEYS`). The training driver rewrites `meta.json` every epoch; `index.csv` is regenerated by `build_model_index.py`.
- **Data placement policy** (`src/data/`, gitignored except `data/models/`): extracted-from-source → `data/raw/<dataset>/`; reusable derived artifacts → `data/cache/<dataset>/` (feature caches, analysis caches, manifests); throwaway output → `data/temp/`, **deleted after use** via `modules.paths.cleanup_temp()` (give producing notebooks a cleanup cell); third-party assets → `data/external/`. Raw kagglehub data never enters the repo.
- **Canonical evaluation** (all GISLR runs must match to be comparable): stratified 90/10 split, `random_state=42`, 9,448-video val set, per-class accuracy from raw parquet — exactly what `modules/scripts/eval_gru.py` reproduces (it also promotes the run's `meta.json` to `eval_status: "canonical"`). A new run displaces a leaderboard entry only on this same split/metric.
- **Docs** split two ways — **logs** (time-ordered) and **reports** (standalone findings): daily logs `docs/logs/daily/<YYYY-MM-DD>.md`; weekly summaries `docs/logs/weekly/<YYYY>-<WW>.md` (**weeks run Sunday → Saturday**, week 1 = the week containing Jan 1 — *not* ISO; `2026-30.md` = 2026-07-19 → 07-25, and the title always states the date range); standalone topic reports from a test/analysis (motion-energy, subset-comparison, confidence-tuning, …) `docs/reports/<topic>.md`. Figures under the matching `assets/` subfolder. Convention: `docs/README.md`.

### How to build a notebook (and any big task)

Three core rules:

1. **Every cell is independently re-runnable.** Tweaking a parameter and re-running *one* cell must be enough to redo that subtask — never a series of cells. Put a subtask's tunables at the top of its own cell; have cells load their inputs from disk/cache rather than from live memory produced by other subtask cells. The only allowed dependencies are the setup cell (imports, shared constants, paths) and `modules/` imports.
2. **Well documented with markdown cells.** Title cell first: what the notebook does, pipeline stage vs standalone diagnostic, a table of the artifacts it produces (path per output), how resumability works, and design decisions vs the TODO spec it implements. Then a numbered `## N.` markdown cell per section, cross-referencing TODO items (e.g. `## 3. Scope 1 — per-video (TODO §1.3)`).
3. **Long tasks (extraction, training, …) save state as they go** — an error/interrupt must never force a complete rerun. Use the existing manifest-driven resumable pattern (per-unit artifact written *before* marking `done` in a `data/cache/.../<scope>_manifest.json`; atomic saves via temp file + `os.replace`; `done` skipped, `failed` retried) rather than inventing a second one; training uses the driver's auto-resume checkpointing. Record seeded samples to JSON in the cache so re-runs are stable.

Supporting conventions:

- **Code cells open with a banner comment** (`# ===== / # <what this cell does> / # =====`).
- **One setup cell** right after the title: all imports together, then every shared tunable as an UPPERCASE constant with a short inline comment; end by printing the resolved data/cache paths.
- **Download only what's needed**: `modules.paths.gislr_dir()` for GISLR work — never `resolve_datasets()` unless POPSIGN raw video is genuinely required.
- **Progress reporting uses ONE bar per long task** (`tqdm.auto`), with everything (sub-progress, metrics) in that bar's description/postfix — no nested bars, no per-iteration prints (`modules/model/train.py` is the pattern).
- **Heavy outputs go to `data/cache/` (or a run's `assets/`), not into cell outputs**: write per-unit parquets/PNGs to disk and display only a couple of representative figures inline (a notebook once hit 17MB from animation outputs and had to be stripped).

## Key domain facts

- GISLR frames have **543 landmarks** (`ROWS_PER_FRAME`), xyz each; sequences uniformly subsampled to `MAX_SEQ_LEN=128`; NaN → 0 (constants: `modules/model/data.py`).
- **ME-126** landmark subset (hands + upper-body pose {11–16, 23, 24} + lips + eyes/nose) is the current leaderboard winner: **~75.7% canonical val acc** (gru and bilstm essentially tied, xy coords, `v2-plateau-300` regime, current 36-run registry as of 2026-07-22 — `src/data/models/index.csv` can lag the actual run folders, so regenerate it with `build_model_index.py` first, or glob `src/data/models/*/meta.json` directly, or run `gislr.2.models.evaluation.ipynb` for the live leaderboard number). Originally discovered pre-registry-reset at 73.73% vs the full-543 baseline's 70.59% (v1 regime, xyz, weights since gone) — that comparison is what motivated the subset in the first place, on the finding that the z channel is mostly noise for pose landmarks (~92% of pose "motion"). Evidence: `docs/logs/daily/2026-07-15.md` and `docs/logs/daily/2026-07-16.md`.

## Known broken / stale (verify before relying on)

This list rots within days (things here have gone stale mid-week before) — treat
`TODO.md` as the source of truth, not this file. Read `TODO.md` §0 (repo-restructure
follow-ups) and §2 (POPSIGN extraction) before relying on any specific broken/stale
claim; don't snapshot specifics here.
