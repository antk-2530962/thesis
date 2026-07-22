# sign2speech

A sign language recognition system focused on **streaming, real-time inference** rather than offline-only accuracy. The end goal is a deployable pipeline that classifies signs frame-by-frame with low latency, trained on hand/pose/face landmark sequences extracted via MediaPipe Holistic.

## Datasets

| Dataset | Role | Source (via `kagglehub`) | Status |
|---|---|---|---|
| **GISLR** | Fast-iteration dataset — landmarks are pre-extracted | Kaggle competition `asl-signs` | Ready to preprocess/train immediately |
| **POPSIGN** | Primary dataset (~870GB raw video) | `mrgeislinger/popsign-asl-v1-0-game-train-{a-e,f-m,n-s,t-z}-signs` + `...-game-test` | All 4 train parts + test now downloaded (2026-07-21); test-split landmarks extracted (33,599/33,600), train extraction pending manifest regeneration |

Raw data is **not stored in this repository**. It's downloaded on demand via `kagglehub` into its default cache (`~/.cache/kagglehub/`), resolved lazily by `src/modules/paths.py` (`gislr_dir()` / `resolve_datasets()` — importing the module never downloads anything). GISLR is a Kaggle *competition* download, so the Kaggle account in use must have accepted the competition rules and `kagglehub` must be authenticated.

POPSIGN's extracted landmarks (the large intermediate artifact, pre-feature-caching) are written to a separate drive configured via `.env` — see [Environment setup](#environment-setup).

## Models

- **GRU** (unidirectional `StreamingGRU`) — the deployment baseline. Chosen because it supports true causal/streaming inference.
- **Landmark-subset ablations** — motivated by the motion-energy analysis and the Kaggle 1st-place cross-check ([docs/logs/daily/2026-07-15.md](docs/logs/daily/2026-07-15.md)): the **ME-126** subset (hands + upper-body pose + lips + eyes/nose) beat the full-543 baseline **73.73% vs 70.59% val accuracy with half the parameters** (v1-regime runs, pre-reset — see registry note below), independently confirmed by the discriminability probe comparison ([docs/logs/daily/2026-07-16.md](docs/logs/daily/2026-07-16.md)). Canonical subset index lists: `src/modules/dataset/landmark/subsets.py`. Remaining ablations in `TODO.md` §3.1.
- **Architecture benchmarks (runs pending)** — `StreamingLSTM` (streaming-viable), `BiLSTM` (offline-only accuracy reference — prices the causality gap), `CausalConv1D` (dilated causal 1D-CNN, streaming-viable) — all trained from `src/gislr.1.models.training.ipynb`, a thin driver over the shared stack in `src/modules/model/` with hyperparameters in `src/config/gislr.training.json`. Still planned: the full 1st-place 1D-CNN + Transformer port, ST-GCN, TCN, Conformer. See `TODO.md` §4.

## Model registry

Every training run gets **one flat folder** at `src/data/models/<run_id>/`, where `run_id` is the **seconds since the Unix epoch** at training start. Dataset, architecture and subset are *fields in `meta.json`* (and columns of `index.csv`), not directory levels.

| file | content |
|---|---|
| `meta.json` | the single machine-readable run record (schema below) |
| `best.pt` | best-val-accuracy checkpoint (gitignored) |
| `last.pt` | latest checkpoint, saved every epoch — auto-resume state (gitignored) |
| `assets/` | every other artifact: learning curves, per-class CSVs/plots, landmark indices, eval summary — each linked from `meta.json["assets"]` |

All `meta.json` files are flattened into the queryable **[src/data/models/index.csv](src/data/models/index.csv)** by `src/modules/scripts/build_model_index.py` (runs from anywhere), which also answers filter queries directly — e.g. `--dataset gislr --architecture gru --top 3` or `--subset ME_126`. The training driver writes `meta.json` **every epoch** with `eval_status: "pending"`; `src/modules/scripts/eval_gru.py` fills in the canonical eval numbers and flips it to `"canonical"`.

> **Registry reset (2026-07-18).** The registry was restarted empty when the flat epoch-seconds layout was adopted. The 8 pre-reset runs (GRU full-543 baseline 70.59%, ME-126 73.73%, the xy ablations, …) survive only in git history (`3668dae` and earlier, under the old `src/models/` tree) and in the daily reports — their weights are gone, so their canonical evals cannot be completed; the numbers remain as historical references.

### meta.json schema

**This section is the source of truth for the schema** (`schema_version: 3`; the machine-side key check lives in `modules/model/registry.py::REQUIRED_KEYS`). All keys are required; unknown extra keys are not written.

| key | type | content |
|---|---|---|
| `schema_version` | int | `3` |
| `run_id` | int | seconds since Unix epoch at training start = run folder name |
| `created` | str | ISO-8601 local timestamp derived from `run_id` |
| `dataset` | str | e.g. `"gislr"` |
| `architecture` | str | key into `modules.model.ARCHS`: `gru` / `lstm` / `bilstm` / `cnn1d` |
| `model_name` | str | class name, e.g. `"StreamingGRU"` |
| `streaming` | bool | streaming-viable? (`false` = offline-only reference, never deployable) |
| `subset` | str | landmark-subset name from `modules/dataset/landmark/subsets.py` |
| `coords` | str | `"xyz"` or `"xy"` (z-drop ablation) |
| `n_landmarks` / `feature_dim` / `n_classes` / `n_params` | int | model/input dimensions |
| `split` | object | `{strategy, random_state, n_val}` — the canonical split (`stratified 90/10`, seed 42, 9,448 val) |
| `training` | object | `{regime, source, epoch_cap, epochs_trained, best_epoch, early_stopped, finished, wall_time_min}` |
| `hyperparameters` | object | full `HYP` dict + `seed`, `max_seq_len`, `num_workers`, `loss`, `precision` |
| `metrics` | object | `{train_val_acc, eval_status ("pending"\|"canonical"), overall_accuracy, macro_accuracy, median_class_accuracy, n_classes_below_50pct}` — canonical fields are `null` until `eval_gru.py` runs and then survive training-loop rewrites |
| `checkpoints` | object | `{best: "best.pt", last: "last.pt"}` — run-dir-relative |
| `assets` | object | `{name: run-dir-relative path}` for every asset file, e.g. `{"landmarks": "assets/landmarks.npy", "history": "assets/history.json", "learning_curves": "assets/learning_curves.png"}` |
| `submission` | object | `{tested, platform, submitted_at, public_score, private_score, reference, notes}` — see below |
| `notes` | str | free-text run notes |

#### The `submission` block (schema v3)

`tested` answers one question: **has this run been scored on the official/held-out test set?**
It is deliberately *dataset-agnostic* — for GISLR that means a Kaggle submission landed
(`platform: "kaggle"`), for a dataset with no leaderboard it means a local held-out evaluation
ran (`platform: "local"`). No Kaggle vocabulary appears in the required keys, so POPSIGN runs
reuse the block unchanged.

It exists because Kaggle allows **100 submissions/day** and every trained model should
eventually be scored, so "which models still need submitting" has to be a query rather than
something remembered by hand:

```sql
dataset = 'gislr' AND submission.tested = false   ORDER BY accuracy DESC   LIMIT 100
```

`modules/model/submission.py::untested_runs` is exactly that query; `registry.mark_tested()`
flips the flag after a successful submission, and `registry.write_meta` protects the block from
the training loop's per-epoch rewrites (the same protection canonical eval metrics get).
Pre-v3 records are backfilled with the default (never submitted) by `registry.migrate_all`,
which `build_model_index.py` runs automatically.

## Reports & docs

See [docs/README.md](docs/README.md): day-by-day logs in `docs/logs/daily/<YYYY-MM-DD>.md`, weekly summaries in `docs/logs/weekly/<YYYY>-<WW>.md` (**weeks run Sunday → Saturday**, e.g. `2026-30.md` = 2026-07-19 → 07-25), standalone topic reports in `docs/reports/<topic>.md` — figures always under the matching `assets/` subfolder.

**Weekly summaries**

| week | dates | summary |
|---|---|---|
| 2026-29 | Jul 12 – Jul 18 | [docs/logs/weekly/2026-29.md](docs/logs/weekly/2026-29.md) — motion energy → discriminability (ME-126 wins on both, rho −0.12 between them) · xy beats xyz · registry v1 → the restructure (registry reset, pre-reset weights gone) · 4 architecture notebooks |
| 2026-30 | Jul 19 – Jul 25 | [docs/logs/weekly/2026-30.md](docs/logs/weekly/2026-30.md) — *in progress*: plateau diagnosed as **overfitting** · TFLite export working for all 4 archs (Keras rebuild) · training consolidated to one notebook + config · POPSIGN **test-split extraction finished** (33,599/33,600) and all 4 train dataset parts now downloaded (train manifest regeneration + bulk run still pending) |

**Standalone reports**

| report | contents |
|---|---|
| [docs/reports/motion-energy.md](docs/reports/motion-energy.md) | GISLR per-landmark motion analysis (three scopes, 94,477 videos): **~92% of pose "motion" is z-axis noise** · seeded 50-video samples reproduce the global ranking (rho 0.95+) · ME-126 keep/discard recommendation, cross-checked against the Kaggle 1st-place subset |
| [docs/reports/subset-comparison.md](docs/reports/subset-comparison.md) | Landmark-subset discriminability (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins** the 6-subset leaderboard (49.9% global probe) · discriminability ≈ uncorrelated with motion energy (rho −0.12) · probe difficulty profile tracks the trained GRU's (rho 0.640) |
| [docs/reports/confidence-tuning.md](docs/reports/confidence-tuning.md) | POPSIGN extraction-quality threshold sweep (**partial — 2 of 7 arms**): `min_hand_landmarks_confidence` is inert and the *pose* thresholds gate the hands · thresholds move hand detection by only ~0.02 · **the quality proxies are dominated by clip padding** — ~half of every clip is non-signing lead-in/lead-out, and detection is 0.85–0.94 within the signing span |

**Daily logs**

| date | report | contents |
|---|---|---|
| 2026-07-15 | [docs/logs/daily/2026-07-15.md](docs/logs/daily/2026-07-15.md) | GISLR landmark motion-energy analysis (z-noise finding, keep/discard recommendation) · 1st-place solution landmark cross-check · GRU training in detail: full-543 baseline vs ME-126 subset (+3.1 pts at half the parameters) |
| 2026-07-16 | [docs/logs/daily/2026-07-16.md](docs/logs/daily/2026-07-16.md) | Landmark-subset discriminability comparison (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins**; discriminability ≠ motion energy (rho −0.12) · POPSIGN extraction module + driver notebook (resource-capped, resumable) built & validated |
| 2026-07-17 | [docs/logs/daily/2026-07-17.md](docs/logs/daily/2026-07-17.md) | Registry tooling v1: per-run metadata + queryable index · fresh-run-folder policy · training regime **v2-plateau-300** · LSTM / BiLSTM / CausalConv1D benchmark notebooks built · eval script generalized (arch dispatch + xy mode) |
| 2026-07-19 | [docs/logs/daily/2026-07-19.md](docs/logs/daily/2026-07-19.md) | **Plateau diagnosed as overfitting** (train 90–99% vs val ~75%, gap 0.16–0.24) · most-confused pairs are semantic near-synonyms · GISLR **stage-2 evaluation/submission notebook** + arch-generic TFLite export + meta.json **schema v3** (`submission.tested`, submission queue as a query) · `docs/` split into logs/ vs reports/ · POPSIGN **confidence-tuning** first results ([report](docs/reports/confidence-tuning.md)) — thresholds barely matter, the quality proxies are measuring clip padding · POPSIGN extraction driver given a **CLI handoff** (`extract_popsign.py`) · **bulk test-split extraction running** (15,549/33,600, 1 failed, 1.46 videos/s) + the on-disk npz format recorded |
| 2026-07-18 | [docs/logs/daily/2026-07-18.md](docs/logs/daily/2026-07-18.md) | **Repo restructure**: unified `modules/model` training stack · flat epoch-seconds model registry + meta.json schema v2 (registry reset) · `data/{raw,cache,temp,models}` tree + temp-cleanup policy · docs daily/weekly/reports split · single-progress-bar training |

## Project structure

```
sign2speech/
├── README.md
├── TODO.md                   # living task list, organized by workstream
├── pyproject.toml            # uv-managed dependencies (Python >= 3.12, incl. torch cu130 via [tool.uv.sources])
├── uv.lock
├── .env                      # machine-specific config (POPSIGN_LANDMARKS_DRIVE) — not committed
├── docs/
│   ├── logs/                          # time-ordered: what happened when
│   │   ├── daily/<YYYY-MM-DD>.md      #   day-by-day logs (assets in logs/daily/assets/<date>/)
│   │   └── weekly/<YYYY>-<WW>.md      #   weekly summaries (weeks run Sunday → Saturday)
│   └── reports/<topic>.md             # standalone test/analysis reports (assets in reports/assets/<topic>/)
├── scripts/                  # housekeeping / misc only (NO project Python scripts — those live in src/modules/scripts/)
│   └── sys_disk_usage.ps1    # disk-usage helper (POPSIGN raw video is ~870GB)
└── src/                      # all code; notebooks assume the kernel CWD is src/
    ├── config/
    │   └── gislr.training.json   # SHARED TRAINING HYPERPARAMETERS (source of truth, not in any cell)
    ├── gislr.0.dataset.motion-energy.ipynb      # diagnostic: landmark motion-over-time analysis (TODO §1)
    ├── gislr.0.dataset.subset-comparison.ipynb  # diagnostic: landmark-subset discriminability comparison (TODO §3)
    ├── gislr.1.models.training.ipynb            # ALL GISLR training: one section per architecture (TODO §4)
    ├── gislr.2.models.evaluation.ipynb          # ALL GISLR evaluation + export + Kaggle submission (TODO §6)
    ├── popsign.0.dataset.extraction.ipynb       # POPSIGN landmark extraction driver (TODO §2)
    ├── popsign.0.dataset.confidence-tuning.ipynb # diagnostic: extraction-quality threshold sweep (TODO §2.3)
    ├── popsign.0.dataset.output-inspection.ipynb # diagnostic: what an extracted npz contains (read-only, safe during a live run)
    ├── popsign.1.mediapipe.ipynb                # stale — slated for retirement (TODO §0.1)
    ├── popsign.2.model.ipynb                    # label distribution, split, earlier TF experiments
    ├── popsign.3.pipeline.ipynb                 # end-to-end pipeline (stub)
    ├── modules/
    │   ├── paths.py                  # canonical tree constants (absolute, CWD-independent) + lazy dataset resolution + cleanup_temp()
    │   ├── model/                    # unified training stack behind gislr.1.models.training.ipynb
    │   │   ├── architectures.py      # StreamingGRU / StreamingLSTM / BiLSTM / CausalConv1D + ARCHS registry (single definition, shared with eval)
    │   │   ├── data.py               # canonical split, per-subset feature caches, in-RAM dataset
    │   │   ├── registry.py           # run folders (epoch seconds), meta.json writing, asset registration
    │   │   ├── train.py              # training driver: auto-resume, early stopping, ONE progress bar per run
    │   │   ├── report.py             # learning curves, per-epoch history, confusion matrices
    │   │   ├── export.py             # arch-generic TFLite export → submission.zip (parity-gated)
    │   │   ├── keras_export.py       # PyTorch → native Keras rebuild + weight transfer (the export route)
    │   │   └── submission.py         # DuckDB submission queue (untested runs, daily cap) + the Kaggle call
    │   ├── scripts/                  # project Python CLIs (run from anywhere — they bootstrap their own imports)
    │   │   ├── eval_gru.py           # canonical per-class eval of any run (all archs, xy/xyz); promotes meta.json to "canonical"
    │   │   ├── build_model_index.py  # flattens all meta.json files into data/models/index.csv + answers filter queries
    │   │   ├── extract_popsign.py    # POPSIGN extraction: pilot benchmark + resumable bulk run (worker pool — cannot run in a notebook)
    │   │   └── tune_confidence.py    # POPSIGN confidence-threshold sweep (same reason)
    │   └── dataset/landmark/
    │       ├── subsets.py            # canonical landmark-subset registry (FULL_543, ME_126, FP_118, …)
    │       ├── extraction.py         # MediaPipe Holistic video→landmark extraction (multiprocess, resource-capped, resumable)
    │       ├── quality.py            # extraction-quality proxies + composite score (confidence tuning)
    │       └── overlay.py            # landmark-on-video rendering + contact sheets (the visual quality test)
    └── data/                         # gitignored EXCEPT the model registry (data/models/)
        ├── raw/                      # extracted-from-source data (POPSIGN landmark npz: raw/popsign/{train,test}/)
        ├── cache/                    # reusable derived artifacts, one subtree per dataset:
        │   ├── gislr/features/       #   per-subset feature caches (GBs, shared across architecture notebooks)
        │   ├── gislr/motion_analysis/, gislr/subset_comparison/   # diagnostic-notebook caches
        │   ├── popsign/dataframes/, popsign/extraction/           # POPSIGN manifests + extraction measurements
        │   └── runs/                 #   run-dir pointer files (auto-resume)
        ├── temp/                     # throwaway scratch (e.g. POPSIGN pilot npz) — deleted after use (cleanup_temp())
        ├── external/                 # third-party assets (MediaPipe holistic_landmarker.task)
        └── models/                   # COMMITTED model registry (weights gitignored)
            ├── index.csv             # queryable table of all runs
            └── <run_id>/             # one flat folder per run (epoch seconds): meta.json + best.pt/last.pt + assets/
```

### Conventions

- **Notebooks are flat in `src/`, named `<dataset>.<stage>.<topic>.ipynb`** — dataset first, then a stage number ordering the pipeline, then what the stage does. No nested notebook folders.
- **Shared logic lives in `src/modules/`, not in notebooks; shared *parameters* live in a config file, not in cells.** All GISLR training is one notebook (`gislr.1.models.training.ipynb`) with a section per architecture, and every hyperparameter comes from **[`src/config/gislr.training.json`](src/config/gislr.training.json)** via `modules/model/config.py`. Architectures inherit the `shared` block; a deviation must be declared as an explicit `overrides` entry, which the notebook prints — so "all else identical" is enforced rather than maintained by hand. Project Python CLIs live in **`src/modules/scripts/`**; the root `scripts/` folder holds housekeeping/misc scripts only.
- **One flat registry folder per training run** at `src/data/models/<epoch-seconds>/` holding `meta.json` + `best.pt`/`last.pt` + `assets/`. A run's artifacts are never split across parallel trees; `index.csv` is the queryable view.
- **Docs**: daily logs in `docs/logs/daily/`, weekly summaries in `docs/logs/weekly/` (`<YYYY>-<WW>.md`, weeks Sunday → Saturday), standalone topic reports in `docs/reports/` — see `docs/README.md`.
- **Data placement policy**: extracted-from-source data → `data/raw/<dataset>/…`; reusable derived artifacts → `data/cache/<dataset>/…`; throwaway output → `data/temp/`, **deleted after use** (`modules.paths.cleanup_temp()`); third-party assets → `data/external/`. Raw kagglehub downloads never enter the repo — `modules/paths.py` resolves them lazily at call time (`gislr_dir()`, `resolve_datasets()`), never at import time.
- **POPSIGN's extracted landmarks go to `data/raw/popsign/{train,test}`**, rooted at the drive configured via `POPSIGN_LANDMARKS_DRIVE` in `.env` when set (falling back to `src/data/raw/`, gitignored). The extraction module (`modules/dataset/landmark/extraction.py`) resolves this in one place.
- **Paths are CWD-independent**: `modules/paths.py` anchors every constant to the `src/` directory, so notebooks (kernel CWD = `src/`) and the `modules/scripts/` CLIs (any CWD) agree on the same tree.
- **Canonical evaluation** (all GISLR runs must match to be comparable): stratified 90/10 split, `random_state=42`, 9,448-video val set, per-class accuracy from raw parquet — exactly what `modules/scripts/eval_gru.py` reproduces. A new run displaces a leaderboard entry only on this same split/metric.

## Environment setup

```bash
# Install dependencies (uv-managed, Python >= 3.12)
uv sync
```

Create a `.env` file at the project root (not committed) with:

```
POPSIGN_LANDMARKS_DRIVE=D:/    # or wherever the extraction-output drive is mounted
```

The Jupyter kernel must use this project's `uv`-managed virtual environment (`.venv`) and run with `src/` as its working directory — `import modules...` depends on it.

## Running the pipeline

**GISLR** (landmarks already extracted by Kaggle):

1. `src/gislr.0.dataset.motion-energy.ipynb` — *optional diagnostic*: per-video / per-category / global landmark motion-energy analysis. Executed end-to-end; findings in `docs/logs/daily/2026-07-15.md`. (Caches live at `data/cache/gislr/motion_analysis/`; the pre-restructure caches were cleared, so a re-run recomputes them.)
2. `src/gislr.0.dataset.subset-comparison.ipynb` — *diagnostic*: landmark-subset discriminability comparison across three scopes, scoring the subsets registered in `modules/dataset/landmark/subsets.py`. Findings in `docs/logs/daily/2026-07-16.md`.
3. `src/gislr.1.models.training.ipynb` — the training stage, **all four architectures in one notebook**: shared feature caches (`data/cache/gislr/features/`, skip-if-exists, built once and reused by every architecture), then one section per architecture (`StreamingGRU`, `StreamingLSTM`, `BiLSTM`, `CausalConv1D`) calling `modules.model.train_from_config`. Hyperparameters come from `src/config/gislr.training.json` (regime **v2-plateau-300**), not from cells. One epoch-seconds registry folder per (architecture, subset), `meta.json` + `assets/history.json` updated every epoch, one progress bar per run. `bilstm` is an **offline-only accuracy reference** (never a deployment candidate); the other three are streaming-viable. Per-class evaluation is handed off to `modules/scripts/eval_gru.py`. **Training only** — no export section.
5. `src/gislr.2.models.evaluation.ipynb` — **everything after training** (TODO §6): DuckDB leaderboard over all `meta.json` files, canonical-eval backfill, top-5 learning-curve overlay, per-run and aggregate confusion matrices + most-confused pairs, arch-generic TFLite export, and the Kaggle submission queue (untested runs only, 100/day cap). All GISLR evaluation and submission lives here and nowhere else.

**POPSIGN** (raw video, requires extraction first — in progress):

1. `src/popsign.0.dataset.confidence-tuning.ipynb` — *diagnostic, run before bulk extraction* (TODO §2.3): sweeps `HolisticLandmarker` thresholds over a seeded 50-video sample (5 classes × 10), scores each config with quality proxies (detection rates, jitter, gaps, rigid-bone variance) **and** renders 100 landmark-overlay frames weighted toward the worst detections. Establishes which thresholds actually matter — measured: `min_hand_landmarks_confidence` is inert; the *pose* thresholds gate the hands.
2. `src/popsign.0.dataset.extraction.ipynb` — the extraction driver, and where extraction actually runs: manifest generation + verification (`data/cache/popsign/dataframes/{train,test}.csv` — regenerated from the raw video tree, 30,867 train / 33,600 test), a **pilot batch (≤100 videos)** writing throwaway npz to `data/temp/popsign_pilot/` (auto-cleaned), then the resumable bulk run to `data/raw/popsign/{train,test}`. The worker pool runs in the kernel; MediaPipe's C++ logs are redirected per worker to `<out_dir>/<split>/_worker_stderr.log` so they never reach cell output. `modules/scripts/extract_popsign.py` (`pilot` / `run <split>`) is an optional CLI for unattended runs — same module, same manifests, resumable either way. See `TODO.md` §2.
3. `src/popsign.0.dataset.output-inspection.ipynb` — *diagnostic, safe to run **during** an extraction*: opens one completed npz and shows the saved format — keys/shapes/dtypes, the 543-row holistic group layout, per-group detection rates, a frame plot, and the reference npz→model-input loader. Read-only by construction: no writes to the landmarks tree, no worker pool, no MediaPipe import, `.tmp.npz` staging files excluded so a half-written video is never opened.

   **Saved format** — one `np.savez_compressed` per video at `<root>/data/raw/popsign/<split>/<label>/<video_id>.npz`, the label carried by the path rather than stored inside: `landmarks` `(T, 543, 3)` float16 (NaN where undetected), `fps` float32, `num_frames` int32. Row order is GISLR holistic order (face 0–467, left hand 468–488, pose 489–521, right hand 522–542), so the `modules/dataset/landmark/subsets.py` indices apply to POPSIGN unchanged. ~165 KB/video → **~5.4 GB** for the full test split.
4. `src/popsign.2.model.ipynb` — label-distribution analysis, stratified split, earlier TensorFlow experiments.
5. `src/popsign.3.pipeline.ipynb` — end-to-end pipeline (stub).

(`src/popsign.1.mediapipe.ipynb` is stale — slated for retirement per `TODO.md` §0.1.)

## Constraints & known limitations

- **Streaming-viability drives architecture choice** — the deployment path is the unidirectional GRU; bidirectional models (BiLSTM) can only ever be offline accuracy benchmarks.
- **TensorFlow GPU is not supported on native Windows** — training uses PyTorch (CUDA); TFLite conversion happens post-hoc by **rebuilding the trained model in native Keras and transferring the weights** (`modules/model/keras_export.py`), gated on numerical parity with the PyTorch model. The PyTorch → ONNX → `onnx2tf` route was tried and abandoned: onnx2tf failed on 3 of the 4 architectures (see `TODO.md` §6.2).
- **MediaPipe GPU delegate is Ubuntu-only** — landmark extraction runs on CPU on this Windows machine, parallelized across worker processes.
- Hardware: Windows 11, i7-14700K, RTX 4080 Super, 64GB DDR5 RAM.
