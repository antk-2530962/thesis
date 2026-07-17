# sign2speech

A sign language recognition system focused on **streaming, real-time inference** rather than offline-only accuracy. The end goal is a deployable pipeline that classifies signs frame-by-frame with low latency, trained on hand/pose/face landmark sequences extracted via MediaPipe Holistic.

## Datasets

| Dataset | Role | Source (via `kagglehub`) | Status |
|---|---|---|---|
| **GISLR** | Fast-iteration dataset — landmarks are pre-extracted | Kaggle competition `asl-signs` | Ready to preprocess/train immediately |
| **POPSIGN** | Primary dataset (~870GB raw video) | `mrgeislinger/popsign-asl-v1-0-game-train-{a-e,f-m,n-s,t-z}-signs` + `...-game-test` | Requires landmark extraction first |

Raw data is **not stored in this repository**. It's downloaded on demand via `kagglehub` into its default cache (`~/.cache/kagglehub/`), resolved at runtime by `src/modules/paths.py`. GISLR is a Kaggle *competition* download, so the Kaggle account in use must have accepted the competition rules and `kagglehub` must be authenticated.

POPSIGN's extracted landmarks (the large intermediate artifact, pre-feature-caching) are written to a separate drive configured via `.env` — see [Environment setup](#environment-setup).

## Models

- **GRU** (unidirectional `StreamingGRU`) — the implemented baseline, built in `src/gislr.1.model.gru.ipynb`. Chosen because it supports true causal/streaming inference.
- **Landmark-subset ablations** — motivated by the motion-energy analysis and the Kaggle 1st-place cross-check ([docs/2026-07-15.md](docs/2026-07-15.md)): the **ME-126** subset (hands + upper-body pose + lips + eyes/nose) beats the full-543 baseline **73.73% vs 70.59% val accuracy with half the parameters**, and is independently confirmed by the discriminability probe comparison ([docs/2026-07-16.md](docs/2026-07-16.md), best of 6 registered subsets). Canonical subset index lists: `src/modules/dataset/landmark/subsets.py`. Remaining ablations in `TODO.md` §3.1.
- **Planned benchmarks** — 1D-CNN + Transformer (the GISLR 1st-place architecture), LSTM, BiLSTM (offline-only accuracy reference), ST-GCN, TCN, Conformer. See `TODO.md` §4 for scoping notes.

## Model registry

**Leaderboard of best-scoring models per dataset × architecture: [src/models/README.md](src/models/README.md).**

Every training run gets one folder at `src/models/<dataset>/<architecture>/<timestamp>/` containing:

| file | content |
|---|---|
| `README.md` | training conditions in detail + performance & evaluation metrics |
| `data.md` | exact dataset / landmark subset / split the model was trained on |
| `metadata.json` | machine-readable run record: dataset, architecture, subset, hyperparameters, accuracies, eval status |
| `assets/` | all PNGs referenced by the README (learning curves, per-class accuracy) |
| `cache/` | misc artifacts backing the README (per-class CSVs, landmark index lists, training script) |

Weights (`*.pt`) are gitignored — the documentation is the committed record.

All `metadata.json` files are flattened into the queryable **[src/models/index.csv](src/models/index.csv)** by `scripts/build_model_index.py` (runs from anywhere), which also answers filter queries directly — e.g. `--dataset gislr --architecture gru --top 3` or `--subset ME_126`. Training notebooks write `metadata.json` with `eval_status: "pending"`; `scripts/eval_gru.py` fills in the canonical eval numbers and flips it to `"canonical"`.

## Reports

Dated analysis/experiment reports live in [docs/](docs/) — one `docs/<YYYY-MM-DD>.md` per analysis day, figures under `docs/assets/<YYYY-MM-DD>/`.

| date | report | contents |
|---|---|---|
| 2026-07-15 | [docs/2026-07-15.md](docs/2026-07-15.md) | GISLR landmark motion-energy analysis (z-noise finding, keep/discard recommendation) · 1st-place solution landmark cross-check · GRU training in detail: full-543 baseline vs ME-126 subset (+3.1 pts at half the parameters) |
| 2026-07-16 | [docs/2026-07-16.md](docs/2026-07-16.md) | Landmark-subset discriminability comparison (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins**; discriminability ≠ motion energy (rho −0.12); FULL-543 worst even for a linear probe · POPSIGN extraction module + driver notebook (resource-capped, resumable) built & validated |
| 2026-07-17 | [docs/2026-07-17.md](docs/2026-07-17.md) | Registry tooling: per-run `metadata.json` + queryable `src/models/index.csv` (`scripts/build_model_index.py`), canonical-eval promotion via `eval_gru.py` · fresh-run-folder policy (completed runs never reused) · index snapshot: FP-118 train-loop 74.60% pending canonical eval |

## Project structure

```
sign2speech/
├── README.md
├── TODO.md                   # living task list, organized by workstream
├── pyproject.toml            # uv-managed dependencies (Python >= 3.12, incl. torch cu130 via [tool.uv.sources])
├── uv.lock
├── .env                      # machine-specific config (POPSIGN_LANDMARKS_DRIVE) — not committed
├── docs/                     # dated analysis/experiment reports
│   ├── <YYYY-MM-DD>.md       # one report per analysis day (e.g. 2026-07-15.md: motion-energy + 1st-place cross-check)
│   └── assets/<YYYY-MM-DD>/  # figures referenced by that report
├── scripts/
│   ├── eval_gru.py           # per-class val evaluation of a StreamingGRU checkpoint (run with CWD anywhere; reads raw parquet); updates the run's metadata.json
│   ├── build_model_index.py  # flattens all run metadata.json files into src/models/index.csv + answers filter queries (run with CWD anywhere)
│   └── sys_disk_usage.ps1    # disk-usage helper (POPSIGN raw video is ~870GB)
└── src/                      # all code; notebooks assume the kernel CWD is src/
    ├── gislr.0.dataset.motion-energy.ipynb  # diagnostic: landmark motion-over-time analysis (TODO §1) — executed, see docs/2026-07-15.md
    ├── gislr.0.dataset.subset-comparison.ipynb  # diagnostic: landmark-subset discriminability comparison (TODO §3) — F-ratio/MI/probe across 3 scopes
    ├── gislr.0.competition.entry.1st.ipynb  # reference: Kaggle GISLR 1st-place solution (1D-CNN + Transformer, 118-landmark subset).
    │                                        #   NOTE: 15MB of animation outputs stripped for repo size; full copy in src/cache/ (gitignored) or Kaggle discussion 406978
    ├── gislr.1.model.gru.ipynb              # per-subset feature caches → all-else-identical GRU runs (top-3 probe subsets) → run docs → (deferred) TFLite export
    ├── gislr.1.model.bilstm.ipynb           # BiLSTM offline accuracy reference (stub — TODO §4)
    ├── popsign.0.dataset.extraction.ipynb   # POPSIGN landmark extraction driver: manifests → pilot batch (param tuning + speed) → resumable bulk run (TODO §2)
    ├── popsign.1.mediapipe.ipynb            # stale (early GISLR motion-energy exploration) — superseded, to be retired (TODO §0.1)
    ├── popsign.2.model.ipynb                # label distribution, train/val/test split, earlier TF model experiments
    ├── popsign.3.pipeline.ipynb             # end-to-end pipeline (stub)
    ├── modules/
    │   ├── paths.py                         # kagglehub downloads + shared Path constants (DATASETS, DATA_DIR, CKPT_DIR, CACHE_DIR)
    │   └── dataset/
    │       └── landmark/
    │           ├── subsets.py               # canonical landmark-subset registry (FULL_543, ME_126, FP_118, …) + holistic index helpers
    │           └── extraction.py            # MediaPipe Holistic video→landmark extraction (multiprocess, resource-capped, resumable)
    ├── data/                                # gitignored — caches & external assets, never raw datasets
    │   ├── cache/dataframes/                # POPSIGN video manifests (train.csv / test.csv: file_path, label, id)
    │   └── external/mediapipe/tasks/holistic_landmarker.task
    ├── cache/                               # gitignored, created at runtime — GISLR memmapped feature caches, motion-analysis cache
    └── models/
        ├── README.md                        # leaderboard: best-scoring model per dataset × architecture
        ├── index.csv                        # queryable table of all runs (generated by scripts/build_model_index.py from the metadata.json files)
        └── <dataset>/<architecture>/<timestamp>/   # one folder per training run (timestamp = run start, YYYYMMDD-HHMMSS)
            ├── README.md                    # training conditions in detail + performance & evaluation metrics
            ├── data.md                      # exact dataset / landmark subset / split the model was trained on
            ├── metadata.json                # machine-readable run record (subset, hyperparameters, accuracies, eval status)
            ├── gru_best.pt                  # weights (gitignored)
            ├── assets/                      # PNGs referenced by the README (learning curves, per-class accuracy)
            └── cache/                       # misc items backing the README (per-class CSV, landmark list, eval summary)
```

### Conventions

- **Notebooks are flat in `src/`, named `<dataset>.<stage>.<topic>.ipynb`** — dataset first, then a stage number ordering the pipeline, then what the stage does. No nested notebook folders.
- **One folder per training run** at `src/models/<dataset>/<architecture>/<timestamp>/`, holding weights, docs (`README.md` + `data.md`), plots (`assets/`) and eval artifacts (`cache/`) together. A run's artifacts are never split across parallel trees. `src/models/README.md` tracks the best run per dataset × architecture; `src/models/index.csv` (regenerated by `scripts/build_model_index.py` from the per-run `metadata.json` files) indexes every run for filter/sort queries.
- **Dated reports in `docs/`** — every substantial analysis/experiment gets a `docs/<YYYY-MM-DD>.md` write-up with its figures under `docs/assets/<YYYY-MM-DD>/`.
- **Raw downloads never enter the repo.** Raw GISLR/POPSIGN data lives in `kagglehub`'s cache and is resolved at runtime by `modules/paths.py` — never duplicated into the repo.
- **POPSIGN's extracted landmarks go to `data/raw/popsign/{train,test}`**, rooted at the drive configured via `POPSIGN_LANDMARKS_DRIVE` in `.env` when set (falling back to `src/data/`, which is gitignored). The extraction module (`modules/dataset/landmark/extraction.py`) resolves this in one place.
- **Everything runs from `src/`** — `modules/paths.py` and the notebooks use relative paths (`data/`, `cache/`, `checkpoints/`), and `import modules...` resolves because the notebooks sit next to `modules/`. The Jupyter kernel's working directory must point at `src/`.

## Environment setup

```bash
# Install dependencies (uv-managed, Python >= 3.12)
uv sync
```

Create a `.env` file at the project root (not committed) with:

```
POPSIGN_LANDMARKS_DRIVE=D:/    # or wherever the extraction-output drive is mounted
```

The Jupyter kernel must use this project's `uv`-managed virtual environment (`.venv`) and run with `src/` as its working directory — both the `modules` import and every relative data/cache/checkpoint path depend on it.

## Running the pipeline

**GISLR** (landmarks already extracted by Kaggle):

1. `src/gislr.0.dataset.motion-energy.ipynb` — *optional diagnostic*: per-video / per-category / global landmark motion-energy analysis. Not part of the training pipeline. Executed end-to-end; findings and the landmark keep/discard recommendation are in `docs/2026-07-15.md`.
2. `src/gislr.0.dataset.subset-comparison.ipynb` — *diagnostic*: landmark-subset discriminability comparison (ANOVA F-ratio, mutual information, probe classifier) at three scopes (10 videos of one class / all videos of 10 classes / global with per-class stats), scoring the subsets registered in `modules/dataset/landmark/subsets.py`.
3. `src/gislr.0.competition.entry.1st.ipynb` — *reference*: the Kaggle 1st-place solution (1D-CNN + Transformer). Kept for the planned architecture port (TODO §4) and for its 118-landmark subset, which the motion-energy findings are cross-checked against.
4. `src/gislr.1.model.gru.ipynb` — the training stage (overhauled 2026-07-16): builds one feature cache per landmark subset (skip-if-exists, atomic writes), then trains one all-else-identical `StreamingGRU` run per subset in `TRAIN_SUBSETS` (default: the top-3 discriminability subsets `ME_126`, `ME_132`, `FP_118`) with per-subset auto-resume, one registry run folder each (always a fresh timestamped folder per training start — completed runs are never reused, only interrupted ones are resumed), and auto-generated `data.md`/`README.md`/`metadata.json` drafts. Per-class evaluation is handed off to `scripts/eval_gru.py` (commands printed by the notebook); the ONNX → TensorFlow → TFLite export chain is kept as a deferred, run-folder-parameterized final section for the later Kaggle-submission step.

**POPSIGN** (raw video, requires extraction first — in progress):

1. `src/popsign.0.dataset.extraction.ipynb` — the extraction driver: video manifest generation/verification (`data/cache/dataframes/`), a **pilot batch (≤100 videos)** for extraction-parameter optimization and speed measurement, then the resumable bulk run via `modules/dataset/landmark/extraction.py`. See `TODO.md` §2.
2. `src/popsign.2.model.ipynb` — label-distribution analysis, stratified train/val/test split, and earlier TensorFlow model experiments.
3. `src/popsign.3.pipeline.ipynb` — end-to-end pipeline (stub).

(`src/popsign.1.mediapipe.ipynb` is stale — it holds early GISLR exploration, superseded by `gislr.0.dataset.motion-energy.ipynb`, and is slated for retirement per `TODO.md` §0.1.)

## Constraints & known limitations

- **Streaming-viability drives architecture choice** — the deployment path is the unidirectional GRU; bidirectional models (BiLSTM) can only ever be offline accuracy benchmarks.
- **TensorFlow GPU is not supported on native Windows** — training uses PyTorch (CUDA); TFLite conversion happens post-hoc via ONNX → TensorFlow SavedModel → TFLite.
- **MediaPipe GPU delegate is Ubuntu-only** — landmark extraction runs on CPU on this Windows machine, parallelized across worker processes.
- Hardware: Windows 11, i7-14700K, RTX 4080 Super, 64GB DDR5 RAM.
