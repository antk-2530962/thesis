"""MediaPipe Holistic landmark extraction from POPSIGN video files.

Replaces the deleted ``modules/data/landmark_worker.py`` and fixes its known
bugs (TODO §2.1): landmarks are actually written (all four groups), the model
path is resolved from the repo layout instead of a stale hardcoded string, and
the output root is wired to ``POPSIGN_LANDMARKS_DRIVE`` from ``.env``.

Output layout (one npz per video, GISLR-compatible holistic row order so
``modules.dataset.landmark.subsets`` indices apply unchanged):

    <root>/data/raw/popsign/<split>/<label>/<id>.npz
        landmarks   (T, 543, 3) float16 — face 0-467, left_hand 468-488,
                    pose 489-521, right_hand 522-542; NaN where undetected
        fps         float
        num_frames  int

``<root>`` = ``POPSIGN_LANDMARKS_DRIVE`` from the environment / repo ``.env``
when set, else the (gitignored) ``src/data/`` tree. All writes are atomic
(temp file + ``os.replace``); resumability follows the repo manifest pattern
(artifact written before ``done``; ``done`` skipped, ``failed`` retried) with
batched manifest saves — one JSON rewrite per video would dominate runtime at
~30K units.

Resource cap (the "≤70%" rule):

- **CPU** — worker count defaults to ``floor(0.70 × logical cores)`` and every
  worker pins its math libraries to one thread, so steady-state CPU stays
  around the cap.
- **RAM** — the job feeder blocks while system RAM usage is above
  ``ram_limit_pct`` (psutil-based backpressure).
- **GPU** — untouched: the MediaPipe GPU delegate is Ubuntu-only, extraction
  is CPU-bound on this machine by construction.

Typical use (from a notebook with CWD = ``src/``):

    from modules.dataset.landmark import extraction as ex
    videos = pd.read_csv("data/cache/dataframes/train.csv")   # file_path, label, id
    summary = ex.extract_dataset(videos, split="train")
"""

import json
import os
import time
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm.auto import tqdm

N_LANDMARKS = 543
GROUP_LAYOUT = (          # (holistic row offset, result attribute, group size)
    (0, "face_landmarks", 468),
    (468, "left_hand_landmarks", 21),
    (489, "pose_landmarks", 33),
    (522, "right_hand_landmarks", 21),
)

DEFAULT_CPU_FRACTION = 0.70   # ≤70% of logical cores
DEFAULT_RAM_LIMIT_PCT = 70.0  # feeder blocks above this system-RAM usage
DEFAULT_MODEL_PATH = Path("data/external/mediapipe/tasks/holistic_landmarker.task")
MANIFEST_SAVE_EVERY = 50      # results per manifest rewrite (30K-unit runs)
DEFAULT_FPS = 30.0            # cv2 sometimes reports fps=0; assume 30


# ============================================================
# Output root resolution (.env: POPSIGN_LANDMARKS_DRIVE)
# ============================================================

def _read_env_file() -> dict[str, str]:
    """Minimal KEY=VALUE parse of the repo-root .env (CWD is src/)."""
    for candidate in (Path(".env"), Path("../.env")):
        if candidate.exists():
            pairs = (line.split("=", 1) for line in candidate.read_text().splitlines()
                     if "=" in line and not line.lstrip().startswith("#"))
            return {k.strip(): v.strip() for k, v in pairs}
    return {}


def landmarks_root() -> Path:
    """`<POPSIGN_LANDMARKS_DRIVE>/data/raw/popsign`, or `data/raw/popsign` (gitignored) unset."""
    drive = os.environ.get("POPSIGN_LANDMARKS_DRIVE") or _read_env_file().get(
        "POPSIGN_LANDMARKS_DRIVE")
    base = Path(drive) if drive else Path(".")
    return base / "data" / "raw" / "popsign"


def default_n_workers(cpu_fraction: float = DEFAULT_CPU_FRACTION) -> int:
    return max(1, int((cpu_count() or 1) * cpu_fraction))


# ============================================================
# Worker process — one persistent HolisticLandmarker per process
# ============================================================

_MP = None            # mediapipe module (imported lazily, workers only)
_LANDMARKER = None    # per-process HolisticLandmarker (VIDEO mode)
_LANDMARKER_MODEL_PATH = str(DEFAULT_MODEL_PATH)
_TS_OFFSET = 0        # VIDEO mode needs strictly increasing timestamps


def _init_worker(model_path: str) -> None:
    """Pool initializer: single-threaded math libs + one landmarker per process."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import cv2
    cv2.setNumThreads(1)
    global _MP, _LANDMARKER, _LANDMARKER_MODEL_PATH
    import mediapipe as mp
    _MP = mp
    _LANDMARKER_MODEL_PATH = model_path
    _LANDMARKER = _make_landmarker(model_path)


def _make_landmarker(model_path: str):
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    return vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO))


def _result_to_frame(result) -> np.ndarray:
    """HolisticLandmarkerResult → (543, 3) float32, NaN where undetected."""
    frame = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for offset, attr, size in GROUP_LAYOUT:
        lms = getattr(result, attr, None)
        if lms:
            arr = np.asarray([[lm.x, lm.y, lm.z] for lm in lms[:size]],
                             dtype=np.float32)
            frame[offset:offset + len(arr)] = arr
    return frame


def _extract_one(job: tuple[str, str, str]) -> dict:
    """(video_path, out_path, video_id) → status dict. Runs in a worker process.

    Writes `<out>.npz` atomically; never raises (errors are reported in the
    returned dict so the driver can record them in the manifest).
    """
    import cv2
    global _TS_OFFSET, _LANDMARKER
    video_path, out_path, video_id = job
    t0 = time.time()
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        frames: list[np.ndarray] = []
        ts_ms = 0.0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            image = _MP.Image(image_format=_MP.ImageFormat.SRGB,
                              data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            try:
                result = _LANDMARKER.detect_for_video(image, _TS_OFFSET + round(ts_ms))
            except Exception:
                # timestamp/session hiccup — one fresh landmarker, then retry
                _LANDMARKER = _make_landmarker(_LANDMARKER_MODEL_PATH)
                _TS_OFFSET = 0
                result = _LANDMARKER.detect_for_video(image, round(ts_ms))
            frames.append(_result_to_frame(result))
            ts_ms += 1000.0 / fps
        cap.release()
        # next video in this process must keep timestamps increasing
        _TS_OFFSET += round(ts_ms) + 1000

        landmarks = (np.stack(frames).astype(np.float16) if frames
                     else np.empty((0, N_LANDMARKS, 3), dtype=np.float16))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, landmarks=landmarks,
                            fps=np.float32(fps), num_frames=np.int32(len(frames)))
        os.replace(tmp, out)
        return {"id": video_id, "status": "done", "artifact": str(out),
                "n_frames": len(frames), "fps": float(fps),
                "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        return {"id": video_id, "status": "failed", "artifact": None,
                "error": repr(e), "seconds": round(time.time() - t0, 2)}


# ============================================================
# Resumable driver
# ============================================================

def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "_manifest.json"


def load_manifest(out_dir: Path) -> dict:
    p = _manifest_path(out_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    """Atomic write — a crash mid-save can't corrupt the manifest."""
    p = _manifest_path(out_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, p)


def _throttled(jobs, ram_limit_pct: float):
    """Feed jobs to the pool only while system RAM usage is below the cap."""
    for job in jobs:
        while psutil.virtual_memory().percent >= ram_limit_pct:
            time.sleep(2.0)
        yield job


def pending_jobs(videos: pd.DataFrame, out_dir: Path,
                 manifest: dict) -> list[tuple[str, str, str]]:
    """Jobs still to run: not `done`-with-artifact and no npz already on disk.

    An existing npz without a manifest entry (crash between artifact write and
    manifest save) is adopted as `done`.
    """
    jobs = []
    for r in videos.itertuples(index=False):
        vid = str(r.id)
        out_path = out_dir / str(r.label) / f"{vid}.npz"
        entry = manifest.get(vid)
        if entry and entry["status"] == "done" and Path(entry["artifact"]).exists():
            continue
        if out_path.exists():
            manifest[vid] = {"status": "done", "artifact": str(out_path),
                             "timestamp": datetime.now(timezone.utc).isoformat(),
                             "note": "adopted existing npz"}
            continue
        jobs.append((str(r.file_path), str(out_path), vid))
    return jobs


def extract_dataset(
    videos: pd.DataFrame,
    split: str,
    out_root: Path | str | None = None,
    n_workers: int | None = None,
    cpu_fraction: float = DEFAULT_CPU_FRACTION,
    ram_limit_pct: float = DEFAULT_RAM_LIMIT_PCT,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    manifest_save_every: int = MANIFEST_SAVE_EVERY,
    limit: int | None = None,
    chunksize: int = 2,
    progress: bool = True,
) -> dict:
    """Extract landmarks for every video in `videos` (columns: file_path, label, id).

    Resumable: skips videos already `done`, retries `failed`. `limit` caps how
    many *pending* videos this call processes (pilot batches). Returns a summary
    dict incl. measured videos/s (basis for full-run ETAs).
    """
    out_root = Path(out_root) if out_root is not None else landmarks_root()
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"holistic task model not found: {model_path}")

    manifest = load_manifest(out_dir)
    n_before = sum(1 for v in manifest.values() if v["status"] == "done")
    jobs = pending_jobs(videos, out_dir, manifest)
    save_manifest(out_dir, manifest)   # persist adopted entries
    if limit is not None:
        jobs = jobs[:limit]
    n_workers = n_workers or default_n_workers(cpu_fraction)
    print(f"[{split}] {len(jobs)} pending of {len(videos)} videos "
          f"({n_before} already done), {n_workers} workers "
          f"(cap {cpu_fraction:.0%} CPU / {ram_limit_pct:.0f}% RAM) -> {out_dir}")
    if not jobs:
        return {"split": split, "n_done": 0, "n_failed": 0, "n_skipped": len(videos),
                "videos_per_s": float("nan"), "elapsed_s": 0.0,
                "n_workers": n_workers, "out_dir": str(out_dir)}

    t0 = time.time()
    n_ok = n_fail = since_save = 0
    frames_total = 0
    cpu_samples: list[float] = []
    psutil.cpu_percent(interval=None)   # prime the sampler
    bar = tqdm(total=len(jobs), desc=f"extract {split}", disable=not progress)
    with Pool(n_workers, initializer=_init_worker, initargs=(model_path,)) as pool:
        for res in pool.imap_unordered(_extract_one, _throttled(jobs, ram_limit_pct),
                                       chunksize=chunksize):
            res["timestamp"] = datetime.now(timezone.utc).isoformat()
            manifest[res.pop("id")] = res
            if res["status"] == "done":
                n_ok += 1
                frames_total += res.get("n_frames", 0)
            else:
                n_fail += 1
            since_save += 1
            if since_save >= manifest_save_every:
                save_manifest(out_dir, manifest)
                since_save = 0
            cpu = psutil.cpu_percent(interval=None)
            cpu_samples.append(cpu)
            bar.update(1)
            bar.set_postfix(cpu=f"{cpu:.0f}%",
                            ram=f"{psutil.virtual_memory().percent:.0f}%",
                            failed=n_fail)
    bar.close()
    save_manifest(out_dir, manifest)

    elapsed = time.time() - t0
    summary = {
        "split": split, "n_done": n_ok, "n_failed": n_fail,
        "n_skipped": len(videos) - len(jobs),
        "elapsed_s": round(elapsed, 1),
        "videos_per_s": round(len(jobs) / elapsed, 3) if elapsed else float("nan"),
        "frames_per_s": round(frames_total / elapsed, 1) if elapsed else float("nan"),
        "n_workers": n_workers,
        "cpu_mean_pct": round(float(np.mean(cpu_samples)), 1) if cpu_samples else None,
        "cpu_max_pct": round(float(np.max(cpu_samples)), 1) if cpu_samples else None,
        "out_dir": str(out_dir),
    }
    if n_fail:
        summary["failed_ids"] = [k for k, v in manifest.items()
                                 if v["status"] == "failed"][:20]
    return summary


# ============================================================
# Pilot: worker-count benchmark (parameter optimization)
# ============================================================

def benchmark_worker_counts(
    videos: pd.DataFrame,
    worker_counts: tuple[int, ...],
    videos_per_trial: int = 20,
    out_root: Path | str | None = None,
    **extract_kwargs,
) -> pd.DataFrame:
    """Time `extract_dataset` at several worker counts on disjoint video slices.

    Each trial writes under `<root>/_pilot/w<N>/` (disposable) and uses its own
    slice of `videos`, so no trial is sped up by another's cached results.
    Returns one row per worker count: videos/s, frames/s, CPU stats, ETA basis.
    """
    out_root = Path(out_root) if out_root is not None else landmarks_root()
    rows = []
    for k, n in enumerate(worker_counts):
        batch = videos.iloc[k * videos_per_trial:(k + 1) * videos_per_trial]
        if len(batch) < videos_per_trial:
            raise ValueError("not enough pilot videos for all trials — "
                             "lower videos_per_trial or pass more videos")
        summary = extract_dataset(batch, split=f"_pilot/w{n}", out_root=out_root,
                                  n_workers=n, **extract_kwargs)
        rows.append({"n_workers": n, "videos_per_s": summary["videos_per_s"],
                     "frames_per_s": summary["frames_per_s"],
                     "cpu_mean_pct": summary["cpu_mean_pct"],
                     "cpu_max_pct": summary["cpu_max_pct"],
                     "n_failed": summary["n_failed"],
                     "elapsed_s": summary["elapsed_s"]})
        print(rows[-1])
    return pd.DataFrame(rows)
