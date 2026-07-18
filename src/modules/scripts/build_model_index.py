"""Rebuild src/models/index.csv from the per-run metadata.json files, and query it.

Every training run folder (src/models/<dataset>/<architecture>/<timestamp>/)
carries a metadata.json — written by the training notebook (section 7 of
gislr.1.model.gru.ipynb) and updated in place by scripts/eval_gru.py when the
canonical per-class eval runs. This script flattens them into one table so
"best 3 gru runs on gislr" or "all ME_126 runs" is a one-liner instead of a
folder crawl.

Like eval_gru.py, this is self-contained and runs with CWD anywhere:

    python scripts/build_model_index.py                       # rebuild index.csv + print full leaderboard
    python scripts/build_model_index.py --dataset gislr --architecture gru --top 3
    python scripts/build_model_index.py --subset ME_126

Sorting uses val_acc = the canonical-eval overall accuracy when available
(eval_status "canonical"), falling back to the training-loop best val accuracy
(eval_status "pending"). Runs missing metadata.json are listed as warnings but
never block the rebuild.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
INDEX_CSV = MODELS_DIR / "index.csv"

# column order of index.csv: identity, then results, then config; the flattened
# hyp_* columns follow in whatever order the metadata files introduce them
LEAD_COLUMNS = [
    "dataset", "architecture", "run_id", "subset", "coords",
    "val_acc", "eval_status", "overall_accuracy", "macro_accuracy",
    "median_class_accuracy", "n_classes_below_50pct", "train_val_acc",
    "n_landmarks", "feature_dim", "n_params", "n_classes", "n_val", "split",
    "model_name", "streaming", "trained_date", "training_source",
    "training_regime", "epochs", "best_epoch", "early_stopped",
]


def load_runs() -> pd.DataFrame:
    rows, missing = [], []
    for run_dir in sorted(MODELS_DIR.glob("*/*/*/")):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            missing.append(run_dir)
            continue
        meta = json.loads(meta_path.read_text())
        for key, value in meta.pop("hyperparameters", {}).items():
            meta[f"hyp_{key}"] = value
        meta.pop("schema_version", None)
        meta["path"] = run_dir.relative_to(MODELS_DIR).as_posix()
        rows.append(meta)
    for run_dir in missing:
        print(f"WARNING: no metadata.json in {run_dir} — run not indexed")
    if not rows:
        raise SystemExit(f"no metadata.json found under {MODELS_DIR}")

    df = pd.DataFrame(rows)
    df["val_acc"] = df["overall_accuracy"].fillna(df["train_val_acc"])
    ordered = [c for c in LEAD_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in ordered and c not in ("notes", "path")]
    df = df[ordered + rest + ["notes", "path"]]
    return df.sort_values(["dataset", "architecture", "val_acc"],
                          ascending=[True, True, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", help="filter printed view by dataset (e.g. gislr)")
    ap.add_argument("--architecture", "--arch", dest="architecture",
                    help="filter printed view by architecture (e.g. gru)")
    ap.add_argument("--subset", help="filter printed view by landmark subset (e.g. ME_126)")
    ap.add_argument("--top", type=int, help="print only the top N rows of the filtered view")
    args = ap.parse_args()

    df = load_runs()
    df.to_csv(INDEX_CSV, index=False)
    print(f"wrote {INDEX_CSV} ({len(df)} runs)\n")

    view = df
    for col in ("dataset", "architecture", "subset"):
        if getattr(args, col):
            view = view[view[col].str.lower() == getattr(args, col).lower()]
    if args.top:
        view = view.head(args.top)

    show = ["dataset", "architecture", "run_id", "subset", "coords",
            "val_acc", "eval_status", "n_params", "trained_date"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(view[show].to_string(index=False))


if __name__ == "__main__":
    main()
