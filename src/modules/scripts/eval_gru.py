"""Per-class evaluation of a trained checkpoint on the canonical val split.

Handles every architecture the gislr.1.model.*.ipynb notebooks train (gru,
lstm, bilstm, cnn1d) by dispatching on the checkpoint's "arch" key (absent in
pre-2026-07-17 checkpoints -> gru). The model classes below must stay
byte-identical to their notebook definitions or state_dict loading breaks.

Reproduces the val split from the training notebooks (stratified 10%, seed 42)
and the dataset preprocessing (NaN->0, uniform subsample to max_seq_len frames).
Evaluates straight from the raw parquet files — no memmap cache needed.

Usage:
    python eval_gru.py <checkpoint.pt> <out_run_dir> [--landmarks <npy-file>]

If --landmarks is given (a .npy int array of landmark indices into 0..542),
only those landmarks are fed to the model. Coordinate channels follow the
checkpoint's "coords" key ("xyz" or "xy" — the z-drop ablation), so
feature_dim = len(coords) * n_landmarks must match.

Besides the per-class artifacts, the canonical numbers are written into the
run's metadata.json (eval_status -> "canonical"), so a subsequent
scripts/build_model_index.py rebuild picks them up in src/models/index.csv.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROWS_PER_FRAME = 543
MAX_SEQ_LEN = 128
BATCH = 256


class StreamingGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                  nn.Linear(hidden_size, num_classes))

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))


class StreamingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                  nn.Linear(hidden_size, num_classes))

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden_size), nn.Dropout(dropout),
                                  nn.Linear(2 * hidden_size, num_classes))

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        H = out.size(-1) // 2
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, H).to(out.device)
        fwd_last = out[..., :H].gather(1, idx).squeeze(1)
        bwd_first = out[:, 0, H:]
        return self.head(torch.cat([fwd_last, bwd_first], dim=-1))


class CausalConv1D(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.3, kernel_size=5):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch = input_size
        for i in range(num_layers):
            d = 2 ** i
            self.convs.append(nn.Sequential(
                nn.ConstantPad1d(((kernel_size - 1) * d, 0), 0.0),
                nn.Conv1d(ch, hidden_size, kernel_size, dilation=d)))
            self.norms.append(nn.LayerNorm(hidden_size))
            ch = hidden_size
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                  nn.Linear(hidden_size, num_classes))

    def forward(self, x, lengths):
        x = self.input_norm(x).transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x).transpose(1, 2)
            x = self.drop(self.act(norm(x))).transpose(1, 2)
        out = x.transpose(1, 2)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))


ARCHS = {"gru": StreamingGRU, "lstm": StreamingLSTM,
         "bilstm": BiLSTM, "cnn1d": CausalConv1D}


def load_video(path, landmarks=None, coords="xyz"):
    cols = list(coords)
    table = pq.read_table(path, columns=cols)
    data = np.column_stack([table.column(c).to_numpy() for c in cols])
    n = data.shape[0] // ROWS_PER_FRAME
    arr = data.reshape(n, ROWS_PER_FRAME, len(cols)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if landmarks is not None:
        arr = arr[:, landmarks, :]
    T = arr.shape[0]
    if T > MAX_SEQ_LEN:
        arr = arr[np.linspace(0, T - 1, MAX_SEQ_LEN).astype(int)]
        T = MAX_SEQ_LEN
    return arr.reshape(T, -1), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_run_dir")
    ap.add_argument("--landmarks", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    landmarks = np.load(args.landmarks) if args.landmarks else None

    data_dir = Path(kagglehub.competition_download("asl-signs"))
    sign2idx = json.loads((data_dir / "sign_to_prediction_index_map.json").read_text())
    idx2sign = {v: k for k, v in sign2idx.items()}

    train_df = pd.read_csv(data_dir / "train.csv")
    train_df["label"] = train_df["sign"].map(sign2idx)
    _, val_split = train_test_split(train_df, test_size=0.1,
                                    stratify=train_df["sign"], random_state=42)
    val_split = val_split.reset_index(drop=True)
    print(f"val split: {len(val_split)} videos")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hyp = ckpt["hyp"]
    arch = ckpt.get("arch", "gru")  # pre-2026-07-17 checkpoints are all GRU
    coords = ckpt.get("coords", "xyz")  # "xy" for the z-drop ablation runs
    model = ARCHS[arch](ckpt["feature_dim"], hyp["hidden_size"], hyp["num_layers"],
                        len(sign2idx), hyp["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"checkpoint: arch={arch} coords={coords} feature_dim={ckpt['feature_dim']} "
          f"best_val_acc={ckpt['best_val_acc']:.4f}")

    paths = [data_dir / p for p in val_split["path"]]
    labels_all = val_split["label"].to_numpy()
    preds_all = np.zeros(len(val_split), dtype=np.int64)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for b0 in range(0, len(paths), BATCH):
            chunk = list(ex.map(lambda p: load_video(p, landmarks, coords), paths[b0:b0 + BATCH]))
            order = np.argsort([-t for _, t in chunk])
            lengths = torch.tensor([chunk[i][1] for i in order])
            padded = torch.zeros(len(chunk), int(lengths[0]), chunk[0][0].shape[1])
            for j, i in enumerate(order):
                padded[j, : chunk[i][1]] = torch.from_numpy(chunk[i][0])
            logits = model(padded.to(device), lengths)
            pred = logits.argmax(-1).cpu().numpy()
            inv = np.empty_like(order); inv[order] = np.arange(len(order))
            preds_all[b0:b0 + len(chunk)] = pred[inv]
            if (b0 // BATCH) % 10 == 0:
                print(f"  {b0 + len(chunk)}/{len(paths)}  ({time.time() - t0:.0f}s)", flush=True)

    correct = preds_all == labels_all
    overall = correct.mean()
    df = pd.DataFrame({"label": labels_all, "correct": correct})
    per_class = (df.groupby("label")["correct"].agg(["mean", "count"])
                 .rename(columns={"mean": "accuracy", "count": "n_val"}))
    per_class["sign"] = per_class.index.map(idx2sign)
    per_class = per_class[["sign", "accuracy", "n_val"]].sort_values("accuracy")
    macro = per_class["accuracy"].mean()

    out = Path(args.out_run_dir)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    per_class.to_csv(out / "cache" / "per_class_accuracy.csv", index_label="label")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(per_class["accuracy"], bins=25, color="tab:blue", edgecolor="white")
    axes[0].axvline(overall, color="black", ls="--", label=f"overall {overall:.3f}")
    axes[0].axvline(macro, color="tab:red", ls=":", label=f"macro {macro:.3f}")
    axes[0].set_xlabel("per-class accuracy"); axes[0].set_ylabel("# classes")
    axes[0].set_title("Distribution of per-class accuracy (250 signs)"); axes[0].legend()
    worst = per_class.head(15)
    axes[1].barh(worst["sign"], worst["accuracy"], color="tab:red")
    axes[1].set_title("15 worst classes"); axes[1].set_xlabel("accuracy")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "assets" / "per_class_accuracy.png", dpi=110)

    summary = {
        "overall_accuracy": float(overall),
        "macro_accuracy": float(macro),
        "n_val": int(len(val_split)),
        "worst5": per_class.head(5)[["sign", "accuracy"]].values.tolist(),
        "best5": per_class.tail(5)[["sign", "accuracy"]].values.tolist(),
        "n_classes_below_50pct": int((per_class["accuracy"] < 0.5).sum()),
        "median_class_accuracy": float(per_class["accuracy"].median()),
    }
    (out / "cache" / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # promote the canonical numbers into the run's metadata.json (the record
    # scripts/build_model_index.py aggregates into src/models/index.csv)
    meta_path = out / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta.update({
            "eval_status": "canonical",
            "overall_accuracy": summary["overall_accuracy"],
            "macro_accuracy": summary["macro_accuracy"],
            "median_class_accuracy": summary["median_class_accuracy"],
            "n_classes_below_50pct": summary["n_classes_below_50pct"],
            "n_val": summary["n_val"],
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"updated {meta_path} (eval_status=canonical) — rebuild the index "
              f"with scripts/build_model_index.py")
    else:
        print(f"NOTE: no {meta_path} to update — create one so the run is "
              f"indexed by scripts/build_model_index.py")


if __name__ == "__main__":
    main()
