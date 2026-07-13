from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class GISLRRawDataset(Dataset):
    def __init__(
        self, dataframe, cache_dir, cache_prefix, rows_per_frame, max_seq_len=128
    ):
        self.df = dataframe.reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.rows_per_frame = rows_per_frame
        self.cache_dir = Path(cache_dir)
        self.cache_prefix = cache_prefix
        # small — safe to pickle as-is
        self.offsets = np.load(self.cache_dir / f"{cache_prefix}_offsets.npy")
        self._data = None  # opened lazily, per-process

    @property
    def data(self):
        # each worker process opens its own memmap handle on first access
        if self._data is None:
            self._data = np.load(
                self.cache_dir / f"{self.cache_prefix}_data.npy", mmap_mode="r"
            )
        return self._data

    def __getstate__(self):
        # strip the live memmap reference before pickling to a worker —
        # this is what avoids the >2GB pipe write on Windows spawn
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        start_frame, end_frame = self.offsets[i], self.offsets[i + 1]
        n_frames = end_frame - start_frame

        stride = self.rows_per_frame * 3
        start_flat = start_frame * stride
        end_flat = end_frame * stride

        arr = self.data[start_flat:end_flat].reshape(n_frames, self.rows_per_frame, 3)
        T = arr.shape[0]

        if T > self.max_seq_len:
            idxs = np.linspace(0, T - 1, self.max_seq_len).astype(int)
            arr = arr[idxs]
            T = self.max_seq_len

        feats = arr.reshape(T, -1)
        return torch.from_numpy(np.ascontiguousarray(feats)), T, int(row["label"])


def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)
    feats, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    feature_dim = feats[0].shape[1]
    max_len = lengths[0].item()
    padded = torch.zeros(len(feats), max_len, feature_dim, dtype=torch.float32)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    return padded, lengths, labels
