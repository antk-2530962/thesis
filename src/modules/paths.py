from pathlib import Path
from typing import Generic, TypedDict, TypeVar

import kagglehub

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
CACHE_DIR = Path("cache")


T = TypeVar("T")


class DatasetMap(TypedDict, Generic[T]):
    TRAIN: list[T]
    TEST: T
    GISLR: T


type DatasetIds = DatasetMap[str]
type Datasets = DatasetMap[Path]


DATASET_IDS: DatasetIds = {
    "TRAIN": [
        "mrgeislinger/popsign-asl-v1-0-game-train-a-e-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-f-m-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-n-s-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-t-z-signs",
    ],
    "TEST": "mrgeislinger/popsign-asl-v1-0-game-test",
    "GISLR": "asl-signs",
}


DATASETS: Datasets = {
    "TRAIN": [
        Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][0])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][1])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][2])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][3])),
    ],
    "TEST": Path(kagglehub.dataset_download(DATASET_IDS["TEST"])),
    "GISLR": Path(kagglehub.competition_download(DATASET_IDS["GISLR"])),
}
