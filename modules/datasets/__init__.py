import kagglehub
from pathlib import Path


DATA_DIR = Path("data")

DATASET_IDS = {
    "TRAIN": [
        "mrgeislinger/popsign-asl-v1-0-game-train-a-e-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-f-m-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-n-s-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-t-z-signs",
    ],
    "TEST": "mrgeislinger/popsign-asl-v1-0-game-test",
    "ISLR": "asl-signs",
}


DATASETS = {
    "TRAIN": [
        Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][0])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][1])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][2])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][3])),
    ],
    "TEST": Path(kagglehub.dataset_download(DATASET_IDS["TEST"])),
    "ISLR": Path(kagglehub.competition_download(DATASET_IDS["ISLR"])),
}
