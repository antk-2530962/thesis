from pathlib import Path

import kagglehub

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
CACHE_DIR = Path("cache")


DATASET_IDS: dict[str, list[str] | str] = {
    "TRAIN": [
        "mrgeislinger/popsign-asl-v1-0-game-train-a-e-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-f-m-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-n-s-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-t-z-signs",
    ],
    "TEST": "mrgeislinger/popsign-asl-v1-0-game-test",
    "GISLR": "asl-signs",
}


DATASETS: dict[str, list[Path] | Path] = {
    "TRAIN": [
        Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][0])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][1])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][2])),
        # Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][3])),
    ],
    "TEST": Path(kagglehub.dataset_download(DATASET_IDS["TEST"])),
    "GISLR": Path(kagglehub.competition_download(DATASET_IDS["GISLR"])),
}
