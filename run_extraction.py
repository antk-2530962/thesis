import os
import time
import multiprocessing as mp
import pandas as pd
from tqdm.auto import tqdm

from modules.landmark_worker import init_worker, process_video, OUTPUT_DIR


def main():

    TRAIN_DATAFRAME_PATH = "data/dataframes/train.csv"
    NUM_WORKERS = 16

    BAR_FORMAT = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[elapsed: {elapsed}, eta: {remaining}, rate: {rate_fmt}]"
    )
    df = pd.read_csv(TRAIN_DATAFRAME_PATH)
    all_video_paths = df["file_path"].to_list()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    samples = all_video_paths[:10]

    total_logical = os.cpu_count()
    print(
        f"Workers: {NUM_WORKERS} / {total_logical} logical threads available "
        f"({NUM_WORKERS / total_logical:.0%} utilization if fully scheduled)."
    )

    start_time = time.time()
    completed = 0

    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        with tqdm(
            total=len(samples),
            desc="Extracting landmarks",
            unit="video",
            bar_format=BAR_FORMAT,
        ) as pbar:
            for result in pool.imap_unordered(process_video, samples):
                completed += 1
                pbar.set_postfix_str(result, refresh=False)
                pbar.update(1)

    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    print(
        f"Done: {completed}/{len(samples)} videos in {elapsed / 60:.1f} min "
        f"({rate:.2f} videos/sec, {rate * 3600:.0f} videos/hr)."
    )


if __name__ == "__main__":
    main()
