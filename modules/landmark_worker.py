import os
import numpy as np
import cv2
import mediapipe as mp_image_mod
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path


MODEL_PATH_STRING = "tools/mediapipe/tasks/holistic_landmarker.task"


OUTPUT_DIR = Path("./data/landmarks/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_landmarker: vision.HolisticLandmarker | None = None


def init_worker():
    """Runs once per worker process. Printing here is a diagnostic —
    if you never see these lines in the terminal, workers aren't
    bootstrapping at all, which is the 'stuck at 0' failure mode."""
    global _landmarker
    print(f"[pid {os.getpid()}] worker starting, loading model...", flush=True)

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH_STRING,
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    _landmarker = vision.HolisticLandmarker.create_from_options(options)
    print(f"[pid {os.getpid()}] worker ready.", flush=True)


def _extract_frame_arrays(result):
    # HolisticLandmarker returns flat lists here (one holistic subject per
    # video frame), unlike PoseLandmarkerResult which nests per detected
    # person. No [0] indexing needed — iterate the lists directly.
    pose = np.zeros((33, 4), dtype=np.float32)
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks):
            pose[i] = (lm.x, lm.y, lm.z, lm.visibility)

    left_hand = np.zeros((21, 3), dtype=np.float32)
    if result.left_hand_landmarks:
        for i, lm in enumerate(result.left_hand_landmarks):
            left_hand[i] = (lm.x, lm.y, lm.z)

    right_hand = np.zeros((21, 3), dtype=np.float32)
    if result.right_hand_landmarks:
        for i, lm in enumerate(result.right_hand_landmarks):
            right_hand[i] = (lm.x, lm.y, lm.z)

    return pose, left_hand, right_hand


def process_video(video_path):
    global _landmarker

    if _landmarker is None:
        raise RuntimeError("Worker not initialized with a landmarker instance.")

    video_path = Path(video_path)
    video_name = video_path.name.split(".")[0]
    output_file_name = f"{OUTPUT_DIR}/{video_name}.npz"
    output_file = Path(output_file_name)

    if output_file.exists():
        output_file.unlink(missing_ok=True)  # Remove existing file to avoid overwriting

    if not video_path.exists():
        return f"ERROR file not found: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"ERROR opening {video_name}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    pose_frames, lh_frames, rh_frames = [], [], []
    frame_idx = 0

    result = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(type(result))
                break

            timestamp_ms = int(frame_idx * (1000.0 / fps))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_image_mod.Image(
                image_format=mp_image_mod.ImageFormat.SRGB, data=rgb_frame
            )

            result = _landmarker.detect_for_video(mp_image, timestamp_ms)
            pose, lh, rh = _extract_frame_arrays(result)
            pose_frames.append(pose)
            lh_frames.append(lh)
            rh_frames.append(rh)

            frame_idx += 1

    finally:
        cap.release()

    if frame_idx == 0:
        return f"ERROR no frames decoded for {video_name}"

    np.savez_compressed(
        output_file_name,
        fps=fps,
        num_frames=frame_idx,
    )
    return f"OK {video_name} ({frame_idx} frames)"
