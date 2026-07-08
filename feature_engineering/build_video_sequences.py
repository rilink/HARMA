"""Builds raw VideoMAE sequence data: one (15, 768) window per 1-second IMU window.
Writes X_video_raw.npy + meta_video_raw.csv, used by run_adversarial.py and
run_imu_temporal_pca.py. Run from repo root: python feature_engineering/build_video_sequences.py
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR      = REPO_ROOT / "data"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed_data"

N_VIDEO_FRAMES = 15
EMBED_DIM      = 768

activities = {
    "null": 0,
    "jogging": 1,
    "jogging (rotating arms)": 2,
    "jogging (skipping)": 3,
    "jogging (sidesteps)": 4,
    "jogging (butt-kicks)": 5,
    "stretching (triceps)": 6,
    "stretching (lunging)": 7,
    "stretching (shoulders)": 8,
    "stretching (hamstrings)": 9,
    "stretching (lumbar rotation)": 10,
    "push-ups": 11,
    "push-ups (complex)": 12,
    "sit-ups": 13,
    "sit-ups (complex)": 14,
    "burpees": 15,
    "lunges": 16,
    "lunges (complex)": 17,
    "bench-dips": 18,
}


def _centered_window(v_start, v_end, n_frames, total_len):
    """Extract exactly n_frames centered within [v_start, v_end)."""
    span = v_end - v_start
    if span <= n_frames:
        center = (v_start + v_end) // 2
        start = max(0, center - n_frames // 2)
    else:
        start = v_start + (span - n_frames) // 2
    end = min(total_len, start + n_frames)
    start = max(0, end - n_frames)
    return start, end


def list_train_subjects():
    return sorted(p.stem for p in (DATA_DIR / "train" / "inertial_feat").glob("*.csv"))


def _load_inertial(sbj_name):
    df = pd.read_csv(DATA_DIR / "train" / "inertial_feat" / f"{sbj_name}.csv", low_memory=False)
    df["label"] = df["label"].fillna("null")
    df["label_encoded"] = df["label"].map(activities)
    return df


def _windows_meta_for_subject(sbj_name):
    """Pass 1: window boundaries + labels only, no video data loaded."""
    inertial_df = _load_inertial(sbj_name)
    video_len = np.load(DATA_DIR / "train" / "videomae_feat" / f"{sbj_name}.npy", mmap_mode="r").shape[0]
    ratio = len(inertial_df) / video_len

    windows = []
    for start in range(0, len(inertial_df) - 49, 25):
        chunk = inertial_df.iloc[start:start + 50]
        if len(chunk) < 50:
            break
        label_counts = Counter(chunk["label"].tolist())
        most_common_label, count = label_counts.most_common(1)[0]
        if count <= 40:
            continue
        label_encoded = activities[most_common_label]
        sbj_id = chunk["sbj_id"].iloc[0]

        v_start = int(start / ratio)
        v_end = max(v_start + 1, int((start + 50) / ratio))
        v_start, v_end = _centered_window(v_start, v_end, N_VIDEO_FRAMES, video_len)

        windows.append({"sbj_id": sbj_id, "label_encoded": label_encoded, "v_start": v_start, "v_end": v_end})
    return windows


def build_all(subject_names):
    print("Pass 1: computing window metadata (no video loading)...")
    all_windows = []
    subject_window_ranges = {}
    for sbj_name in subject_names:
        start_idx = len(all_windows)
        windows = _windows_meta_for_subject(sbj_name)
        all_windows.extend(windows)
        subject_window_ranges[sbj_name] = (start_idx, len(all_windows))
        print(f"  {sbj_name}: {len(all_windows)} rows so far")

    n_total = len(all_windows)
    size_gb = n_total * N_VIDEO_FRAMES * EMBED_DIM * 4 / 1e9
    print(f"Total windows: {n_total} - allocating memmap (~{size_gb:.1f} GB on disk)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_video_path = PROCESSED_DIR / "X_video_raw.npy"
    X_video = np.lib.format.open_memmap(
        X_video_path, mode="w+", dtype=np.float32, shape=(n_total, N_VIDEO_FRAMES, EMBED_DIM)
    )

    print("Pass 2: writing video windows directly to disk (memory-mapped)...")
    for sbj_name in subject_names:
        start_idx, end_idx = subject_window_ranges[sbj_name]
        sbj_windows = all_windows[start_idx:end_idx]
        video_feat = np.load(DATA_DIR / "train" / "videomae_feat" / f"{sbj_name}.npy", allow_pickle=True)
        for offset, w in enumerate(sbj_windows):
            X_video[start_idx + offset] = video_feat[w["v_start"]:w["v_end"]]
        del video_feat
        X_video.flush()
        print(f"  wrote {sbj_name}: rows {start_idx}-{end_idx}")

    meta_df = pd.DataFrame([{"sbj_id": w["sbj_id"], "label_encoded": w["label_encoded"]} for w in all_windows])
    meta_df.insert(0, "id", range(len(meta_df)))
    meta_df.to_csv(PROCESSED_DIR / "meta_video_raw.csv", index=False)

    print(f"Saved X_video_raw {X_video.shape} -> {X_video_path}")
    print(f"Saved meta_video_raw.csv {meta_df.shape}")
    return X_video, meta_df


if __name__ == "__main__":
    build_all(list_train_subjects())
