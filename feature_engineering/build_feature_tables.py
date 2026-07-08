"""Builds train_features.csv + test_features.csv: IMU stats via imu_features.feature_calculation,
plus a PCA-pooled video summary over the centered 15 frames of each window."""

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from imu_features import feature_calculation

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed_data"
PCA_PATH = PROCESSED_DIR / "video_pca.joblib"
N_PCA_COMPONENTS = 32

TRAIN_WINDOWS_PATH = PROCESSED_DIR / "train_windows.csv"
TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.csv"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.csv"

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

SENSOR_MAPPINGS = {
    "right_arm": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"],
    "left_arm": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"],
    "right_leg": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"],
    "left_leg": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"],
}


def list_train_subjects():
    return sorted(p.stem for p in (DATA_DIR / "train" / "inertial_feat").glob("*.csv"))


def load_subject_inertial(sbj_name):
    df = pd.read_csv(DATA_DIR / "train" / "inertial_feat" / f"{sbj_name}.csv", low_memory=False)
    df["label"] = df["label"].fillna("null")
    df["label_encoded"] = df["label"].map(activities)
    return df


def fit_video_reducer(subject_names, reducer, sample_frames_per_subject=3000, seed=0, save_path=None):
    """Fit any sklearn-style reducer (.fit/.transform) on a random sample of training video
    frames only (no test leakage). Works for PCA, UMAP, or anything with the same interface."""
    rng = np.random.default_rng(seed)
    samples = []
    for sbj_name in subject_names:
        video_feat = np.load(DATA_DIR / "train" / "videomae_feat" / f"{sbj_name}.npy", allow_pickle=True)
        n = min(sample_frames_per_subject, video_feat.shape[0])
        idx = rng.choice(video_feat.shape[0], size=n, replace=False)
        samples.append(video_feat[idx])
    samples = np.concatenate(samples, axis=0)

    reducer.fit(samples)
    if hasattr(reducer, "explained_variance_ratio_"):
        print(f"Fit {type(reducer).__name__} on {samples.shape[0]} frames from {len(subject_names)} subjects; "
              f"explained variance ratio (sum): {reducer.explained_variance_ratio_.sum():.3f}")
    else:
        print(f"Fit {type(reducer).__name__} on {samples.shape[0]} frames from {len(subject_names)} subjects")

    if save_path:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, save_path)
    return reducer


N_VIDEO_FRAMES = 15  # matches the test set's fixed 15-frame window length


def centered_window(v_start, v_end, n_frames, total_len):
    """Extract exactly n_frames centered within [v_start, v_end), matching the challenge
    authors' convention of taking n_frames from the middle of each 1-second IMU window (trimming
    roughly half from each side - 7 from the start, 8 from the end, for a 30-frame span)."""
    span = v_end - v_start
    if span <= n_frames:
        center = (v_start + v_end) // 2
        start = max(0, center - n_frames // 2)
    else:
        start = v_start + (span - n_frames) // 2
    end = min(total_len, start + n_frames)
    start = max(0, end - n_frames)
    return start, end


def pool_video_window(video_window, prefix):
    video_mean = video_window.mean(axis=0)
    video_std = video_window.std(axis=0)
    video_cols = {f"{prefix}_mean_{i}": v for i, v in enumerate(video_mean)}
    video_cols.update({f"{prefix}_std_{i}": v for i, v in enumerate(video_std)})
    return video_cols


def build_windows_for_subject(sbj_name, reducer, prefix):
    """One row per (window, sensor_location), with pooled video features attached to every
    sensor-location row of the same window (video doesn't depend on sensor location)."""
    inertial_df = load_subject_inertial(sbj_name)
    video_feat = np.load(DATA_DIR / "train" / "videomae_feat" / f"{sbj_name}.npy", allow_pickle=True)
    video_reduced = reducer.transform(video_feat)  # (T_video, n_components)
    ratio = len(inertial_df) / video_feat.shape[0]

    rows = []
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
        v_start, v_end = centered_window(v_start, v_end, N_VIDEO_FRAMES, video_reduced.shape[0])
        video_window = video_reduced[v_start:v_end]
        video_cols = pool_video_window(video_window, prefix)

        for sensor_location, (xc, yc, zc) in SENSOR_MAPPINGS.items():
            rows.append({
                "sbj_id": sbj_id,
                "sensor_location": sensor_location,
                "x_axis": str(chunk[xc].tolist()),
                "y_axis": str(chunk[yc].tolist()),
                "z_axis": str(chunk[zc].tolist()),
                "label": most_common_label,
                "label_encoded": label_encoded,
                **video_cols,
            })
    return rows


def build_train_table(subject_names, reducer, prefix, windows_path, features_path):
    all_rows = []
    for sbj_name in subject_names:
        all_rows.extend(build_windows_for_subject(sbj_name, reducer, prefix))
        print(f"  windowed {sbj_name}: {len(all_rows)} rows so far")

    windows_df = pd.DataFrame(all_rows)
    windows_df.insert(0, "id", range(len(windows_df)))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(windows_path, index=False)
    print(f"Saved {len(windows_df)} windows to {windows_path}")
    return windows_df


def compute_train_features(windows_df, features_path):
    features_df = feature_calculation(windows_df, output_path=str(features_path))
    print(f"Saved {features_df.shape} train features to {features_path}")
    return features_df


def build_test_table(reducer, prefix, features_path):
    """Test video features come pre-windowed: (12234, 768, 15) = (n_samples, embed_dim, n_frames)."""
    test_video = np.load(DATA_DIR / "test" / "test_videomae_data.npy", allow_pickle=True)
    n_samples, embed_dim, n_frames = test_video.shape

    all_frames = test_video.transpose(0, 2, 1).reshape(-1, embed_dim)  # (n_samples * n_frames, 768)
    all_frames_reduced_flat = reducer.transform(all_frames)
    n_components = all_frames_reduced_flat.shape[1]
    all_frames_reduced = all_frames_reduced_flat.reshape(n_samples, n_frames, n_components)

    pooled_mean = all_frames_reduced.mean(axis=1)
    pooled_std = all_frames_reduced.std(axis=1)

    video_df = pd.DataFrame(pooled_mean, columns=[f"{prefix}_mean_{i}" for i in range(n_components)])
    video_df[[f"{prefix}_std_{i}" for i in range(n_components)]] = pooled_std

    # Build test IMU features from raw test data.
    # test_inertial_data.npy expected shape: (n_samples, 50, 3) — 50 timesteps × 3 axes per window
    test_raw  = np.load(DATA_DIR / "test" / "test_inertial_data.npy")
    test_meta = pd.read_csv(DATA_DIR / "test" / "test_meta_data.csv")
    assert len(test_raw) == n_samples, \
        f"test_inertial_data has {len(test_raw)} rows but video has {n_samples}"
    test_windows = pd.DataFrame({
        "id": range(n_samples),
        "sbj_id": test_meta["sbj_id"].values,
        "sensor_location": test_meta["sensor_location"].values,
        "x_axis": [str(test_raw[i, :, 0].tolist()) for i in range(n_samples)],
        "y_axis": [str(test_raw[i, :, 1].tolist()) for i in range(n_samples)],
        "z_axis": [str(test_raw[i, :, 2].tolist()) for i in range(n_samples)],
    })
    imu_test_features = feature_calculation(
        test_windows, output_path=str(PROCESSED_DIR / "test_imu_features.csv")
    )

    combined = pd.concat([imu_test_features.reset_index(drop=True), video_df], axis=1)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(features_path, index=False)
    print(f"Saved {combined.shape} test features to {features_path}")
    return combined


if __name__ == "__main__":
    subjects = list_train_subjects()
    print(f"{len(subjects)} train subject files found")

    pca = fit_video_reducer(subjects, PCA(n_components=N_PCA_COMPONENTS, random_state=0), save_path=PCA_PATH)
    windows_df = build_train_table(subjects, pca, "video_pca", TRAIN_WINDOWS_PATH, TRAIN_FEATURES_PATH)
    compute_train_features(windows_df, TRAIN_FEATURES_PATH)
    build_test_table(pca, "video_pca", TEST_FEATURES_PATH)
