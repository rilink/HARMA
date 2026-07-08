"""Feature prep and undersampling utilities shared across base models."""

import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample

LABEL_COL = "label_encoded"
NON_FEATURE_COLS = ["id", "label", "label_encoded", "sbj_id"]

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


def undersample_majority(df, label_col=LABEL_COL, adjustment=15000):
    """Downsample the 'null' majority class toward the average minority-class count + adjustment."""
    majority = df[df[label_col] == 0]
    minority = df[df[label_col] != 0]
    target_count = min(int(minority[label_col].value_counts().mean()) + adjustment, len(majority))

    majority_downsampled = resample(majority, replace=False, n_samples=target_count, random_state=42)
    df_balanced = pd.concat([majority_downsampled, minority])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def split_feature_columns(df):
    """Identify which columns are video-derived (video_pca_*) vs. IMU-derived (everything else
    that isn't metadata), so late-fusion approaches can train separate per-modality models."""
    video_cols = [c for c in df.columns if c.startswith("video_pca_")]
    imu_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + video_cols and c != "sensor_location"]
    return imu_cols, video_cols


def prepare_xy(df, feature_cols=None):
    """feature_cols=None means 'everything except the standard non-feature columns'."""
    if feature_cols is None:
        x = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    else:
        x = df[feature_cols].copy()
    if "sensor_location" in x.columns:
        x["sensor_location"] = x["sensor_location"].astype("category")
    y = df[LABEL_COL]
    return x, y


def group_holdout_cross_validate(
    df, fit_predict_fn, n_holdout_subjects=4, seeds=(0, 1, 2, 3, 4, 5), undersample_adjustment=15000,
):
    """Repeated group-holdout CV: each fold randomly holds out n_holdout_subjects subjects."""
    subjects = sorted(df["sbj_id"].unique())
    fold_results = []
    y_true_all, y_pred_all = [], []

    for seed in seeds:
        held_out = random.Random(seed).sample(subjects, n_holdout_subjects)

        train_df = undersample_majority(df[~df["sbj_id"].isin(held_out)], adjustment=undersample_adjustment)
        test_df = df[df["sbj_id"].isin(held_out)]
        y_test = test_df[LABEL_COL]

        y_pred = fit_predict_fn(train_df, test_df)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        fold_results.append({
            "seed": seed, "held_out_subjects": held_out, "n_test": len(test_df),
            "accuracy": accuracy, "macro_f1": macro_f1,
        })
        print(f"[fold seed={seed}] held_out={held_out} n={len(test_df)} acc={accuracy:.4f} macro_f1={macro_f1:.4f}")

        y_true_all.append(y_test.values)
        y_pred_all.append(np.asarray(y_pred))

    results_df = pd.DataFrame(fold_results)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    print("\n=== Cross-validation summary ===")
    print(f"Mean accuracy: {results_df['accuracy'].mean():.4f} (+/- {results_df['accuracy'].std():.4f})")
    print(f"Mean macro F1: {results_df['macro_f1'].mean():.4f} (+/- {results_df['macro_f1'].std():.4f})")
    print(f"Pooled macro F1 (all folds combined): {f1_score(y_true_all, y_pred_all, average='macro'):.4f}")

    return results_df, y_true_all, y_pred_all


def plot_confusion_matrix(y_true, y_pred, title="Pooled recall matrix"):
    labels = list(activities.values())
    label_names = list(activities.keys())
    C = confusion_matrix(y_true, y_pred, labels=labels)
    recall = (C.T / C.sum(axis=1)).T

    plt.figure(figsize=(20, 7))
    sns.heatmap(recall, annot=True, cmap="Greens", fmt=".2f", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title(title)
    plt.tight_layout()
    plt.show()
