"""OOF pass for the DANN adversarial video encoder. Must run before run_imu_adversarial.py.

Run from repo root (uses GPU if available, else CPU): python stacking/run_adversarial.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STACKING_DIR = Path(__file__).resolve().parent
REPO_ROOT    = STACKING_DIR.parent
VIDEO_DIR    = REPO_ROOT / "feature_engineering" / "processed_data"

sys.path.insert(0, str(STACKING_DIR))

from adversarial_encoder import make_adversarial_video_fn
from oof_utils import generate_oof_predictions, generate_test_predictions

OUT_DIR  = STACKING_DIR / "oof_outputs"
OUT_DIR.mkdir(exist_ok=True)

print("Loading video windows and metadata ...")
X_video_train = np.load(VIDEO_DIR / "X_video_raw.npy")          # (136849, 15, 768)
meta_df       = pd.read_csv(VIDEO_DIR / "meta_video_raw.csv")   # id, sbj_id, label_encoded, ...
print(f"  X_video_train: {X_video_train.shape}")
print(f"  meta_df: {meta_df.shape}")

print("Loading test video windows ...")
test_video_raw = np.load(REPO_ROOT / "data" / "test" / "test_videomae_data.npy")
# Test array is (12234, 768, 15) — transpose to (12234, 15, 768) to match training shape
if test_video_raw.ndim == 3 and test_video_raw.shape[1] == 768:
    test_video_raw = test_video_raw.transpose(0, 2, 1)
X_video_test = test_video_raw.astype(np.float32)
print(f"  X_video_test: {X_video_test.shape}")

# meta_df must have: id (0..136848), sbj_id, label_encoded
# The adversarial closure indexes X_video_train with train_df["id"].values directly
model_fn = make_adversarial_video_fn(
    X_video_train, X_video_test,
    epochs=25, batch_size=128, lr=1e-3, lambda_max=1.0, seed=42,
)

print("\n" + "=" * 60)
print("OOF pass — subject-adversarial LSTM video encoder")
oof, labels, fold_res = generate_oof_predictions(meta_df, model_fn, n_folds=6, seed=42)
np.save(OUT_DIR / "oof_adversarial_video.npy", oof)
pd.DataFrame(fold_res).to_csv(OUT_DIR / "fold_results_adversarial_video.csv", index=False)
print(f"Saved -> oof_adversarial_video.npy  shape: {oof.shape}")

print("\n" + "=" * 60)
print("Generating test predictions (trained on ALL data) ...")
test_meta = pd.read_csv(REPO_ROOT / "data" / "test" / "test_meta_data.csv")
test_meta["id"] = test_meta.index   # 0..12233, matching X_video_test row order
test_preds = generate_test_predictions(meta_df, test_meta, model_fn)
np.save(OUT_DIR / "test_adversarial_video.npy", test_preds)
print(f"Saved -> test_adversarial_video.npy  shape: {test_preds.shape}")
print("\nDone.")
