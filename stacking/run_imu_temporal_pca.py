"""OOF pass for the IMU-TempPCA base model.

Run from repo root: python stacking/run_imu_temporal_pca.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STACKING_DIR = Path(__file__).resolve().parent
REPO_ROOT    = STACKING_DIR.parent
PROCESSED    = REPO_ROOT / "feature_engineering" / "processed_data"

sys.path.insert(0, str(STACKING_DIR))

from base_models import lightgbm_imu_temporal_pca
from oof_utils import generate_oof_predictions, generate_test_predictions

OUT_DIR = STACKING_DIR / "oof_outputs"
OUT_DIR.mkdir(exist_ok=True)

print("Loading sensor-level train features ...")
train_df = pd.read_csv(PROCESSED / "train_features.csv", low_memory=False)
print(f"  train_df: {train_df.shape}")

print("\n" + "=" * 60)
print("OOF pass — IMU + temporal PCA LightGBM (sensor-level)")
oof, labels, fold_res = generate_oof_predictions(train_df, lightgbm_imu_temporal_pca, n_folds=6, seed=42)
np.save(OUT_DIR / "oof_lgbm_imu_temporal_pca.npy", oof)
pd.DataFrame(fold_res).to_csv(OUT_DIR / "fold_results_lgbm_imu_temporal_pca.csv", index=False)
print(f"Saved -> oof_lgbm_imu_temporal_pca.npy  shape: {oof.shape}")

print("\n" + "=" * 60)
print("Generating test predictions (trained on ALL data) ...")
test_feat = pd.read_csv(PROCESSED / "test_features.csv", low_memory=False)
print(f"  test_feat: {test_feat.shape}")
test_preds = generate_test_predictions(train_df, test_feat, lightgbm_imu_temporal_pca)
np.save(OUT_DIR / "test_lgbm_imu_temporal_pca.npy", test_preds)
print(f"Saved -> test_lgbm_imu_temporal_pca.npy  shape: {test_preds.shape}")
print("\nDone.")
