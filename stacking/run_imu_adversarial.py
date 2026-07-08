"""OOF pass for the Adversarial-IMU base model.

Run from repo root: python stacking/run_imu_adversarial.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STACKING_DIR = Path(__file__).resolve().parent
REPO_ROOT    = STACKING_DIR.parent
PROCESSED    = REPO_ROOT / "feature_engineering" / "processed_data"

sys.path.insert(0, str(STACKING_DIR))

from base_models import lightgbm_imu_adversarial
from oof_utils import generate_oof_predictions, generate_test_predictions

OUT_DIR = STACKING_DIR / "oof_outputs"
OUT_DIR.mkdir(exist_ok=True)

NAME = "lgbm_imu_adversarial"

print("Loading sensor-level train features ...")
train_df = pd.read_csv(PROCESSED / "train_features.csv", low_memory=False)
print(f"  train_df: {train_df.shape}")

print("\n" + "=" * 60)
print(f"OOF pass — {NAME} (sensor-level)")
oof, labels, fold_res = generate_oof_predictions(
    train_df, lightgbm_imu_adversarial, n_folds=6, seed=42
)
np.save(OUT_DIR / f"oof_{NAME}.npy", oof)
pd.DataFrame(fold_res).to_csv(OUT_DIR / f"fold_results_{NAME}.csv", index=False)
print(f"Saved -> oof_{NAME}.npy  shape: {oof.shape}")

print("\n" + "=" * 60)
print("Generating test predictions (trained on ALL data) ...")
test_feat = pd.read_csv(PROCESSED / "test_features.csv", low_memory=False)
print(f"  test_feat: {test_feat.shape}")
test_preds = generate_test_predictions(train_df, test_feat, lightgbm_imu_adversarial)
np.save(OUT_DIR / f"test_{NAME}.npy", test_preds)
print(f"Saved -> test_{NAME}.npy  shape: {test_preds.shape}")
print("\nDone.")
