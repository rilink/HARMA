"""OOF pass for the LightGBM-IMU base model.

Run from repo root: python -u stacking/run_imu_only.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STACKING_DIR = Path(__file__).resolve().parent
REPO_ROOT    = STACKING_DIR.parent
PROCESSED    = REPO_ROOT / "feature_engineering" / "processed_data"

sys.path.insert(0, str(STACKING_DIR))

from base_models import lightgbm_imu_only_v2
from oof_utils import generate_oof_predictions, generate_test_predictions

OUT_DIR = STACKING_DIR / "oof_outputs"
OUT_DIR.mkdir(exist_ok=True)

NAME = "lgbm_imu_only"

print("Loading sensor-level train features ...", flush=True)
train_df = pd.read_csv(PROCESSED / "train_features.csv", low_memory=False)
print(f"  train_df: {train_df.shape}", flush=True)

print("\n" + "=" * 60, flush=True)
print(f"OOF pass — {NAME} (sensor-level)", flush=True)
oof, labels, fold_res = generate_oof_predictions(train_df, lightgbm_imu_only_v2, n_folds=6, seed=42)
np.save(OUT_DIR / f"oof_{NAME}.npy", oof)
pd.DataFrame(fold_res).to_csv(OUT_DIR / f"fold_results_{NAME}.csv", index=False)
print(f"Saved -> oof_{NAME}.npy  shape: {oof.shape}", flush=True)

print("\n" + "=" * 60, flush=True)
print("Generating test predictions (trained on ALL data) ...", flush=True)
test_feat = pd.read_csv(PROCESSED / "test_features.csv", low_memory=False)
print(f"  test_feat: {test_feat.shape}", flush=True)
test_preds = generate_test_predictions(train_df, test_feat, lightgbm_imu_only_v2)
np.save(OUT_DIR / f"test_{NAME}.npy", test_preds)
print(f"Saved -> test_{NAME}.npy  shape: {test_preds.shape}", flush=True)
print("\nDone.", flush=True)
