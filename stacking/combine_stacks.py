"""CV evaluation of the Ridge / LightGBM meta-learner on the 3-model OOF stack.
Evaluation only; run combine_stacks_alt_meta.py for the final submission.

Run from repo root: python stacking/combine_stacks.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STACKING_DIR = Path(__file__).resolve().parent
REPO_ROOT    = STACKING_DIR.parent

sys.path.insert(0, str(STACKING_DIR))

from meta_learner import build_meta_features, evaluate_meta_learner

OUT_DIR    = STACKING_DIR / "oof_outputs"
OUT_DIR.mkdir(exist_ok=True)
TRAIN_PATH = REPO_ROOT / "feature_engineering" / "processed_data" / "train_features.csv"

ACTIVE_MODELS = [
    ("lgbm_imu_only",         "oof_lgbm_imu_only.npy"),
    ("lgbm_imu_temporal_pca", "oof_lgbm_imu_temporal_pca.npy"),
    ("lgbm_imu_adversarial",  "oof_lgbm_imu_adversarial.npy"),
]

print("Loading DataFrames...")
train_df = pd.read_csv(TRAIN_PATH, usecols=["id", "sbj_id", "label_encoded"], low_memory=False)
print(f"  train_features: {len(train_df)} sensor rows  ({len(train_df)//4} windows)")
print(f"  Active models:  {[m[0] for m in ACTIVE_MODELS]}")

labels  = train_df["label_encoded"].values
sbj_ids = train_df["sbj_id"].values

print("\nLoading OOF matrices...")
oof_arrays, model_names = [], []
for name, oof_file in ACTIVE_MODELS:
    oof = np.load(OUT_DIR / oof_file)
    print(f"  {name}: {oof.shape}")
    oof_arrays.append(oof)
    model_names.append(name)

meta_train = build_meta_features(oof_arrays, model_names)
print(f"\nMeta-feature table: {meta_train.shape}")

print("\n" + "=" * 60)
print("CV evaluation — Ridge meta-learner...")
fold_res = evaluate_meta_learner(meta_train, labels, sbj_ids, n_folds=6, seed=42, kind="ridge")
pd.DataFrame(fold_res).to_csv(OUT_DIR / "fold_results_combined_meta_ridge.csv", index=False)

print("\n" + "=" * 60)
print("CV evaluation — LightGBM meta-learner...")
fold_res_lgbm = evaluate_meta_learner(meta_train, labels, sbj_ids, n_folds=6, seed=42, kind="lgbm")
pd.DataFrame(fold_res_lgbm).to_csv(OUT_DIR / "fold_results_combined_meta_lgbm.csv", index=False)

print("\nDone. Results saved to stacking/oof_outputs/")
