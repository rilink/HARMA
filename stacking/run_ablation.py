"""Leave-one-out ablation over the 3 base models.

Run from repo root: python stacking/run_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

STACKING_DIR = Path(__file__).resolve().parent
REPO_ROOT    = STACKING_DIR.parent
sys.path.insert(0, str(STACKING_DIR))

from meta_learner import build_meta_features, _make_meta_model
from oof_utils import partition_subjects, LABEL_COL, SBJ_COL

TRAIN_PATH = REPO_ROOT / "feature_engineering" / "processed_data" / "train_features.csv"
OUT_DIR    = STACKING_DIR / "oof_outputs"
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    ("lgbm_imu_only",        "oof_lgbm_imu_only.npy"),
    ("lgbm_imu_temporal_pca","oof_lgbm_imu_temporal_pca.npy"),
    ("lgbm_imu_adversarial", "oof_lgbm_imu_adversarial.npy"),
]

N_FOLDS = 6
SEED    = 42
KIND    = "ridge"   # fast for ablation

print("Loading train metadata...")
train_df = pd.read_csv(TRAIN_PATH, usecols=["id", SBJ_COL, LABEL_COL], low_memory=False)
labels   = train_df[LABEL_COL].values
sbj_ids  = train_df[SBJ_COL].values
print(f"  train rows: {len(train_df)}   unique subjects: {train_df[SBJ_COL].nunique()}")

print("\nLoading OOF arrays...")
oof_dict = {}
for name, fname in MODELS:
    arr = np.load(OUT_DIR / fname)
    oof_dict[name] = arr
    print(f"  {name}: {arr.shape}")

groups = partition_subjects(sorted(set(sbj_ids)), N_FOLDS, seed=SEED)


def evaluate_stack(included_names):
    """Run meta-learner CV on the given subset of models. Returns list of per-fold F1s."""
    arrays = [oof_dict[n] for n in included_names]
    meta_df = build_meta_features(arrays, included_names)
    fold_f1s = []
    y_true_all, y_pred_all = [], []
    for held_out in groups:
        test_mask  = np.isin(sbj_ids, held_out)
        train_mask = ~test_mask
        x_tr, y_tr = meta_df.values[train_mask], labels[train_mask]
        x_te, y_te = meta_df.values[test_mask],  labels[test_mask]
        model = _make_meta_model(KIND)
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_te)
        fold_f1s.append(f1_score(y_te, y_pred, average="macro"))
        y_true_all.append(y_te)
        y_pred_all.append(y_pred)
    pooled_f1 = f1_score(np.concatenate(y_true_all), np.concatenate(y_pred_all), average="macro")
    return np.array(fold_f1s), pooled_f1


all_names = [n for n, _ in MODELS]

print(f"\nFull 3-model stack ({KIND} meta-learner, {N_FOLDS}-fold CV)...")
full_f1s, full_pooled = evaluate_stack(all_names)
full_mean = full_f1s.mean()
full_std  = full_f1s.std()
print(f"  Full stack: {full_mean:.4f} +/- {full_std:.4f}  (pooled: {full_pooled:.4f})")

print("\nLOO ablation...")
rows = [{"model_removed": "none", "loo_mean_f1": full_mean, "loo_std_f1": full_std,
         "pooled_f1": full_pooled, "delta_f1": 0.0}]

for leave_out in all_names:
    subset = [n for n in all_names if n != leave_out]
    f1s, pooled = evaluate_stack(subset)
    delta = full_mean - f1s.mean()
    print(f"  w/o {leave_out:<30}  F1={f1s.mean():.4f} +/- {f1s.std():.4f}  delta=+{delta:.4f}")
    rows.append({"model_removed": leave_out, "loo_mean_f1": f1s.mean(),
                 "loo_std_f1": f1s.std(), "pooled_f1": pooled, "delta_f1": delta})

results_df = pd.DataFrame(rows)
out_path = OUT_DIR / "ablation_table.csv"
results_df.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")
print(results_df.to_string(index=False))
