"""Trains LGBM, GPBoost, and TabM meta-learners on the 3-model OOF stack, then
takes the most-confident prediction per test sample across the three.

Run from repo root: python stacking/combine_stacks_alt_meta.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

STACKING_DIR = Path(__file__).resolve().parent
REPO_ROOT    = STACKING_DIR.parent
PROCESSED    = REPO_ROOT / "feature_engineering" / "processed_data"
OUT_DIR      = STACKING_DIR / "oof_outputs"
OUT_DIR.mkdir(exist_ok=True)
SUB_DIR      = REPO_ROOT / "submission_files"
SUB_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(STACKING_DIR))
from meta_learner import build_meta_features

# -- 3 base model OOF stack ----------------------------------------------------
MODELS = [
    ("lgbm_imu_temporal_pca", "oof_lgbm_imu_temporal_pca.npy",  "test_lgbm_imu_temporal_pca.npy"),
    ("lgbm_imu_only",         "oof_lgbm_imu_only.npy",          "test_lgbm_imu_only.npy"),
    ("lgbm_imu_adversarial",  "oof_lgbm_imu_adversarial.npy",   "test_lgbm_imu_adversarial.npy"),
]
N_CLASSES = 19
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# -- Load data -----------------------------------------------------------------
print("Loading train/test metadata...")
train_df  = pd.read_csv(PROCESSED / "train_features.csv",
                        usecols=["id", "sbj_id", "label_encoded"], low_memory=False)
test_feat = pd.read_csv(PROCESSED / "test_features.csv",
                        usecols=["id", "sbj_id"], low_memory=False)
labels   = train_df["label_encoded"].values
test_ids = test_feat["id"].values
print(f"  train rows: {len(train_df)}   test rows: {len(test_feat)}")

print("Loading OOF / test matrices...")
oof_arrays, test_arrays, names = [], [], []
for name, oof_file, test_file in MODELS:
    oof  = np.load(OUT_DIR / oof_file)
    test = np.load(OUT_DIR / test_file)
    print(f"  {name}: OOF {oof.shape}  test {test.shape}")
    oof_arrays.append(oof)
    test_arrays.append(test)
    names.append(name)

meta_train = build_meta_features(oof_arrays, names)
meta_test  = build_meta_features(test_arrays, names)
X_tr  = meta_train.values.astype(np.float32)
X_te  = meta_test.values.astype(np.float32)
print(f"  meta-feature shape: train={X_tr.shape}  test={X_te.shape}")


# -- Helper: save intermediate probabilities -----------------------------------
def save_intermediate(pred, proba, suffix):
    sub = pd.DataFrame({"id": test_ids, "target_feature": pred})
    proba_df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in range(N_CLASSES)])
    proba_df.insert(0, "id", test_ids)
    csv  = OUT_DIR / f"meta_{suffix}.csv"
    pcsv = OUT_DIR / f"meta_{suffix}_probas.csv"
    sub.to_csv(csv, index=False)
    proba_df.to_csv(pcsv, index=False)
    print(f"  Saved {csv.name}")


# ===============================================================================
# 1.  LGBM meta-learner
# ===============================================================================
print("\n" + "=" * 60)
print("Training LGBM meta-learner...")
from lightgbm import LGBMClassifier

lgbm_meta = LGBMClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    num_leaves=31, n_jobs=-1, reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
    random_state=42, deterministic=True, force_row_wise=True,
)
lgbm_meta.fit(X_tr, labels)
lgbm_proba = lgbm_meta.predict_proba(X_te).astype(np.float32)
lgbm_pred  = lgbm_proba.argmax(axis=1)
save_intermediate(lgbm_pred, lgbm_proba, "lgbm")


# ===============================================================================
# 2.  GPBoost meta-learner
# ===============================================================================
print("\n" + "=" * 60)
print("Training GPBoost meta-learner...")
import gpboost as gpb

gpb_clf = gpb.GPBoostClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.05,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
    deterministic=True,
    force_row_wise=True,
)
gpb_clf.fit(X_tr, labels)
gpb_proba = gpb_clf.predict_proba(X_te).astype(np.float32)
gpb_pred  = gpb_proba.argmax(axis=1)
save_intermediate(gpb_pred, gpb_proba, "gpboost")


# ===============================================================================
# 3.  TabM meta-learner
# ===============================================================================
print("\n" + "=" * 60)
print("Training TabM meta-learner...")
import tabm

N_FEAT    = X_tr.shape[1]
K         = 8
N_EPOCHS  = 50
BATCH     = 8192
LR        = 1e-3
SEED      = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

tabm_model = tabm.TabM(
    n_num_features=N_FEAT,
    cat_cardinalities=None,
    d_out=N_CLASSES,
    n_blocks=1,
    d_block=128,
    dropout=0.05,
    k=K,
    arch_type="tabm-mini",
    start_scaling_init="random-signs",
).to(DEVICE)

optimizer = torch.optim.AdamW(tabm_model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t  = torch.tensor(labels, dtype=torch.long)
ds      = TensorDataset(X_tr_t, y_tr_t)
loader  = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                     generator=torch.Generator().manual_seed(SEED))

tabm_model.train()
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        logits = tabm_model(bx)          # (batch, k, n_classes)
        loss = torch.stack(
            [F.cross_entropy(logits[:, j, :], by) for j in range(K)]
        ).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}/{N_EPOCHS}  loss={epoch_loss/len(loader):.4f}", flush=True)

tabm_model.eval()
with torch.no_grad():
    X_te_t    = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    chunks    = []
    for i in range(0, len(X_te_t), BATCH):
        logits = tabm_model(X_te_t[i:i+BATCH])   # (b, k, 19)
        proba_chunk = F.softmax(logits, dim=-1).mean(dim=1)  # (b, 19)
        chunks.append(proba_chunk.cpu().numpy())
    tabm_proba = np.vstack(chunks).astype(np.float32)

tabm_pred = tabm_proba.argmax(axis=1)
save_intermediate(tabm_pred, tabm_proba, "tabm")


# ===============================================================================
# 4.  Confidence fusion — most confident model wins per sample
# ===============================================================================
print("\n" + "=" * 60)
print("Building max-confidence fusion...")

all_probas  = np.stack([lgbm_proba, gpb_proba, tabm_proba], axis=0)  # (3, N, 19)
model_tags  = ["LGBM", "GPBoost", "TabM"]
max_conf    = all_probas.max(axis=2)          # (3, N)
best_idx    = max_conf.argmax(axis=0)         # (N,)
final_proba = all_probas[best_idx, np.arange(len(best_idx))]  # (N, 19)
final_pred  = final_proba.argmax(axis=1)

for i, tag in enumerate(model_tags):
    n = (best_idx == i).sum()
    print(f"  {tag} most confident: {n:6d} / {len(best_idx)} samples ({100*n/len(best_idx):.1f}%)")

save_intermediate(final_pred, final_proba, "maxconf")

# Write final challenge submission to submission_files/
final_sub = pd.DataFrame({"id": test_ids, "target_feature": final_pred})
final_sub.to_csv(SUB_DIR / "final_submission.csv", index=False)
print(f"\nSaved submission_files/final_submission.csv  ({len(final_sub)} rows)")
print("\nDone. Max-confidence fusion -> submission_files/final_submission.csv")
