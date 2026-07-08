"""Train and evaluate a meta-learner on stacked OOF probability features."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_utils import undersample_majority  # noqa: E402
from oof_utils import generate_oof_predictions, partition_subjects, N_CLASSES, LABEL_COL, SBJ_COL  # noqa: E402


def build_meta_features(oof_matrices, model_names):
    """Concatenate per-model OOF probability matrices into one meta-feature table."""
    cols, arrays = [], []
    for name, mat in zip(model_names, oof_matrices):
        arrays.append(mat)
        cols += [f"{name}_prob_{c}" for c in range(N_CLASSES)]
    return pd.DataFrame(np.concatenate(arrays, axis=1), columns=cols)


def _make_meta_model(kind="lgbm"):
    """Create the meta-learner: kind is 'ridge', 'logistic', or 'lgbm'."""
    if kind == "ridge":
        from sklearn.linear_model import RidgeClassifier
        return RidgeClassifier(alpha=1.0, random_state=42)
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial",
                                  n_jobs=-1, random_state=42)
    if kind == "lgbm":
        return LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, n_jobs=-1, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, deterministic=True, force_row_wise=True,
        )
    raise ValueError(f"Unknown kind={kind!r}, choose 'ridge', 'logistic', or 'lgbm'")


def evaluate_meta_learner(meta_df, labels, sbj_ids, n_folds=6, seed=42, undersample=False, kind="lgbm"):
    """Evaluate the meta-learner with a second round of subject-grouped CV."""
    groups = partition_subjects(sorted(set(sbj_ids)), n_folds, seed=seed)
    fold_results = []
    y_true_all, y_pred_all = [], []

    for fold_n, held_out in enumerate(groups):
        test_mask = np.isin(sbj_ids, held_out)
        train_mask = ~test_mask

        x_train, y_train = meta_df.values[train_mask], labels[train_mask]
        x_test, y_test = meta_df.values[test_mask], labels[test_mask]

        if undersample:
            tmp = pd.DataFrame(x_train)
            tmp[LABEL_COL] = y_train
            tmp = undersample_majority(tmp)
            y_train = tmp[LABEL_COL].values
            x_train = tmp.drop(columns=[LABEL_COL]).values

        model = _make_meta_model(kind=kind)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        fold_results.append({
            "fold": fold_n, "held_out_subjects": held_out,
            "n_test": test_mask.sum(), "macro_f1": macro_f1,
        })
        print(f"[fold {fold_n}] held_out={held_out}  n={test_mask.sum()}  macro_f1={macro_f1:.4f}")

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    results_df = pd.DataFrame(fold_results)
    print(f"\nMeta-learner CV summary:")
    print(f"  Mean macro F1: {results_df['macro_f1'].mean():.4f} (+/- {results_df['macro_f1'].std():.4f})")
    print(f"  Pooled macro F1: {f1_score(y_true_all, y_pred_all, average='macro'):.4f}")
    return fold_results


def train_final_meta_learner(meta_df, labels, undersample=False):
    """Train the meta-learner on all available OOF meta-features (no holdout).
    Use for producing the final submission."""
    x, y = meta_df.values, labels
    if undersample:
        tmp = pd.DataFrame(x)
        tmp[LABEL_COL] = y
        tmp = undersample_majority(tmp)
        y = tmp[LABEL_COL].values
        x = tmp.drop(columns=[LABEL_COL]).values
    model = _make_meta_model(kind="lgbm")
    model.fit(x, y)
    return model
