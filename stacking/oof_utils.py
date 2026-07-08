"""Out-of-fold (OOF) prediction utility for stacked generalisation. Each base model
is run once per subject-disjoint fold so every training row is scored by a model
that never saw its subject; results stack into a meta-feature table for meta_learner.py."""

import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

N_CLASSES = 19
LABEL_COL = "label_encoded"
SBJ_COL = "sbj_id"


def partition_subjects(subjects, n_folds, seed=42):
    """Split a list of subjects into n_folds roughly equal groups.
    With 22 subjects and n_folds=6: four groups of 4 and two groups of 3."""
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    return [list(g) for g in np.array_split(shuffled, n_folds)]


def generate_oof_predictions(df, fit_predict_proba_fn, n_folds=6, seed=42, undersample_fn=None):
    """Generate OOF probability predictions for every row in df, one subject-disjoint
    fold at a time. Returns (oof_probas, oof_labels, fold_results)."""
    subjects = sorted(df[SBJ_COL].unique())
    groups = partition_subjects(subjects, n_folds, seed=seed)

    oof_probas = np.zeros((len(df), N_CLASSES), dtype=np.float32)
    oof_labels = df[LABEL_COL].values.copy()
    fold_results = []

    for fold_n, held_out in enumerate(groups):
        train_mask = ~df[SBJ_COL].isin(held_out)
        test_mask = df[SBJ_COL].isin(held_out)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if undersample_fn is not None:
            train_df = undersample_fn(train_df)

        probas = fit_predict_proba_fn(train_df, test_df)  # (n_test, n_classes)

        oof_probas[test_mask] = probas
        macro_f1 = f1_score(test_df[LABEL_COL].values, probas.argmax(axis=1), average="macro")

        fold_results.append({
            "fold": fold_n,
            "held_out_subjects": held_out,
            "n_test": test_mask.sum(),
            "macro_f1": macro_f1,
        })
        print(f"[fold {fold_n}] held_out={held_out}  n={test_mask.sum()}  macro_f1={macro_f1:.4f}")

    overall_f1 = f1_score(oof_labels, oof_probas.argmax(axis=1), average="macro")
    print(f"\nOOF macro F1 (all folds pooled): {overall_f1:.4f}")
    results_df = pd.DataFrame(fold_results)
    print(f"Mean macro F1: {results_df['macro_f1'].mean():.4f} (+/- {results_df['macro_f1'].std():.4f})")

    return oof_probas, oof_labels, fold_results


def generate_test_predictions(train_df, test_df, fit_predict_proba_fn, undersample_fn=None):
    """Train on ALL training data and predict test set probabilities.
    Used at inference time to build meta-features for the real test set."""
    if undersample_fn is not None:
        train_df = undersample_fn(train_df)
    return fit_predict_proba_fn(train_df, test_df)
