"""fit_predict_proba implementations for the 3 base models. Each function is
(train_df, test_df) -> np.ndarray of shape (n_test, 19). Run via run_*.py."""

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_utils import NON_FEATURE_COLS, LABEL_COL, prepare_xy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO_ROOT / "feature_engineering" / "processed_data"
N_CLASSES = 19
SBJ_COL   = "sbj_id"

_X_VIDEO        = None
_PCA_64_GLOBAL  = None
_ADV_OOF_BY_WIN = None
_ADV_TEST       = None


def _get_x_video():
    global _X_VIDEO
    if _X_VIDEO is None:
        _X_VIDEO = np.load(VIDEO_DIR / "X_video_raw.npy", mmap_mode="r")
    return _X_VIDEO


def _get_pca_64_global(n_components=64):
    """Fit PCA once on all training + challenge test windows and cache it."""
    global _PCA_64_GLOBAL
    if _PCA_64_GLOBAL is not None:
        return _PCA_64_GLOBAL

    from sklearn.decomposition import IncrementalPCA

    X_train = _get_x_video()                    # (136849, 15, 768) mmap
    n_train = X_train.shape[0]

    test_video_path = REPO_ROOT / "data" / "test" / "test_videomae_data.npy"
    X_test = np.load(test_video_path, allow_pickle=True).transpose(0, 2, 1).astype(np.float32)
    n_test = X_test.shape[0]

    batch = 2000
    pca = IncrementalPCA(n_components=n_components, batch_size=batch)
    for i in range(0, n_train, batch):
        pca.partial_fit(X_train[i:i+batch].reshape(-1, 15 * 768).astype(np.float32))
    for i in range(0, n_test, batch):
        pca.partial_fit(X_test[i:i+batch].reshape(-1, 15 * 768))

    print(f"Global PCA fitted on {n_train} train + {n_test} test windows; "
          f"EVR sum: {pca.explained_variance_ratio_.sum():.3f}")
    _PCA_64_GLOBAL = pca
    return _PCA_64_GLOBAL


def _get_adversarial_features():
    """Load adversarial OOF/test probability arrays, indexed by window ID."""
    global _ADV_OOF_BY_WIN, _ADV_TEST
    if _ADV_OOF_BY_WIN is None:
        import pandas as pd
        out_dir = Path(__file__).resolve().parent / "oof_outputs"
        oof     = np.load(out_dir / "oof_adversarial_video.npy")       # (136849, 19)
        meta    = pd.read_csv(VIDEO_DIR / "meta_video_raw.csv")
        arr     = np.zeros((int(meta["id"].max()) + 1, 19), dtype=np.float32)
        arr[meta["id"].values] = oof
        _ADV_OOF_BY_WIN = arr
        _ADV_TEST = np.load(out_dir / "test_adversarial_video.npy")    # (12234, 19)
    return _ADV_OOF_BY_WIN, _ADV_TEST


def _prepare_test_x(test_df):
    """Like prepare_xy but safe for unlabelled test data (no label_encoded column)."""
    x = test_df.drop(columns=[c for c in NON_FEATURE_COLS if c in test_df.columns])
    if "sensor_location" in x.columns:
        x["sensor_location"] = x["sensor_location"].astype("category")
    return x


def _make_lightgbm():
    return LGBMClassifier(
        subsample=0.9, boosting_type="gbdt", n_estimators=400, max_depth=15,
        learning_rate=0.1, colsample_bytree=1, n_jobs=-1, min_split_gain=0.05,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, deterministic=True, force_row_wise=True,
    )


def _transform_pca_batched(pca, X_video, ids, batch_size=2000):
    """Transform rows from a video array (mmap or regular) in batches to cap peak RAM."""
    chunks = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch = X_video[batch_ids].reshape(-1, 15 * 768).astype(np.float32)
        chunks.append(pca.transform(batch).astype(np.float32))
    return np.vstack(chunks)


def lightgbm_imu_only_v2(train_df, test_df):
    """LightGBM on IMU tabular features only + 3x median null-class undersampling."""
    x_train, y_train = prepare_xy(train_df)
    video_cols = [c for c in x_train.columns if c.startswith("video_pca_")]
    x_train    = x_train.drop(columns=video_cols)

    has_label = LABEL_COL in test_df.columns
    x_test_df, _ = prepare_xy(test_df) if has_label else (_prepare_test_x(test_df), None)
    x_test = x_test_df.drop(columns=[c for c in video_cols if c in x_test_df.columns])
    x_test = x_test[x_train.columns]

    y_vals       = y_train.values
    nonnull_mask = y_vals != 0
    null_mask    = y_vals == 0
    target_n     = int(np.median(np.bincount(y_vals[nonnull_mask]))) * 3
    rng          = np.random.default_rng(42)
    keep_null    = rng.choice(np.where(null_mask)[0], size=min(target_n, null_mask.sum()), replace=False)
    keep_idx     = np.concatenate([np.where(nonnull_mask)[0], keep_null])
    x_train      = x_train.iloc[keep_idx]
    y_train_us   = y_vals[keep_idx]

    model = _make_lightgbm()
    model.fit(x_train, y_train_us)
    return model.predict_proba(x_test)


def lightgbm_imu_temporal_pca(train_df, test_df, n_components=64):
    """LightGBM on IMU + 64-dim temporal PCA video features. PCA is fitted once on
    all train+test windows combined (see _get_pca_64_global), shared across folds."""
    X_video = _get_x_video()              # (136849, 15, 768) mmap
    pca     = _get_pca_64_global(n_components)

    x_imu_train, y_train = prepare_xy(train_df)
    video_cols = [c for c in x_imu_train.columns if c.startswith("video_pca_")]
    x_imu_train = x_imu_train.drop(columns=video_cols)

    has_label = LABEL_COL in test_df.columns
    x_imu_test_df, _ = prepare_xy(test_df) if has_label else (_prepare_test_x(test_df), None)
    x_imu_test = x_imu_test_df.drop(columns=[c for c in video_cols if c in x_imu_test_df.columns])

    # Deduplicate to unique window IDs: each window appears 4× in sensor-level data
    train_win_ids = train_df["id"].values // 4
    unique_tr_wins, first_tr_idx, inv_tr = np.unique(
        train_win_ids, return_index=True, return_inverse=True
    )
    tr_subjs_unique = train_df[SBJ_COL].values[first_tr_idx]

    x_tr_pca_u = _transform_pca_batched(pca, X_video, unique_tr_wins)

    if has_label:
        test_win_ids = test_df["id"].values // 4
    else:
        test_win_ids = test_df["id"].values   # test_features.csv ids are direct window indices

    unique_te_wins, first_te_idx, inv_te = np.unique(
        test_win_ids, return_index=True, return_inverse=True
    )
    te_subjs_unique = test_df[SBJ_COL].values[first_te_idx]

    if has_label:
        x_te_pca_u = _transform_pca_batched(pca, X_video, unique_te_wins)
    else:
        test_video_path = REPO_ROOT / "data" / "test" / "test_videomae_data.npy"
        x_te_all = np.load(test_video_path, allow_pickle=True).transpose(0, 2, 1).astype(np.float32)
        x_te_pca_u = _transform_pca_batched(pca, x_te_all, unique_te_wins)

    # Transductive per-subject residualization on unique windows
    n_tr_u    = len(x_tr_pca_u)
    all_pca_u = np.concatenate([x_tr_pca_u, x_te_pca_u], axis=0)
    all_subjs_u = np.concatenate([tr_subjs_unique, te_subjs_unique])
    for sbj in np.unique(all_subjs_u):
        mask = all_subjs_u == sbj
        all_pca_u[mask] -= all_pca_u[mask].mean(axis=0)
    x_tr_pca_u = all_pca_u[:n_tr_u]
    x_te_pca_u = all_pca_u[n_tr_u:]

    # Broadcast unique-window PCA features back to sensor level
    x_tr_pca = x_tr_pca_u[inv_tr]
    x_te_pca = x_te_pca_u[inv_te]

    pca_cols = [f"tpca_{i}" for i in range(n_components)]
    x_tr_combined = x_imu_train.copy().reset_index(drop=True)
    x_te_combined = x_imu_test.copy().reset_index(drop=True)
    for i, col in enumerate(pca_cols):
        x_tr_combined[col] = x_tr_pca[:, i]
        x_te_combined[col] = x_te_pca[:, i]

    # Null undersampling
    nonnull_mask = y_train.values != 0
    null_mask    = y_train.values == 0
    target_n     = int(np.median(np.bincount(y_train.values[nonnull_mask])))
    rng          = np.random.default_rng(42)
    keep_null    = rng.choice(np.where(null_mask)[0], size=min(target_n, null_mask.sum()), replace=False)
    keep_idx     = np.concatenate([np.where(nonnull_mask)[0], keep_null])
    x_tr_combined = x_tr_combined.iloc[keep_idx]
    y_train_us    = y_train.values[keep_idx]

    model = _make_lightgbm()
    model.fit(x_tr_combined, y_train_us)
    return model.predict_proba(x_te_combined)


def lightgbm_imu_temporal_pca_perfold(train_df, test_df, n_components=64):
    """Per-fold-PCA variant of lightgbm_imu_temporal_pca, fitted fresh per fold on
    that fold's training windows only. Kept separate for comparison purposes."""
    from sklearn.decomposition import IncrementalPCA

    X_video = _get_x_video()   # (136849, 15, 768) mmap

    x_imu_train, y_train = prepare_xy(train_df)
    video_cols = [c for c in x_imu_train.columns if c.startswith("video_pca_")]
    x_imu_train = x_imu_train.drop(columns=video_cols)

    has_label = LABEL_COL in test_df.columns
    x_imu_test_df, _ = prepare_xy(test_df) if has_label else (_prepare_test_x(test_df), None)
    x_imu_test = x_imu_test_df.drop(columns=[c for c in video_cols if c in x_imu_test_df.columns])

    # Deduplicate to unique window IDs: each window appears 4× in sensor-level data
    train_win_ids = train_df["id"].values // 4
    unique_tr_wins, first_tr_idx, inv_tr = np.unique(
        train_win_ids, return_index=True, return_inverse=True
    )
    tr_subjs_unique = train_df[SBJ_COL].values[first_tr_idx]

    # PCA fitted fresh on this fold's training windows only
    batch_size_load = 2000
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size_load)
    for i in range(0, len(unique_tr_wins), batch_size_load):
        batch = X_video[unique_tr_wins[i:i+batch_size_load]].reshape(-1, 15 * 768).astype(np.float32)
        pca.partial_fit(batch)

    x_tr_pca_u = _transform_pca_batched(pca, X_video, unique_tr_wins)

    if has_label:
        test_win_ids = test_df["id"].values // 4
    else:
        test_win_ids = test_df["id"].values   # test_features.csv ids are direct window indices

    unique_te_wins, first_te_idx, inv_te = np.unique(
        test_win_ids, return_index=True, return_inverse=True
    )
    te_subjs_unique = test_df[SBJ_COL].values[first_te_idx]

    if has_label:
        x_te_pca_u = _transform_pca_batched(pca, X_video, unique_te_wins)
    else:
        test_video_path = REPO_ROOT / "data" / "test" / "test_videomae_data.npy"
        x_te_all = np.load(test_video_path, allow_pickle=True).transpose(0, 2, 1).astype(np.float32)
        x_te_pca_u = _transform_pca_batched(pca, x_te_all, unique_te_wins)

    # Transductive per-subject residualization on unique windows
    n_tr_u    = len(x_tr_pca_u)
    all_pca_u = np.concatenate([x_tr_pca_u, x_te_pca_u], axis=0)
    all_subjs_u = np.concatenate([tr_subjs_unique, te_subjs_unique])
    for sbj in np.unique(all_subjs_u):
        mask = all_subjs_u == sbj
        all_pca_u[mask] -= all_pca_u[mask].mean(axis=0)
    x_tr_pca_u = all_pca_u[:n_tr_u]
    x_te_pca_u = all_pca_u[n_tr_u:]

    # Broadcast unique-window PCA features back to sensor level
    x_tr_pca = x_tr_pca_u[inv_tr]
    x_te_pca = x_te_pca_u[inv_te]

    pca_cols = [f"tpca_{i}" for i in range(n_components)]
    x_tr_combined = x_imu_train.copy().reset_index(drop=True)
    x_te_combined = x_imu_test.copy().reset_index(drop=True)
    for i, col in enumerate(pca_cols):
        x_tr_combined[col] = x_tr_pca[:, i]
        x_te_combined[col] = x_te_pca[:, i]

    # Null undersampling
    nonnull_mask = y_train.values != 0
    null_mask    = y_train.values == 0
    target_n     = int(np.median(np.bincount(y_train.values[nonnull_mask])))
    rng          = np.random.default_rng(42)
    keep_null    = rng.choice(np.where(null_mask)[0], size=min(target_n, null_mask.sum()), replace=False)
    keep_idx     = np.concatenate([np.where(nonnull_mask)[0], keep_null])
    x_tr_combined = x_tr_combined.iloc[keep_idx]
    y_train_us    = y_train.values[keep_idx]

    model = _make_lightgbm()
    model.fit(x_tr_combined, y_train_us)
    return model.predict_proba(x_te_combined)


def lightgbm_imu_adversarial(train_df, test_df):
    """LightGBM on IMU features + the DANN encoder's 19-dim OOF softmax outputs.
    Requires run_adversarial.py to have been run first."""
    import pandas as pd

    adv_oof, adv_test = _get_adversarial_features()

    x_imu_train, y_train = prepare_xy(train_df)
    video_cols   = [c for c in x_imu_train.columns if c.startswith("video_pca_")]
    x_imu_train  = x_imu_train.drop(columns=video_cols)

    has_label     = LABEL_COL in test_df.columns
    x_imu_test_df, _ = prepare_xy(test_df) if has_label else (_prepare_test_x(test_df), None)
    x_imu_test    = x_imu_test_df.drop(columns=[c for c in video_cols if c in x_imu_test_df.columns])

    train_win_ids = train_df["id"].values // 4
    test_win_ids  = test_df["id"].values // 4 if has_label else test_df["id"].values

    adv_tr = adv_oof[train_win_ids]
    adv_te = adv_oof[test_win_ids] if has_label else adv_test[test_win_ids]

    x_tr_combined = x_imu_train.copy().reset_index(drop=True)
    x_te_combined = x_imu_test.copy().reset_index(drop=True)
    for c in range(19):
        x_tr_combined[f"adv_{c}"] = adv_tr[:, c]
        x_te_combined[f"adv_{c}"] = adv_te[:, c]

    model = _make_lightgbm()
    model.fit(x_tr_combined, y_train)
    return model.predict_proba(x_te_combined)
