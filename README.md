# HARMA

Supplementary code for our submission to the WEAR Dataset Challenge at the HASCA Workshop (UbiComp/ISWC 2026).

Our approach builds three complementary base models and combines them via out-of-fold (OOF) stacking:

| Base model | OOF Macro F1 | Data |
|---|---|---|
| LightGBM-IMU | 0.591 | IMU tabular features |
| IMU-TempPCA | 0.663 | IMU + 64-dim temporal PCA of VideoMAE embeddings |
| Adversarial-IMU | 0.683 | IMU + 19-class softmax of DANN adversarial encoder |

A Ridge meta-learner on the stacked OOF probabilities reaches **F1 = 0.725** (OOF) / **0.732** (leaderboard).
A max-confidence fusion across three meta-learner variants (LightGBM, GPBoost, TabM) yields the final **F1 = 0.737**.

## Repository Structure

```
HARMA/
├── feature_engineering/    Turns raw challenge data into the feature tables the models train on
│   ├── HARMA-supp.pdf      Feature documentation (describes all computed features)
│   ├── imu_features.py     ~83 time/frequency-domain features per window (feature_calculation());
│   │                       imported by build_feature_tables.py, not run standalone
│   ├── build_video_sequences.py  Extracts raw 15-frame VideoMAE windows aligned to IMU
│   │                       windows; writes processed_data/X_video_raw.npy + meta_video_raw.csv
│   ├── build_feature_tables.py   Builds train_features.csv + test_features.csv: its own IMU
│   │                       windowing + imu_features.feature_calculation, plus a 32-dim video-PCA
│   │                       summary (video_pca_* columns - unused by the final 3 models, legacy)
│   └── processed_data/     Generated feature tables
├── stacking/               OOF stacking framework
│   ├── base_models.py      The 3 fit_predict_proba implementations
│   ├── oof_utils.py        6-fold subject-disjoint fold generation
│   ├── cv_utils.py         Feature prep and undersampling utilities
│   ├── meta_learner.py     Ridge / LightGBM meta-learner
│   ├── adversarial_encoder.py  DANN adversarial encoder (LSTM + GRL)
│   ├── tabm.py             TabM neural network (self-contained copy)
│   ├── run_adversarial.py  Pre-computes adversarial video embeddings (GPU optional; runs on CPU too)
│   ├── run_imu_only.py     LightGBM-IMU: runs OOF CV, writes OOF arrays
│   ├── run_imu_temporal_pca.py  IMU-TempPCA: per-subject centering + temporal PCA
│   ├── run_imu_adversarial.py   Adversarial-IMU: DANN encoder + LightGBM
│   ├── combine_stacks.py   CV evaluation of the 3-model Ridge/LightGBM stack (not required
│   │                       for the final submission, but validates the reported Ridge F1)
│   ├── combine_stacks_alt_meta.py  GPBoost + TabM meta-learners + max-confidence fusion
│   └── run_ablation.py     LOO ablation study on the 3-model stack
├── figures/                Paper figures (PDF)
├── submission_files/       All generated submission CSVs, incl. final_submission.csv                 
└── requirements.txt
```

## Data

Place the challenge data under `data/`:
- `data/train/inertial_feat/sbj_*.csv` — per-timestep IMU data
- `data/train/videomae_feat/sbj_*.npy` — per-subject VideoMAE embeddings
- `data/test/test_inertial_data.npy`, `test_videomae_data.npy`, `test_meta_data.csv`

## Setup

```bash
conda create -n wear26 python=3.10
conda activate wear26
pip install -r requirements.txt
```

## Reproducing the Results

1. **Build feature tables:**
   ```bash
   python feature_engineering/build_video_sequences.py   # creates feature_engineering/processed_data/X_video_raw.npy
   python feature_engineering/build_feature_tables.py    # creates feature_engineering/processed_data/train_features.csv + test_features.csv
   ```

2. **Run OOF base models** (writes arrays to `stacking/oof_outputs/`):
   ```bash
   python stacking/run_adversarial.py   # pre-computes adversarial video embeddings; uses GPU if available, else CPU
   python stacking/run_imu_only.py
   python stacking/run_imu_temporal_pca.py
   python stacking/run_imu_adversarial.py
   ```

3. **Stack and produce submission:**
   ```bash
   python stacking/combine_stacks_alt_meta.py   # max-confidence fusion → submission_files/final_submission.csv
   ```

4. **Ablation study:**
   ```bash
   python stacking/run_ablation.py
   ```
