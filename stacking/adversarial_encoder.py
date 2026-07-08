"""Subject-adversarial LSTM video encoder. A gradient reversal layer trains the
encoder to be useful for activity classification but useless for subject ID.

Architecture:
    Input (B, 15, 768)
       -> LSTM(768, 128) -> last hidden (B, 128)
       -> proj: Linear(128, 64) + ReLU + Dropout
       -> latent (B, 64)
         +- activity head: Linear(64, 19)
         `- GRL(lambda) -> subject head: Linear(64, n_subjects)

Usage (from run_adversarial.py): make_adversarial_video_fn(X_video_raw, X_test_raw)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
N_CLASSES  = 19
LABEL_COL  = "label_encoded"
SBJ_COL    = "sbj_id"


# -- Gradient reversal ---------------------------------------------------------

class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


def grad_reverse(x, lambda_=1.0):
    return _GradRev.apply(x, lambda_)


# -- Model ---------------------------------------------------------------------

class SubjectAdversarialEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, latent_dim=64,
                 n_activities=19, n_subjects=22, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj    = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.act_head = nn.Linear(latent_dim, n_activities)
        self.sbj_head = nn.Linear(latent_dim, n_subjects)

    def encode(self, x):
        _, (h, _) = self.lstm(x)   # h: (1, B, hidden_dim)
        return self.proj(h.squeeze(0))

    def forward(self, x, lambda_=1.0):
        z = self.encode(x)
        act_logits = self.act_head(z)
        sbj_logits = self.sbj_head(grad_reverse(z, lambda_))
        return act_logits, sbj_logits


# -- OOF base model function ---------------------------------------------------

def make_adversarial_video_fn(X_video_train, X_video_test,
                               epochs=20, batch_size=128, lr=1e-3,
                               lambda_max=1.0, seed=42):
    """Return a fit_predict_proba_fn(train_df, test_df) -> (n_test, 19) closure.

    X_video_train: (136849, 15, 768) float32, row-aligned with meta_video_raw.csv
    X_video_test:  (12234, 15, 768) float32, from test_videomae_data.npy
    """

    def adversarial_video(train_df, test_df):
        has_label = LABEL_COL in test_df.columns

        x_tr  = X_video_train[train_df["id"].values]           # (n_tr, 15, 768)
        y_act = train_df[LABEL_COL].values.astype(np.int64)

        # Encode subject ids as 0-indexed integers for this fold
        le = LabelEncoder().fit(train_df[SBJ_COL].values)
        y_sbj = le.transform(train_df[SBJ_COL].values).astype(np.int64)
        n_subjects = len(le.classes_)

        x_te = (X_video_train[test_df["id"].values]
                if has_label else X_video_test[test_df["id"].values])

        # Normalise using training statistics
        mean = x_tr.mean(axis=(0, 1), keepdims=True)
        std  = x_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        x_tr = (x_tr - mean) / std
        x_te = (x_te - mean) / std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model     = SubjectAdversarialEncoder(n_subjects=n_subjects).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        act_loss_fn = nn.CrossEntropyLoss()
        sbj_loss_fn = nn.CrossEntropyLoss()

        loader = DataLoader(
            TensorDataset(
                torch.tensor(x_tr),
                torch.tensor(y_act),
                torch.tensor(y_sbj),
            ),
            batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        for epoch in range(epochs):
            # lambda warm-up: ramps from 0 -> lambda_max over training (DANN schedule)
            p       = epoch / max(epochs - 1, 1)
            lambda_ = float(lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0))

            model.train()
            total_act, total_sbj = 0.0, 0.0
            for x_b, y_act_b, y_sbj_b in loader:
                x_b      = x_b.to(device)
                y_act_b  = y_act_b.to(device)
                y_sbj_b  = y_sbj_b.to(device)
                optimizer.zero_grad()
                act_logits, sbj_logits = model(x_b, lambda_=lambda_)
                loss_act = act_loss_fn(act_logits, y_act_b)
                loss_sbj = sbj_loss_fn(sbj_logits, y_sbj_b)
                # Total loss: activity + subject; GRL handles sign reversal for encoder
                loss = loss_act + loss_sbj
                loss.backward()
                optimizer.step()
                total_act += loss_act.item()
                total_sbj += loss_sbj.item()
            scheduler.step()
            print(f"  epoch {epoch+1}/{epochs}  lambda={lambda_:.3f}"
                  f"  act_loss={total_act/len(loader):.4f}"
                  f"  sbj_loss={total_sbj/len(loader):.4f}")

        # Predict activity probabilities for test fold
        model.eval()
        te_tensor   = torch.tensor(x_te)
        test_loader = DataLoader(TensorDataset(te_tensor), batch_size=batch_size * 2, shuffle=False)
        probas = []
        with torch.no_grad():
            for (x_b,) in test_loader:
                act_logits, _ = model(x_b.to(device), lambda_=0.0)
                probas.append(torch.softmax(act_logits, dim=1).cpu().numpy())
        return np.concatenate(probas, axis=0).astype(np.float32)

    return adversarial_video
