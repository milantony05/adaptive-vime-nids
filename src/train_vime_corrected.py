"""
train_vime_corrected.py
=======================
Vanilla VIME Self-Supervised Baseline — Unsupervised NIDS (UNSW-NB15)
---------------------------------------------------------------------

Architecture
    Shared Encoder : 65 -> Dense(32, ReLU) -> Dense(16, ReLU) [latent]
    Mask Head      : 16 -> Dense(65, Sigmoid)
    Impute Head    : 16 -> Dense(65, Linear)

Training
    - Input: X_train_normal.npy (56 000 benign rows)
    - Corruption: mask m ~ Bernoulli(p=MASK_PROB)
      For masked features, replace with sample from empirical marginal
      from the benign training set.
    - Loss: BCE(mask_pred, original_mask) + ALPHA * MSE(imputed, original)
      (Corrected VIME weighting)
    - Optimiser: Adam
    - Early stopping on val loss

Evaluation
    - Calculate centroid c = mean(e(x)) on clean benign train data.
    - Anomaly score S(x) = ||e(x) - c||^2.
    - Evaluate on full test set.

No test labels used during training.
No attack rows used anywhere in training.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve)

# ─── paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROC_DIR   = os.path.join(PROJECT_DIR, "preprocessing", "clean")
OUT_DIR       = os.path.join(PROJECT_DIR, "models", "vime_baseline_corrected")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── hyperparameters ─────────────────────────────────────────────────────────
INPUT_DIM      = 65
LATENT_DIM     = 16
HIDDEN_DIM     = 32
MASK_PROB      = 0.20
ALPHA          = 2.0
BATCH_SIZE     = 256
MAX_EPOCHS     = 200
LR             = 1e-3
VAL_FRACTION   = 0.10
PATIENCE       = 15

# =============================================================================
# 1. Load data
# =============================================================================
def load_data():
    print("\n[1/7] Loading preprocessed data ...")

    X_train = np.load(os.path.join(PREPROC_DIR, "X_train_normal.npy")).astype(np.float32)
    X_test  = np.load(os.path.join(PREPROC_DIR, "X_test.npy")).astype(np.float32)
    y_test  = np.load(os.path.join(PREPROC_DIR, "y_test_binary.npy"))

    assert X_train.shape[1] == INPUT_DIM
    assert X_test.shape[1]  == INPUT_DIM
    assert len(y_test)      == X_test.shape[0]
    assert X_train.shape[0] == 56_000

    print(f"    X_train_normal : {X_train.shape}")
    print(f"    X_test         : {X_test.shape}")
    print(f"    y_test distribution: benign={int((y_test==0).sum())} attack={int((y_test==1).sum())}")

    return X_train, X_test, y_test

# =============================================================================
# 2. VIME Generator
# =============================================================================
class VIMEGenerator(keras.utils.Sequence):
    def __init__(self, x, mask_prob, batch_size, shuffle=True):
        self.x = x
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = x.shape[0]
        self.d = x.shape[1]
        
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))
        
    def __getitem__(self, index):
        batch_idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = self.x[batch_idx]
        bs = batch_x.shape[0]
        
        # mask m ~ Bernoulli(p)
        mask = np.random.binomial(1, self.mask_prob, size=batch_x.shape).astype(np.float32)
        
        # Empirical marginal samples from the feature distributions of self.x
        m_samples = np.zeros_like(batch_x)
        for i in range(self.d):
            m_samples[:, i] = np.random.choice(self.x[:, i], size=bs, replace=True)
            
        # x_tilde = x * (1 - m) + m_samples * m
        x_tilde = batch_x * (1 - mask) + m_samples * mask
        
        return x_tilde, {"impute_head": batch_x, "mask_head": mask}
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# =============================================================================
# 3. Build Model
# =============================================================================
def build_vime(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
    inp = keras.Input(shape=(input_dim,), name="input")
    enc1 = layers.Dense(hidden_dim, activation="relu", name="enc_1")(inp)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(enc1)
    
    mask_head = layers.Dense(input_dim, activation="sigmoid", name="mask_head")(latent)
    impute_head = layers.Dense(input_dim, activation="linear", name="impute_head")(latent)
    
    model = keras.Model(inputs=inp, outputs=[impute_head, mask_head], name="VIME")
    encoder = keras.Model(inputs=inp, outputs=latent, name="VIME_Encoder")
    return model, encoder

# =============================================================================
# 4. Train
# =============================================================================
def train_vime(model, X_tr, X_val):
    print("\n[4/7] Training VIME (Corrected Weighting) ...")
    
    train_gen = VIMEGenerator(X_tr, MASK_PROB, BATCH_SIZE, shuffle=True)
    val_gen = VIMEGenerator(X_val, MASK_PROB, BATCH_SIZE, shuffle=False)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss={
            "impute_head": "mse",
            "mask_head": "binary_crossentropy"
        },
        loss_weights={
            "impute_head": ALPHA,
            "mask_head": 1.0
        }
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUT_DIR, "best_vime.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    
    return history

# =============================================================================
# 5. Evaluate
# =============================================================================
def evaluate_vime(encoder, X_train_clean, X_test, y_test):
    print("\n[5/7] Computing anomaly scores (latent centroid) ...")
    
    Z_train = encoder.predict(X_train_clean, batch_size=1024, verbose=0)
    centroid = np.mean(Z_train, axis=0)
    
    Z_test = encoder.predict(X_test, batch_size=1024, verbose=0)
    scores = np.sum((Z_test - centroid) ** 2, axis=1).astype(np.float64)
    
    assert not np.isnan(scores).any()
    assert not np.isinf(scores).any()
    assert len(scores) == len(y_test)
    
    benign = scores[y_test == 0]
    attack = scores[y_test == 1]
    
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    
    stats = {
        "benign": {
            "count":  int(len(benign)),
            "mean":   float(benign.mean()),
            "std":    float(benign.std()),
            "min":    float(benign.min()),
            "p25":    float(np.percentile(benign, 25)),
            "median": float(np.median(benign)),
            "p75":    float(np.percentile(benign, 75)),
            "p95":    float(np.percentile(benign, 95)),
            "max":    float(benign.max()),
        },
        "attack": {
            "count":  int(len(attack)),
            "mean":   float(attack.mean()),
            "std":    float(attack.std()),
            "min":    float(attack.min()),
            "p25":    float(np.percentile(attack, 25)),
            "median": float(np.median(attack)),
            "p75":    float(np.percentile(attack, 75)),
            "p95":    float(np.percentile(attack, 95)),
            "max":    float(attack.max()),
        },
        "roc_auc": float(roc_auc),
        "pr_auc":  float(pr_auc),
    }
    
    return scores, stats, centroid

# =============================================================================
# 6. Save artefacts
# =============================================================================
def save_artefacts(history, scores, y_test, stats, centroid):
    print("\n[6/7] Saving artefacts ...")
    
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(os.path.join(OUT_DIR, "training_history.csv"))
    
    np.save(os.path.join(OUT_DIR, "test_scores.npy"), scores)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"), y_test)
    np.save(os.path.join(OUT_DIR, "latent_centroid.npy"), centroid)
    
    final_train_loss = float(hist_df["loss"].iloc[-1])
    final_val_loss   = float(hist_df["val_loss"].iloc[-1])
    best_val_loss    = float(hist_df["val_loss"].min())
    best_epoch       = int(hist_df["val_loss"].idxmin())
    
    metrics = {
        "seed": SEED,
        "mask_prob": MASK_PROB,
        "alpha": ALPHA,
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs_trained": len(hist_df),
        "best_epoch": best_epoch + 1,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "roc_auc": stats["roc_auc"],
        "pr_auc": stats["pr_auc"],
        "benign_score_stats": stats["benign"],
        "attack_score_stats": stats["attack"],
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Plot Loss Curve
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(hist_df) + 1)
    ax.plot(epochs, hist_df["loss"], label="Train Total Loss", linewidth=1.5)
    ax.plot(epochs, hist_df["val_loss"], label="Val Total Loss", linewidth=1.5, linestyle="--")
    
    ax.plot(epochs, hist_df["impute_head_loss"], label="Train Impute MSE", linewidth=1)
    ax.plot(epochs, hist_df["val_impute_head_loss"], label="Val Impute MSE", linewidth=1, linestyle=":")
    
    ax.axvline(best_epoch + 1, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("VIME - Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    
    # Plot Score Distribution
    benign_scores = scores[y_test == 0]
    attack_scores = scores[y_test == 1]
    
    fpr, tpr, _ = roc_curve(y_test, scores)
    prec, rec, _ = precision_recall_curve(y_test, scores)
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    ax0 = fig.add_subplot(gs[0, :])
    clip_p = 99.5
    x_max = np.percentile(scores, clip_p)
    bins = np.linspace(0, x_max, 120)
    ax0.hist(benign_scores, bins=bins, alpha=0.60, color="#2196F3", label=f"Benign (n={len(benign_scores):,})", density=True)
    ax0.hist(attack_scores, bins=bins, alpha=0.60, color="#F44336", label=f"Attack (n={len(attack_scores):,})", density=True)
    ax0.set_xlabel("Latent Distance to Centroid (anomaly score)")
    ax0.set_ylabel("Density")
    ax0.set_title(f"Score Distribution | ROC-AUC={stats['roc_auc']:.4f} PR-AUC={stats['pr_auc']:.4f}")
    ax0.legend()
    ax0.grid(True, alpha=0.25)
    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(fpr, tpr, linewidth=2, color="#673AB7", label=f"AUC = {stats['roc_auc']:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.25)
    
    ax2 = fig.add_subplot(gs[1, 1])
    attack_frac = (y_test == 1).mean()
    ax2.plot(rec, prec, linewidth=2, color="#FF9800", label=f"AP = {stats['pr_auc']:.4f}")
    ax2.axhline(attack_frac, color="k", linestyle="--", linewidth=1)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.25)
    
    plt.suptitle("VIME Baseline (Corrected) - UNSW-NB15", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
# =============================================================================
# 7. Print Report
# =============================================================================
def print_report(model, encoder, history, stats, scores):
    hist_df = pd.DataFrame(history.history)
    print("\n" + "=" * 62)
    print("  EXPERIMENT REPORT — CORRECTED VANILLA VIME BASELINE")
    print("=" * 62)
    
    print("\n--- Model Architecture ---")
    model.summary(print_fn=print)
    print(f"\n  Total trainable parameters: {model.count_params():,}")
    
    print("\n--- Training Summary ---")
    print(f"  Epochs trained        : {len(hist_df)}")
    print(f"  Best epoch            : {int(hist_df['val_loss'].idxmin()) + 1}")
    print(f"  Final val Total Loss  : {hist_df['val_loss'].iloc[-1]:.6f}")
    print(f"  Best val  Total Loss  : {hist_df['val_loss'].min():.6f}")
    print(f"  Best val Impute MSE   : {hist_df.loc[hist_df['val_loss'].idxmin(), 'val_impute_head_loss']:.6f}")
    print(f"  Best val Mask BCE     : {hist_df.loc[hist_df['val_loss'].idxmin(), 'val_mask_head_loss']:.6f}")
    
    print("\n--- Test Score Statistics ---")
    for split, label in [("benign", "Benign (label=0)"), ("attack", "Attack (label=1)")]:
        s = stats[split]
        print(f"\n  {label}  (n={s['count']:,})")
        print(f"    Mean   : {s['mean']:.6f}")
        print(f"    Std    : {s['std']:.6f}")
        print(f"    Min    : {s['min']:.6f}")
        print(f"    P25    : {s['p25']:.6f}")
        print(f"    Median : {s['median']:.6f}")
        print(f"    P75    : {s['p75']:.6f}")
        print(f"    P95    : {s['p95']:.6f}")
        print(f"    Max    : {s['max']:.6f}")
        
    print("\n--- Anomaly Detection Metrics ---")
    print(f"  ROC-AUC              : {stats['roc_auc']:.4f}")
    print(f"  PR-AUC (Avg Prec.)   : {stats['pr_auc']:.4f}")
    print("\n--- Comparison ---")
    print(f"  Denoising Autoencoder ROC-AUC : 0.8843")
    print(f"  Denoising Autoencoder PR-AUC  : 0.9029")
    print(f"  Previous VIME (incorrect) ROC : 0.7305")
    print(f"  Previous VIME (incorrect) PR  : 0.7558")
    print(f"  Corrected VIME ROC-AUC        : {stats['roc_auc']:.4f}")
    print(f"  Corrected VIME PR-AUC         : {stats['pr_auc']:.4f}")
    print("=" * 62 + "\n")

# =============================================================================
# 8. README
# =============================================================================
def write_readme(stats, model):
    readme = f"""# Corrected Vanilla VIME Baseline — UNSW-NB15

## Overview
Vanilla VIME baseline, corrected to use the original VIME loss weighting:
L_total = L_mask + alpha * L_impute, alpha = {ALPHA}.

## Input Data
| File | Shape | Contents |
|---|---|---|
| `preprocessing/clean/X_train_normal.npy` | (56 000, 65) | Benign training features |
| `preprocessing/clean/X_test.npy` | (82 332, 65) | Test features (benign + attack) |
| `preprocessing/clean/y_test_binary.npy` | (82 332,) | Test labels (0=benign, 1=attack) |

## Architecture
- The encoder architecture is an adaptation for UNSW-NB15 (65 -> 32 -> 16).
- The imputation head is linear because the input representation is standardized and can contain negative values.
- Mask head uses sigmoid activation.

## Corruption Mechanism
- Corruption uses empirical marginal replacement.
- Mask probability p = {MASK_PROB}.
- Corrupted values are sampled from the empirical marginal distribution of the benign training set.

## Loss
Total Loss = Mask_BCE + {ALPHA} * Impute_MSE

## Anomaly Scoring
The primary anomaly score is latent distance to the benign centroid: ||e(x) - c||^2

## Evaluation Metrics
| Metric | Value | DAE Baseline | Previous VIME (Incorrect) |
|---|---|---|---|
| ROC-AUC | {stats['roc_auc']:.4f} | 0.8843 | 0.7305 |
| PR-AUC | {stats['pr_auc']:.4f} | 0.9029 | 0.7558 |
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

def main():
    print("=" * 62)
    print("  CORRECTED VANILLA VIME BASELINE — UNSW-NB15")
    print("=" * 62)
    
    X_train, X_test, y_test = load_data()
    X_tr, X_val = train_test_split(X_train, test_size=VAL_FRACTION, random_state=SEED, shuffle=True)
    
    print("\n[2/7] Splitting train/val (90/10) ...")
    print(f"    Train rows : {X_tr.shape[0]}")
    print(f"    Val   rows : {X_val.shape[0]}")
    
    print("\n[3/7] Building model ...")
    model, encoder = build_vime()
    
    history = train_vime(model, X_tr, X_val)
    
    scores, stats, centroid = evaluate_vime(encoder, X_train, X_test, y_test)
    
    save_artefacts(history, scores, y_test, stats, centroid)
    write_readme(stats, model)
    
    # Assertions
    print("\n[7/7] Assertions ...")
    assert X_train.shape[0] == 56000
    assert not np.isnan(scores).any()
    
    hist_df = pd.DataFrame(history.history)
    best_idx = hist_df['val_loss'].idxmin()
    best_val_loss = hist_df.loc[best_idx, 'val_loss']
    best_val_impute = hist_df.loc[best_idx, 'val_impute_head_loss']
    best_val_mask = hist_df.loc[best_idx, 'val_mask_head_loss']
    
    # Verify loss calculation: val_loss should be approx val_mask + 2.0 * val_impute
    calc_loss = best_val_mask + ALPHA * best_val_impute
    assert np.isclose(best_val_loss, calc_loss, atol=1e-4), f"Loss mismatch: val_loss={best_val_loss}, calc={calc_loss}"
    
    print("  [PASS] All assertions passed")
    print("  [PASS] Verified Total Loss = Mask_BCE + 2.0 * Impute_MSE")
    
    print_report(model, encoder, history, stats, scores)

if __name__ == "__main__":
    main()
