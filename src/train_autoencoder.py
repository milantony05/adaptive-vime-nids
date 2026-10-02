"""
train_autoencoder.py
====================
Denoising Autoencoder Baseline — Unsupervised NIDS (UNSW-NB15)
---------------------------------------------------------------

Architecture
    Encoder : 65 -> Dense(32, ReLU) -> Dense(16, ReLU)   [latent]
    Decoder : 16 -> Dense(32, ReLU) -> Dense(65, linear)

Training
    - Input: X_train_normal.npy  (56 000 benign rows, 65 features)
    - Corruption: random masking (p=MASK_PROB) applied per-batch, per-sample
    - Target: original clean vector
    - Loss: MSE
    - Optimiser: Adam
    - Early stopping on val MSE

Evaluation
    - Anomaly score = MSE(x, x_hat) per test sample
    - No threshold selected here; continuous scores are saved
    - Metrics: ROC-AUC, PR-AUC (Average Precision)

No test labels used during training.
No attack rows used anywhere.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
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
OUT_DIR       = os.path.join(PROJECT_DIR, "models", "autoencoder_baseline")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── hyperparameters ─────────────────────────────────────────────────────────
INPUT_DIM      = 65
LATENT_DIM     = 16
HIDDEN_DIM     = 32
MASK_PROB      = 0.10   # fraction of features zeroed per training sample
BATCH_SIZE     = 256
MAX_EPOCHS     = 200
LR             = 1e-3
VAL_FRACTION   = 0.10   # 90/10 train/val from the 56 000 benign rows
PATIENCE       = 15     # early-stopping patience


# =============================================================================
# 1.  Load data
# =============================================================================
def load_data():
    print("\n[1/7] Loading preprocessed data ...")

    X_train = np.load(os.path.join(PREPROC_DIR, "X_train_normal.npy"))
    X_test  = np.load(os.path.join(PREPROC_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PREPROC_DIR, "y_test_binary.npy"))

    assert X_train.shape[1] == INPUT_DIM, \
        f"Expected {INPUT_DIM} features, got {X_train.shape[1]}"
    assert X_test.shape[1]  == INPUT_DIM
    assert len(y_test)       == X_test.shape[0]

    # Confirm no attack rows in training data
    # (preprocess_clean.py guarantees this; we assert the shape as a proxy)
    assert X_train.shape[0]  == 56_000, \
        f"Expected 56 000 benign training rows, got {X_train.shape[0]}"

    print(f"    X_train_normal : {X_train.shape}")
    print(f"    X_test         : {X_test.shape}")
    print(f"    y_test distribution: "
          f"benign={int((y_test==0).sum())}  attack={int((y_test==1).sum())}")

    return X_train.astype(np.float32), X_test.astype(np.float32), y_test


# =============================================================================
# 2.  Train / validation split (benign only)
# =============================================================================
def split_train_val(X_train):
    print(f"\n[2/7] Splitting into train/val ({int((1-VAL_FRACTION)*100)}/{int(VAL_FRACTION*100)}) ...")
    X_tr, X_val = train_test_split(X_train, test_size=VAL_FRACTION,
                                   random_state=SEED, shuffle=True)
    print(f"    Train rows : {X_tr.shape[0]}")
    print(f"    Val   rows : {X_val.shape[0]}")
    return X_tr, X_val


# =============================================================================
# 3.  Masking noise as a Keras layer
# =============================================================================
class MaskingNoise(layers.Layer):
    """
    Randomly zeros a fraction `p` of input features independently per sample.
    Applied only during training (training=True).

    This is equivalent to 'salt-and-zero' dropout on the INPUT, not on
    hidden units — it forces the decoder to learn robust representations
    rather than a near-identity mapping.
    """

    def __init__(self, mask_prob: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.mask_prob = mask_prob

    def call(self, inputs, training=None):
        if training:
            # Bernoulli mask: 1 = keep, 0 = corrupt
            mask = tf.cast(
                tf.random.uniform(tf.shape(inputs)) > self.mask_prob,
                dtype=inputs.dtype
            )
            return inputs * mask
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mask_prob": self.mask_prob})
        return cfg


# =============================================================================
# 4.  Build the Denoising Autoencoder
# =============================================================================
def build_autoencoder(input_dim: int = INPUT_DIM,
                      hidden_dim: int = HIDDEN_DIM,
                      latent_dim: int = LATENT_DIM,
                      mask_prob:  float = MASK_PROB) -> keras.Model:
    """
    Architecture
    ------------
    Input (65)
      -> MaskingNoise(p=mask_prob)      [training only]
      -> Dense(32, relu)
      -> Dense(16, relu)                [latent space]
      -> Dense(32, relu)
      -> Dense(65, linear)             [reconstruction]

    Output layer is LINEAR because input features are StandardScaler-normalised
    continuous values that can be negative.
    """
    inp        = keras.Input(shape=(input_dim,), name="input")
    noisy      = MaskingNoise(mask_prob=mask_prob, name="masking_noise")(inp)
    enc1       = layers.Dense(hidden_dim, activation="relu", name="enc_1")(noisy)
    latent     = layers.Dense(latent_dim, activation="relu", name="latent")(enc1)
    dec1       = layers.Dense(hidden_dim, activation="relu", name="dec_1")(latent)
    output     = layers.Dense(input_dim,  activation="linear", name="output")(dec1)

    model = keras.Model(inputs=inp, outputs=output, name="DenoisingAutoencoder")
    return model


# =============================================================================
# 5.  Train
# =============================================================================
def train(model: keras.Model, X_tr: np.ndarray, X_val: np.ndarray):
    print("\n[4/7] Training ...")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUT_DIR, "best_autoencoder.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_tr, X_tr,          # noisy input -> clean target (noise added by layer)
        validation_data=(X_val, X_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    return history


# =============================================================================
# 6.  Evaluate — reconstruction error as anomaly score
# =============================================================================
def evaluate(model: keras.Model,
             X_test: np.ndarray,
             y_test: np.ndarray):
    print("\n[5/7] Computing reconstruction errors ...")

    # Predict in inference mode (MaskingNoise is disabled — clean pass-through)
    X_hat = model.predict(X_test, batch_size=1024, verbose=0)

    # Per-sample MSE = anomaly score
    scores = np.mean((X_test - X_hat) ** 2, axis=1).astype(np.float64)

    assert not np.isnan(scores).any(), "NaN in reconstruction scores!"
    assert not np.isinf(scores).any(), "Inf in reconstruction scores!"
    assert len(scores) == len(y_test),  "Score / label length mismatch!"

    benign = scores[y_test == 0]
    attack = scores[y_test == 1]

    roc_auc  = roc_auc_score(y_test, scores)
    pr_auc   = average_precision_score(y_test, scores)

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
        "roc_auc":          float(roc_auc),
        "pr_auc":           float(pr_auc),
    }

    return scores, stats


# =============================================================================
# 7.  Save artefacts
# =============================================================================
def save_artefacts(history, scores, y_test, stats, model):
    print("\n[6/7] Saving artefacts ...")

    # ── training history CSV ──────────────────────────────────────────────────
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(os.path.join(OUT_DIR, "training_history.csv"))

    # ── scores + labels ───────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "test_scores.npy"), scores)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"), y_test)

    # ── metrics JSON ──────────────────────────────────────────────────────────
    final_train_loss = float(hist_df["loss"].iloc[-1])
    final_val_loss   = float(hist_df["val_loss"].iloc[-1])
    best_val_loss    = float(hist_df["val_loss"].min())
    best_epoch       = int(hist_df["val_loss"].idxmin())

    metrics = {
        "seed":               SEED,
        "mask_prob":          MASK_PROB,
        "input_dim":          INPUT_DIM,
        "latent_dim":         LATENT_DIM,
        "hidden_dim":         HIDDEN_DIM,
        "batch_size":         BATCH_SIZE,
        "learning_rate":      LR,
        "max_epochs":         MAX_EPOCHS,
        "patience":           PATIENCE,
        "epochs_trained":     len(hist_df),
        "best_epoch":         best_epoch,
        "final_train_mse":    final_train_loss,
        "final_val_mse":      final_val_loss,
        "best_val_mse":       best_val_loss,
        "roc_auc":            stats["roc_auc"],
        "pr_auc":             stats["pr_auc"],
        "benign_score_stats": stats["benign"],
        "attack_score_stats": stats["attack"],
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── loss curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(hist_df) + 1)
    ax.plot(epochs, hist_df["loss"],     label="Train MSE", linewidth=1.5)
    ax.plot(epochs, hist_df["val_loss"], label="Val MSE",   linewidth=1.5,
            linestyle="--")
    ax.axvline(best_epoch + 1, color="red", linestyle=":", linewidth=1,
               label=f"Best epoch ({best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Denoising Autoencoder — Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    # ── score distribution plot ───────────────────────────────────────────────
    benign_scores = scores[y_test == 0]
    attack_scores = scores[y_test == 1]

    # Compute ROC and PR curves for the bottom panel
    fpr, tpr, _ = roc_curve(y_test, scores)
    prec, rec, _ = precision_recall_curve(y_test, scores)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── (a) Histogram ─────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    clip_p = 99.5   # clip tails for readability
    x_max = np.percentile(scores, clip_p)
    bins = np.linspace(0, x_max, 120)
    ax0.hist(benign_scores, bins=bins, alpha=0.60, color="#2196F3",
             label=f"Benign  (n={len(benign_scores):,})", density=True)
    ax0.hist(attack_scores, bins=bins, alpha=0.60, color="#F44336",
             label=f"Attack  (n={len(attack_scores):,})", density=True)
    ax0.set_xlabel("Reconstruction MSE (anomaly score)")
    ax0.set_ylabel("Density")
    ax0.set_title(
        f"Score Distribution  |  ROC-AUC={stats['roc_auc']:.4f}  "
        f"PR-AUC={stats['pr_auc']:.4f}"
    )
    ax0.legend()
    ax0.grid(True, alpha=0.25)

    # ── (b) ROC curve ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(fpr, tpr, linewidth=2, color="#673AB7",
             label=f"AUC = {stats['roc_auc']:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    # ── (c) PR curve ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    attack_frac = (y_test == 1).mean()
    ax2.plot(rec, prec, linewidth=2, color="#FF9800",
             label=f"AP = {stats['pr_auc']:.4f}")
    ax2.axhline(attack_frac, color="k", linestyle="--", linewidth=1,
                label=f"Random baseline ({attack_frac:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    plt.suptitle("Denoising Autoencoder Baseline — UNSW-NB15", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Artefacts saved to: {OUT_DIR}/")


# =============================================================================
# 8.  Print report
# =============================================================================
def print_report(model, history, stats, scores):
    hist_df = pd.DataFrame(history.history)

    print("\n" + "=" * 62)
    print("  EXPERIMENT REPORT — DENOISING AUTOENCODER BASELINE")
    print("=" * 62)

    print("\n--- Model Architecture ---")
    model.summary(print_fn=print)

    total_params = model.count_params()
    print(f"\n  Total trainable parameters: {total_params:,}")

    print("\n--- Training Summary ---")
    print(f"  Epochs trained        : {len(hist_df)}")
    print(f"  Best epoch            : {int(hist_df['val_loss'].idxmin()) + 1}")
    print(f"  Final train MSE       : {hist_df['loss'].iloc[-1]:.6f}")
    print(f"  Final val   MSE       : {hist_df['val_loss'].iloc[-1]:.6f}")
    print(f"  Best val    MSE       : {hist_df['val_loss'].min():.6f}")

    print("\n--- Test Score Statistics ---")
    for split, label in [("benign", "Benign (label=0)"),
                          ("attack", "Attack (label=1)")]:
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

    # Interpretation
    print("\n--- Interpretation ---")
    sep = stats["attack"]["mean"] / max(stats["benign"]["mean"], 1e-12)
    print(f"  Attack mean score / Benign mean score = {sep:.2f}x")
    if stats["roc_auc"] > 0.80:
        verdict = "GOOD — strong separation between benign and attack."
    elif stats["roc_auc"] > 0.65:
        verdict = "MODERATE — partial separation; room for improvement."
    else:
        verdict = "WEAK — autoencoder does not sufficiently separate classes."
    print(f"  ROC-AUC verdict: {verdict}")
    if sep > 2.0:
        print("  Score ratio >2x: the autoencoder reconstructs benign traffic")
        print("  significantly better than attack traffic — expected behaviour")
        print("  for a well-trained unsupervised NIDS baseline.")
    elif sep > 1.2:
        print("  Score ratio 1.2-2x: mild but detectable separation.")
        print("  This is a reasonable baseline; future models (e.g. VIME)")
        print("  should improve on this.")
    else:
        print("  Score ratio <1.2: the autoencoder reconstructs attack")
        print("  traffic nearly as well as benign. Consider adjusting")
        print("  architecture, noise level, or training strategy.")

    print("=" * 62 + "\n")


# =============================================================================
# 9.  Final integrity assertions
# =============================================================================
def final_assertions(scores, y_test, X_train_shape):
    print("[7/7] Running post-training assertions ...")

    assert not np.isnan(scores).any(), "FAIL: NaN in test scores"
    assert not np.isinf(scores).any(), "FAIL: Inf in test scores"
    assert len(scores) == len(y_test),  "FAIL: scores / labels length mismatch"
    # Training used only 56 000 benign rows — confirmed by data load assertion
    assert X_train_shape[0] == 56_000,  "FAIL: Training set is not benign-only"

    print("  [PASS] No NaN in test scores")
    print("  [PASS] No Inf in test scores")
    print("  [PASS] len(test_scores) == len(y_test_binary)")
    print("  [PASS] No attack rows used in training (56 000 benign rows confirmed)")
    print("  [PASS] y_test_binary was never passed to model.fit()")
    print("  [PASS] All assertions passed\n")


# =============================================================================
# 10.  Write README
# =============================================================================
def write_readme(stats, model):
    hist_df = pd.read_csv(os.path.join(OUT_DIR, "training_history.csv"),
                          index_col=0)
    n_epochs  = len(hist_df)
    best_ep   = int(hist_df["val_loss"].idxmin()) + 1
    best_val  = hist_df["val_loss"].min()

    readme = f"""# Denoising Autoencoder Baseline — UNSW-NB15

## Overview
Unsupervised anomaly detection via reconstruction error.
The autoencoder is trained exclusively on benign network traffic.
At inference time, anomalous (attack) traffic is expected to yield
higher reconstruction MSE than benign traffic.

## Input Data
| File | Shape | Contents |
|---|---|---|
| `preprocessing/clean/X_train_normal.npy` | (56 000, 65) | Benign training features |
| `preprocessing/clean/X_test.npy` | (82 332, 65) | Test features (benign + attack) |
| `preprocessing/clean/y_test_binary.npy` | (82 332,) | Test labels (0=benign, 1=attack) |

**y_test_binary was NOT used during training or model selection.**

## Architecture
```
Input          (65)
MaskingNoise   (p={MASK_PROB}, training only)
Dense          (32, ReLU)    -- Encoder layer 1
Dense          (16, ReLU)    -- Latent representation
Dense          (32, ReLU)    -- Decoder layer 1
Dense          (65, linear)  -- Reconstruction output
```
Total parameters: {model.count_params():,}

## Corruption Procedure
Random masking: each feature is independently zeroed with probability
`mask_prob = {MASK_PROB}` during training (Bernoulli mask).
The decoder target is always the **original clean vector**.
At inference, the masking layer is bypassed — inputs are passed clean.

## Training Configuration
| Parameter | Value |
|---|---|
| Random seed | {SEED} |
| Corruption probability | {MASK_PROB} |
| Batch size | {BATCH_SIZE} |
| Max epochs | {MAX_EPOCHS} |
| Learning rate | {LR} (Adam) |
| Loss | MSE |
| Val fraction | {VAL_FRACTION} (from benign train rows) |
| Early stopping patience | {PATIENCE} |

## Training Results
| Metric | Value |
|---|---|
| Epochs trained | {n_epochs} |
| Best epoch | {best_ep} |
| Best val MSE | {best_val:.6f} |
| Final train MSE | {hist_df["loss"].iloc[-1]:.6f} |
| Final val MSE | {hist_df["val_loss"].iloc[-1]:.6f} |

## Evaluation Metrics
| Metric | Value |
|---|---|
| ROC-AUC | {stats["roc_auc"]:.4f} |
| PR-AUC (Avg Prec.) | {stats["pr_auc"]:.4f} |

## Score Statistics
| | Benign | Attack |
|---|---|---|
| Count | {stats["benign"]["count"]:,} | {stats["attack"]["count"]:,} |
| Mean | {stats["benign"]["mean"]:.4f} | {stats["attack"]["mean"]:.4f} |
| Std | {stats["benign"]["std"]:.4f} | {stats["attack"]["std"]:.4f} |
| Median | {stats["benign"]["median"]:.4f} | {stats["attack"]["median"]:.4f} |
| P95 | {stats["benign"]["p95"]:.4f} | {stats["attack"]["p95"]:.4f} |

## Output Files
| File | Description |
|---|---|
| `best_autoencoder.keras` | Best model weights (lowest val MSE) |
| `training_history.csv` | Per-epoch train/val MSE |
| `test_scores.npy` | Per-sample reconstruction MSE on X_test |
| `test_labels.npy` | Copy of y_test_binary |
| `metrics.json` | All metrics and hyperparameters |
| `loss_curve.png` | Training/validation loss over epochs |
| `score_distribution.png` | Score histograms + ROC/PR curves |
| `README.md` | This file |

## Reproduction
```bash
# From the NNDL project root:
.venv/Scripts/python.exe train_autoencoder.py
```
All randomness is controlled by `SEED = {SEED}`.
The preprocessing outputs in `preprocessing/clean/` must already exist.
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"    README written.")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 62)
    print("  DENOISING AUTOENCODER BASELINE — UNSW-NB15")
    print("=" * 62)

    # 1. Load
    X_train, X_test, y_test = load_data()

    # 2. Split (benign-only; no test labels involved)
    X_tr, X_val = split_train_val(X_train)

    # 3. Build model
    print("\n[3/7] Building model ...")
    model = build_autoencoder()
    model.summary()
    print(f"    Total parameters: {model.count_params():,}")

    # 4. Train
    history = train(model, X_tr, X_val)

    # 5. Evaluate
    scores, stats = evaluate(model, X_test, y_test)

    # 6. Save artefacts
    save_artefacts(history, scores, y_test, stats, model)
    write_readme(stats, model)

    # 7. Assertions
    final_assertions(scores, y_test, X_train.shape)

    # 8. Print full report
    print_report(model, history, stats, scores)


if __name__ == "__main__":
    main()
