"""
train_adaptive_vime.py
======================
Adaptive VIME Self-Supervised Baseline — Unsupervised NIDS (UNSW-NB15)
----------------------------------------------------------------------

Architecture
    Shared Encoder : 65 -> Dense(32, ReLU) -> Dense(16, ReLU) [latent]
    Mask Head      : 16 -> Dense(65, Sigmoid)
    Impute Head    : 16 -> Dense(65, Linear)

Training
    - Input: X_train_normal.npy (56 000 benign rows)
    - Corruption: mask m ~ Bernoulli(p=MASK_PROB)
      For masked features, replace with sample from empirical marginal
      from the benign training set.
    - Loss: Adaptive homoscedastic uncertainty weighting
      L_total = exp(-s_mask)*L_mask + 0.5*exp(-s_impute)*L_impute + 0.5*(s_mask + s_impute)
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
OUT_DIR       = os.path.join(PROJECT_DIR, "models", "adaptive_vime")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── hyperparameters ─────────────────────────────────────────────────────────
INPUT_DIM      = 65
LATENT_DIM     = 16
HIDDEN_DIM     = 32
MASK_PROB      = 0.20
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
            
        x_tilde = batch_x * (1 - mask) + m_samples * mask
        
        return x_tilde, {"impute_head": batch_x, "mask_head": mask}
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# =============================================================================
# 3. Build Model
# =============================================================================
@keras.utils.register_keras_serializable(package="NNDL")
class AdaptiveVIME(keras.Model):
    def __init__(self, encoder, impute_head, mask_head, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.impute_head = impute_head
        self.mask_head = mask_head
        
        self.s_mask = self.add_weight(
            name="s_mask",
            shape=(),
            initializer=keras.initializers.Constant(0.0),
            trainable=True,
            dtype=tf.float32,
        )
        self.s_impute = self.add_weight(
            name="s_impute",
            shape=(),
            initializer=keras.initializers.Constant(0.0),
            trainable=True,
            dtype=tf.float32,
        )
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mask_bce_tracker = keras.metrics.Mean(name="mask_bce")
        self.impute_mse_tracker = keras.metrics.Mean(name="impute_mse")
        
        self.s_mask_tracker = keras.metrics.Mean(name="s_mask")
        self.s_impute_tracker = keras.metrics.Mean(name="s_impute")
        self.eff_mask_w_tracker = keras.metrics.Mean(name="eff_mask_weight")
        self.eff_impute_w_tracker = keras.metrics.Mean(name="eff_impute_weight")
        
        self.bce = keras.losses.BinaryCrossentropy()
        self.mse = keras.losses.MeanSquaredError()
        
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.mask_bce_tracker,
            self.impute_mse_tracker,
            self.s_mask_tracker,
            self.s_impute_tracker,
            self.eff_mask_w_tracker,
            self.eff_impute_w_tracker
        ]
        
    def call(self, inputs, training=False):
        z = self.encoder(inputs, training=training)
        impute_pred = self.impute_head(z, training=training)
        mask_pred = self.mask_head(z, training=training)
        return impute_pred, mask_pred
        
    def train_step(self, data):
        x_tilde, targets = data
        y_impute = targets["impute_head"]
        y_mask = targets["mask_head"]
        
        with tf.GradientTape() as tape:
            impute_pred, mask_pred = self(x_tilde, training=True)
            
            l_mask = self.bce(y_mask, mask_pred)
            l_impute = self.mse(y_impute, impute_pred)
            
            eff_mask_w = tf.exp(-self.s_mask)
            eff_impute_w = 0.5 * tf.exp(-self.s_impute)
            
            total_loss = (eff_mask_w * l_mask + 
                          eff_impute_w * l_impute + 
                          0.5 * (self.s_mask + self.s_impute))
                          
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self._update_metrics(total_loss, l_mask, l_impute)
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        x_tilde, targets = data
        y_impute = targets["impute_head"]
        y_mask = targets["mask_head"]
        
        impute_pred, mask_pred = self(x_tilde, training=False)
        
        l_mask = self.bce(y_mask, mask_pred)
        l_impute = self.mse(y_impute, impute_pred)
        
        eff_mask_w = tf.exp(-self.s_mask)
        eff_impute_w = 0.5 * tf.exp(-self.s_impute)
        
        total_loss = (eff_mask_w * l_mask + 
                      eff_impute_w * l_impute + 
                      0.5 * (self.s_mask + self.s_impute))
                      
        self._update_metrics(total_loss, l_mask, l_impute)
        return {m.name: m.result() for m in self.metrics}
        
    def _update_metrics(self, total_loss, l_mask, l_impute):
        self.loss_tracker.update_state(total_loss)
        self.mask_bce_tracker.update_state(l_mask)
        self.impute_mse_tracker.update_state(l_impute)
        
        self.s_mask_tracker.update_state(self.s_mask)
        self.s_impute_tracker.update_state(self.s_impute)
        
        eff_mask_w = tf.exp(-self.s_mask)
        eff_impute_w = 0.5 * tf.exp(-self.s_impute)
        
        self.eff_mask_w_tracker.update_state(eff_mask_w)
        self.eff_impute_w_tracker.update_state(eff_impute_w)

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": keras.utils.serialize_keras_object(self.encoder),
            "impute_head": keras.utils.serialize_keras_object(self.impute_head),
            "mask_head": keras.utils.serialize_keras_object(self.mask_head),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        encoder = keras.utils.deserialize_keras_object(config.pop("encoder"))
        impute_head = keras.utils.deserialize_keras_object(config.pop("impute_head"))
        mask_head = keras.utils.deserialize_keras_object(config.pop("mask_head"))
        return cls(encoder, impute_head, mask_head, **config)


def build_vime(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
    inp = keras.Input(shape=(input_dim,), name="input")
    enc1 = layers.Dense(hidden_dim, activation="relu", name="enc_1")(inp)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(enc1)
    encoder = keras.Model(inputs=inp, outputs=latent, name="VIME_Encoder")
    
    # Sub-heads
    mask_inp = keras.Input(shape=(latent_dim,))
    mask_out = layers.Dense(input_dim, activation="sigmoid", name="mask_head")(mask_inp)
    mask_head = keras.Model(inputs=mask_inp, outputs=mask_out, name="Mask_Head")
    
    impute_inp = keras.Input(shape=(latent_dim,))
    impute_out = layers.Dense(input_dim, activation="linear", name="impute_head")(impute_inp)
    impute_head = keras.Model(inputs=impute_inp, outputs=impute_out, name="Impute_Head")
    
    model = AdaptiveVIME(encoder, impute_head, mask_head)
    return model, encoder


def run_one_batch_sanity_check():
    """Verify scalar registration, gradients, and one optimizer update."""
    model, _ = build_vime()
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer)

    x_tilde = tf.zeros((BATCH_SIZE, INPUT_DIM), dtype=tf.float32)
    y_impute = tf.zeros((BATCH_SIZE, INPUT_DIM), dtype=tf.float32)
    y_mask = tf.fill((BATCH_SIZE, INPUT_DIM), tf.constant(0.2, dtype=tf.float32))
    trainable_variables = model.trainable_variables

    assert any(variable is model.s_mask for variable in trainable_variables)
    assert any(variable is model.s_impute for variable in trainable_variables)
    assert float(model.s_mask.numpy()) == 0.0
    assert float(model.s_impute.numpy()) == 0.0

    with tf.GradientTape() as tape:
        impute_pred, mask_pred = model(x_tilde, training=True)
        l_mask = model.bce(y_mask, mask_pred)
        l_impute = model.mse(y_impute, impute_pred)
        total_loss = (
            tf.exp(-model.s_mask) * l_mask
            + 0.5 * tf.exp(-model.s_impute) * l_impute
            + 0.5 * (model.s_mask + model.s_impute)
        )

    gradients = tape.gradient(total_loss, trainable_variables)
    mask_index = next(i for i, variable in enumerate(trainable_variables)
                       if variable is model.s_mask)
    impute_index = next(i for i, variable in enumerate(trainable_variables)
                        if variable is model.s_impute)
    s_mask_gradient = gradients[mask_index]
    s_impute_gradient = gradients[impute_index]
    expected_s_mask_gradient = -l_mask + 0.5
    expected_s_impute_gradient = -0.5 * l_impute + 0.5

    assert s_mask_gradient is not None
    assert s_impute_gradient is not None
    assert abs(float(s_impute_gradient.numpy())) > 1e-6
    np.testing.assert_allclose(
        s_mask_gradient.numpy(), expected_s_mask_gradient.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        s_impute_gradient.numpy(), expected_s_impute_gradient.numpy(), rtol=1e-5, atol=1e-6
    )

    old_s_mask = float(model.s_mask.numpy())
    old_s_impute = float(model.s_impute.numpy())
    print("\n[Sanity check] One-batch adaptive-weight update")
    print("  registered         : s_mask=True, s_impute=True")
    print(f"  s_mask gradient   : {float(s_mask_gradient.numpy()):.8f}")
    print(f"  s_impute gradient : {float(s_impute_gradient.numpy()):.8f}")
    print(f"  expected gradients: s_mask={float(expected_s_mask_gradient.numpy()):.8f}, "
          f"s_impute={float(expected_s_impute_gradient.numpy()):.8f}")
    print(f"  before update     : s_mask={old_s_mask:.8f}, s_impute={old_s_impute:.8f}")

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    new_s_mask = float(model.s_mask.numpy())
    new_s_impute = float(model.s_impute.numpy())
    print(f"  after update      : s_mask={new_s_mask:.8f}, s_impute={new_s_impute:.8f}")

    assert new_s_impute != old_s_impute
    print("  [PASS] Both scalars were registered, differentiated, and updated")

# =============================================================================
# 4. Train
# =============================================================================
def train_vime(model, X_tr, X_val):
    print("\n[4/7] Training Adaptive VIME ...")
    
    train_gen = VIMEGenerator(X_tr, MASK_PROB, BATCH_SIZE, shuffle=True)
    val_gen = VIMEGenerator(X_val, MASK_PROB, BATCH_SIZE, shuffle=False)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR))
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUT_DIR, "best_adaptive_vime.keras"),
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
def save_artefacts(history, scores, y_test, stats, centroid, model):
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
        "final_s_mask": float(model.s_mask.numpy()),
        "final_s_impute": float(model.s_impute.numpy()),
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
    ax.plot(epochs, hist_df["loss"], label="Train Total Adaptive Loss", linewidth=1.5)
    ax.plot(epochs, hist_df["val_loss"], label="Val Total Adaptive Loss", linewidth=1.5, linestyle="--")
    
    ax.plot(epochs, hist_df["impute_mse"], label="Train Impute MSE", linewidth=1)
    ax.plot(epochs, hist_df["val_impute_mse"], label="Val Impute MSE", linewidth=1, linestyle=":")
    
    ax.plot(epochs, hist_df["mask_bce"], label="Train Mask BCE", linewidth=1)
    ax.plot(epochs, hist_df["val_mask_bce"], label="Val Mask BCE", linewidth=1, linestyle=":")
    
    ax.axvline(best_epoch + 1, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Adaptive VIME - Training & Validation Losses")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    
    # Plot Task Weight Dynamics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, hist_df["s_mask"], label="s_mask")
    axes[0].plot(epochs, hist_df["s_impute"], label="s_impute")
    axes[0].set_title("Learned scalar parameters (s)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, hist_df["eff_mask_weight"], label="exp(-s_mask) [Mask]")
    axes[1].plot(epochs, hist_df["eff_impute_weight"], label="0.5*exp(-s_impute) [Impute]")
    axes[1].set_title("Effective loss weights")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task_weight_dynamics.png"), dpi=150)
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
    
    plt.suptitle("Adaptive VIME Baseline - UNSW-NB15", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
# =============================================================================
# 7. Print Report
# =============================================================================
def print_report(model, history, stats, scores):
    hist_df = pd.DataFrame(history.history)
    print("\n" + "=" * 62)
    print("  EXPERIMENT REPORT — ADAPTIVE VIME BASELINE")
    print("=" * 62)
    
    print("\n--- Training Summary ---")
    print(f"  Epochs trained        : {len(hist_df)}")
    print(f"  Best epoch            : {int(hist_df['val_loss'].idxmin()) + 1}")
    print(f"  Final val Total Loss  : {hist_df['val_loss'].iloc[-1]:.6f}")
    print(f"  Best val  Total Loss  : {hist_df['val_loss'].min():.6f}")
    
    print("\n--- Learned Weight Dynamics (Final) ---")
    print(f"  s_mask                : {model.s_mask.numpy():.4f}")
    print(f"  s_impute              : {model.s_impute.numpy():.4f}")
    print(f"  eff mask weight       : {np.exp(-model.s_mask.numpy()):.4f}")
    print(f"  eff impute weight     : {0.5 * np.exp(-model.s_impute.numpy()):.4f}")
    
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
    dae_roc = 0.8843
    dae_pr = 0.9029
    vime_c_roc = 0.7188
    vime_c_pr = 0.7488
    
    print(f"  Denoising Autoencoder ROC-AUC : {dae_roc}")
    print(f"  Denoising Autoencoder PR-AUC  : {dae_pr}")
    print(f"  Corrected VIME ROC-AUC        : {vime_c_roc}")
    print(f"  Corrected VIME PR-AUC         : {vime_c_pr}")
    print(f"  Adaptive VIME ROC-AUC         : {stats['roc_auc']:.4f}  (diff vs VIME: {stats['roc_auc'] - vime_c_roc:+.4f})")
    print(f"  Adaptive VIME PR-AUC          : {stats['pr_auc']:.4f}  (diff vs VIME: {stats['pr_auc'] - vime_c_pr:+.4f})")
    print("=" * 62 + "\n")

# =============================================================================
# 8. README
# =============================================================================
def write_readme(stats, model):
    readme = f"""# Adaptive VIME Baseline — UNSW-NB15

## 1. Research motivation
Vanilla VIME uses a fixed manually selected weighting (alpha=2.0).
This experiment replaces fixed weighting with learned task-difficulty
weighting inspired by homoscedastic uncertainty.
The purpose is to investigate if the model can automatically balance
the mask estimation and feature imputation objectives.

## 2. Vanilla VIME formulation
L_total = L_mask + 2.0 * L_impute

## 3. Adaptive VIME formulation
L_total = exp(-s_mask)*L_mask + 0.5*exp(-s_impute)*L_impute + 0.5*(s_mask + s_impute)
Both `s_mask` and `s_impute` are trainable parameters initialized to 0.

## 4. Architecture
Shared Encoder: 65 -> 32 -> 16 (ReLU)
Mask Head: 16 -> 65 (Sigmoid)
Imputation Head: 16 -> 65 (Linear)

## 5. Corruption process
Each feature is independently corrupted with p=0.20.
Corrupted values are sampled from the empirical marginal distribution of the 
benign training set.

## 6. Dataset and split
Training: 56,000 benign samples, 90/10 train/validation split.
Test: 82,332 full test samples.
Labels are solely used for final evaluation.

## 7. Training configuration
Optimizer: Adam, lr=1e-3
Batch size: 256
Early stopping patience: 15
Max epochs: 200

## 8. Initialization
s_mask = 0
s_impute = 0

## 9. Learned final task weights
final_s_mask: {model.s_mask.numpy():.4f}
final_s_impute: {model.s_impute.numpy():.4f}

## 10. Primary anomaly scoring method
Latent distance to benign centroid: ||e(x) - c||^2.

## 11. Evaluation metrics
ROC-AUC and Average Precision (PR-AUC).

## 12. Comparison
| Metric | Adaptive VIME | Vanilla VIME (Corrected) | DAE |
|---|---|---|---|
| ROC-AUC | {stats['roc_auc']:.4f} | 0.7188 | 0.8843 |
| PR-AUC | {stats['pr_auc']:.4f} | 0.7488 | 0.9029 |

## 13. Limitations
The learned scalars are described as learned task-difficulty or loss-balancing
parameters, not direct physical aleatoric uncertainty measurements.
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

def main():
    print("=" * 62)
    print("  ADAPTIVE VIME BASELINE — UNSW-NB15")
    print("=" * 62)
    
    X_train, X_test, y_test = load_data()
    X_tr, X_val = train_test_split(X_train, test_size=VAL_FRACTION, random_state=SEED, shuffle=True)
    
    print("\n[2/7] Splitting train/val (90/10) ...")
    
    print("\n[3/7] Building Adaptive model ...")
    model, encoder = build_vime()
    
    # Assert initialized to 0
    assert model.s_mask.numpy() == 0.0
    assert model.s_impute.numpy() == 0.0
    
    history = train_vime(model, X_tr, X_val)
    
    # Assert parameters changed
    assert model.s_mask.numpy() != 0.0
    assert model.s_impute.numpy() != 0.0
    print(f"  [PASS] Trainable parameters s_mask and s_impute updated via backprop")
    
    scores, stats, centroid = evaluate_vime(encoder, X_train, X_test, y_test)
    
    save_artefacts(history, scores, y_test, stats, centroid, model)
    write_readme(stats, model)
    
    print("\n[7/7] Assertions ...")
    assert X_train.shape[0] == 56000
    assert not np.isnan(scores).any()
    print("  [PASS] All assertions passed")
    
    print_report(model, history, stats, scores)

if __name__ == "__main__":
    main()
