"""Run isolated Mask-only and Imputation-only VIME pretext-task ablations."""

import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
INPUT_DIM = 65
LATENT_DIM = 16
HIDDEN_DIM = 32
MASK_PROB = 0.20
BATCH_SIZE = 256
MAX_EPOCHS = 200
LR = 1e-3
VAL_FRACTION = 0.10
PATIENCE = 15

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROC_DIR = os.path.join(PROJECT_DIR, "preprocessing", "clean")


def reset_seed():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


class VIMEGenerator(keras.utils.Sequence):
    def __init__(self, x, objective, shuffle):
        self.x = x
        self.objective = objective
        self.shuffle = shuffle
        self.n = x.shape[0]
        self.d = x.shape[1]
        self.indices = np.arange(self.n)
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.n / BATCH_SIZE))

    def __getitem__(self, index):
        batch_idx = self.indices[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
        batch_x = self.x[batch_idx]
        batch_size = batch_x.shape[0]
        mask = np.random.binomial(
            1, MASK_PROB, size=batch_x.shape
        ).astype(np.float32)
        marginal_samples = np.zeros_like(batch_x)
        for feature_index in range(self.d):
            marginal_samples[:, feature_index] = np.random.choice(
                self.x[:, feature_index], size=batch_size, replace=True
            )
        x_tilde = batch_x * (1 - mask) + marginal_samples * mask
        target = mask if self.objective == "mask" else batch_x
        return x_tilde, target

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_data():
    x_train = np.load(os.path.join(PREPROC_DIR, "X_train_normal.npy")).astype(np.float32)
    x_test = np.load(os.path.join(PREPROC_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PREPROC_DIR, "y_test_binary.npy"))
    assert x_train.shape == (56000, INPUT_DIM)
    assert x_test.shape == (82332, INPUT_DIM)
    assert y_test.shape == (82332,)
    return x_train, x_test, y_test


def build_model(objective):
    input_layer = keras.Input(shape=(INPUT_DIM,), name="input")
    hidden = layers.Dense(HIDDEN_DIM, activation="relu", name="enc_1")(input_layer)
    latent = layers.Dense(LATENT_DIM, activation="relu", name="latent")(hidden)
    if objective == "mask":
        output = layers.Dense(
            INPUT_DIM, activation="sigmoid", name="mask_head"
        )(latent)
    else:
        output = layers.Dense(
            INPUT_DIM, activation="linear", name="impute_head"
        )(latent)
    model = keras.Model(inputs=input_layer, outputs=output, name=f"VIME_{objective}_only")
    encoder = keras.Model(inputs=input_layer, outputs=latent, name="VIME_Encoder")
    return model, encoder


def evaluate(encoder, x_train, x_test, y_test):
    train_latent = encoder.predict(x_train, batch_size=1024, verbose=0)
    centroid = np.mean(train_latent, axis=0)
    test_latent = encoder.predict(x_test, batch_size=1024, verbose=0)
    scores = np.sum((test_latent - centroid) ** 2, axis=1).astype(np.float64)
    assert np.isfinite(scores).all()
    return centroid, scores, {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
    }


def run_experiment(objective, x_train, x_test, y_test):
    reset_seed()
    output_name = f"vime_{objective}_only"
    output_dir = os.path.join(PROJECT_DIR, "models", output_name)
    os.makedirs(output_dir, exist_ok=True)

    x_tr, x_val = train_test_split(
        x_train, test_size=VAL_FRACTION, random_state=SEED, shuffle=True
    )
    model, encoder = build_model(objective)
    loss = "binary_crossentropy" if objective == "mask" else "mse"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=loss,
    )
    train_generator = VIMEGenerator(x_tr, objective, shuffle=True)
    val_generator = VIMEGenerator(x_val, objective, shuffle=False)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=MAX_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, f"best_{output_name}.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ],
        verbose=2,
    )

    centroid, scores, metrics = evaluate(encoder, x_train, x_test, y_test)
    history_df = pd.DataFrame(history.history)
    best_index = history_df["val_loss"].idxmin()
    metrics.update({
        "seed": SEED,
        "objective": objective,
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "mask_prob": MASK_PROB,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "train_rows": int(x_tr.shape[0]),
        "validation_rows": int(x_val.shape[0]),
        "epochs_trained": len(history_df),
        "best_epoch": int(best_index) + 1,
        "best_val_loss": float(history_df.loc[best_index, "val_loss"]),
        "final_val_loss": float(history_df["val_loss"].iloc[-1]),
    })
    history_df.index.name = "epoch"
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"))
    np.save(os.path.join(output_dir, "latent_centroid.npy"), centroid)
    np.save(os.path.join(output_dir, "test_scores.npy"), scores)
    np.save(os.path.join(output_dir, "test_labels.npy"), y_test)
    with open(os.path.join(output_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as file:
        file.write(
            f"# VIME {objective.title()}-only Ablation\n\n"
            f"Objective: {'mask BCE' if objective == 'mask' else 'imputation MSE'} only.\n"
            "All other settings match corrected Vanilla VIME.\n"
        )
    return metrics


def main():
    x_train, x_test, y_test = load_data()
    for objective in ("mask", "impute"):
        print(f"\n=== VIME {objective.upper()}-ONLY ABLATION ===")
        metrics = run_experiment(objective, x_train, x_test, y_test)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()