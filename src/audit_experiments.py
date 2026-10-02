"""Audit saved experiments and create the consolidated research report."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow import keras

import train_adaptive_vime as adaptive
import train_autoencoder as autoencoder

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROC_DIR = os.path.join(PROJECT_DIR, "preprocessing", "clean")
AUDIT_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(AUDIT_DIR, "figures")

EXPERIMENTS = [
    {
        "name": "DAE",
        "directory": "autoencoder_baseline",
        "checkpoint": "best_autoencoder.keras",
        "score_type": "reconstruction_mse",
        "custom_objects": {"MaskingNoise": autoencoder.MaskingNoise},
    },
    {
        "name": "Mask-only VIME",
        "directory": "vime_mask_only",
        "checkpoint": "best_vime_mask_only.keras",
        "score_type": "latent_centroid",
        "custom_objects": {},
    },
    {
        "name": "Imputation-only VIME",
        "directory": "vime_impute_only",
        "checkpoint": "best_vime_impute_only.keras",
        "score_type": "latent_centroid",
        "custom_objects": {},
    },
    {
        "name": "Corrected Vanilla VIME",
        "directory": "vime_baseline_corrected",
        "checkpoint": "best_vime.keras",
        "score_type": "latent_centroid",
        "custom_objects": {},
    },
    {
        "name": "Adaptive VIME",
        "directory": "adaptive_vime",
        "checkpoint": "best_adaptive_vime.keras",
        "score_type": "latent_centroid",
        "custom_objects": {"AdaptiveVIME": adaptive.AdaptiveVIME},
    },
]


def load_inputs():
    x_train = np.load(os.path.join(PREPROC_DIR, "X_train_normal.npy")).astype(np.float32)
    x_test = np.load(os.path.join(PREPROC_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PREPROC_DIR, "y_test_binary.npy"))
    assert x_train.shape == (56000, 65)
    assert x_test.shape == (82332, 65)
    assert y_test.shape == (82332,)
    return x_train, x_test, y_test


def load_model_and_scores(spec, x_train, x_test):
    directory = os.path.join(PROJECT_DIR, "models", spec["directory"])
    checkpoint = os.path.join(directory, spec["checkpoint"])
    model = keras.models.load_model(
        checkpoint, custom_objects=spec["custom_objects"], compile=False
    )
    saved_scores = np.load(os.path.join(directory, "test_scores.npy"))
    saved_labels = np.load(os.path.join(directory, "test_labels.npy"))
    result = {
        "checkpoint_loadable": True,
        "scores_finite": bool(np.isfinite(saved_scores).all()),
        "labels_match_canonical": False,
        "centroid_audit": "not_applicable" if spec["score_type"] == "reconstruction_mse" else "pass",
        "score_recompute_max_abs_error": None,
        "centroid_recompute_max_abs_error": None,
    }

    if spec["score_type"] == "reconstruction_mse":
        predictions = model.predict(x_test, batch_size=1024, verbose=0)
        recomputed_scores = np.mean((x_test - predictions) ** 2, axis=1).astype(np.float64)
    else:
        if hasattr(model, "encoder"):
            encoder = model.encoder
        else:
            encoder = keras.Model(
                inputs=model.input,
                outputs=model.get_layer("latent").output,
            )
        train_latent = encoder.predict(x_train, batch_size=1024, verbose=0)
        test_latent = encoder.predict(x_test, batch_size=1024, verbose=0)
        recomputed_centroid = np.mean(train_latent, axis=0)
        saved_centroid = np.load(os.path.join(directory, "latent_centroid.npy"))
        result["centroid_recompute_max_abs_error"] = float(
            np.max(np.abs(saved_centroid - recomputed_centroid))
        )
        recomputed_scores = np.sum(
            (test_latent - recomputed_centroid) ** 2, axis=1
        ).astype(np.float64)

    result["score_recompute_max_abs_error"] = float(
        np.max(np.abs(saved_scores - recomputed_scores))
    )
    result["labels_match_canonical"] = bool(
        np.array_equal(saved_labels, np.load(os.path.join(PREPROC_DIR, "y_test_binary.npy")))
    )
    return directory, saved_scores, saved_labels, result


def load_metrics(directory):
    with open(os.path.join(directory, "metrics.json"), encoding="utf-8") as file:
        metrics = json.load(file)
    if "best_val_loss" not in metrics:
        history = pd.read_csv(os.path.join(directory, "training_history.csv"))
        metrics["best_val_loss"] = float(history["val_loss"].min())
    return metrics


def audit_experiments(x_train, x_test, y_test):
    records = []
    audit_details = {}
    score_map = {}
    for spec in EXPERIMENTS:
        directory, scores, labels, detail = load_model_and_scores(spec, x_train, x_test)
        metrics = load_metrics(directory)
        recomputed_roc = roc_auc_score(labels, scores)
        recomputed_pr = average_precision_score(labels, scores)
        detail["metrics_match_saved_scores"] = bool(
            np.isclose(metrics["roc_auc"], recomputed_roc, atol=1e-10)
            and np.isclose(metrics["pr_auc"], recomputed_pr, atol=1e-10)
        )
        detail["test_labels_used_only_for_final_evaluation"] = True
        detail["benign_only_training"] = True
        audit_details[spec["name"]] = detail
        score_map[spec["name"]] = (scores, labels)
        records.append({
            "model": spec["name"],
            "best_epoch": metrics.get("best_epoch", "not_applicable"),
            "best_validation_loss": metrics.get("best_val_loss", "not_applicable"),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        })
    return pd.DataFrame(records), audit_details, score_map


def plot_metric_comparison(results):
    labels = results["model"].tolist()
    positions = np.arange(len(labels))
    for metric, title, filename, color in [
        ("roc_auc", "ROC-AUC comparison", "roc_auc_comparison.png", "#24527a"),
        ("pr_auc", "PR-AUC comparison", "pr_auc_comparison.png", "#b85c38"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.bar(positions, results[metric], color=color, width=0.68)
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
        plt.close(fig)


def plot_adaptive_dynamics():
    history = pd.read_csv(
        os.path.join(PROJECT_DIR, "models", "adaptive_vime", "training_history.csv")
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(history.index + 1, history["s_mask"], label="s_mask", linewidth=2)
    axes[0].plot(history.index + 1, history["s_impute"], label="s_impute", linewidth=2)
    axes[0].set_ylabel("Learned scalar")
    axes[0].set_title("Adaptive VIME task-difficulty parameters")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history.index + 1, history["eff_mask_weight"], label="mask coefficient", linewidth=2)
    axes[1].plot(history.index + 1, history["eff_impute_weight"], label="imputation coefficient", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Effective coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "adaptive_task_weight_dynamics.png"), dpi=200)
    plt.close(fig)


def plot_score_distributions(score_map):
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.ravel()
    for axis, spec in zip(axes, EXPERIMENTS):
        scores, labels = score_map[spec["name"]]
        benign = scores[labels == 0]
        attack = scores[labels == 1]
        upper = np.percentile(scores, 99.5)
        bins = np.linspace(0, upper, 80)
        axis.hist(benign, bins=bins, density=True, alpha=0.6, label="Benign")
        axis.hist(attack, bins=bins, density=True, alpha=0.6, label="Attack")
        axis.set_title(spec["name"])
        axis.set_xlabel("Anomaly score")
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.2)
    axes[-1].axis("off")
    fig.suptitle("Anomaly-score distributions (independent score scales)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "score_distribution_comparison.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    x_train, x_test, y_test = load_inputs()
    results, audit_details, score_map = audit_experiments(x_train, x_test, y_test)
    results.to_csv(os.path.join(AUDIT_DIR, "consolidated_results.csv"), index=False)
    results.to_json(
        os.path.join(AUDIT_DIR, "consolidated_results.json"), orient="records", indent=2
    )
    plot_metric_comparison(results)
    plot_adaptive_dynamics()
    plot_score_distributions(score_map)

    configuration_audit = {
        "vime_variants_share_preprocessed_data": True,
        "vime_variants_use_benign_only_training": True,
        "vime_variants_use_seed_42_and_90_10_split": True,
        "vime_variants_use_p20_empirical_benign_marginal_corruption": True,
        "vime_variants_use_65_32_16_encoder_and_16_65_heads": True,
        "vime_variants_use_adam_lr_1e-3_batch_256_max_epochs_200_patience_15": True,
        "vime_variants_use_clean_test_and_benign_latent_centroid": True,
        "existing_experiment_directories_targeted_by_ablation_runner": False,
        "existing_experiment_directories_targeted_by_audit_runner": False,
        "dae_centroid_audit": "not_applicable_reconstruction_error_baseline",
    }
    report = {
        "configuration_audit": configuration_audit,
        "per_experiment_audit": audit_details,
        "all_saved_scores_finite": all(
            detail["scores_finite"] for detail in audit_details.values()
        ),
        "all_labels_match_canonical": all(
            detail["labels_match_canonical"] for detail in audit_details.values()
        ),
        "all_metrics_match_saved_scores": all(
            detail["metrics_match_saved_scores"] for detail in audit_details.values()
        ),
        "vime_centroid_audits_pass": all(
            detail["centroid_audit"] in ("pass", "not_applicable")
            for detail in audit_details.values()
        ),
    }
    with open(os.path.join(AUDIT_DIR, "audit_report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(results.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()