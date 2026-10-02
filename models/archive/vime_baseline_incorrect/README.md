# Vanilla VIME Baseline — UNSW-NB15

## Overview
Vanilla VIME self-supervised baseline.
The model is trained to reconstruct corrupted features and estimate which features were corrupted.
Anomaly scoring uses the distance to the benign latent centroid.

## Input Data
| File | Shape | Contents |
|---|---|---|
| `preprocessed_clean/X_train_normal.npy` | (56 000, 65) | Benign training features |
| `preprocessed_clean/X_test.npy` | (82 332, 65) | Test features (benign + attack) |
| `preprocessed_clean/y_test_binary.npy` | (82 332,) | Test labels (0=benign, 1=attack) |

## Architecture
Encoder: 65 -> 32 -> 16
Impute Head: 16 -> 65
Mask Head: 16 -> 65

## Corruption Mechanism
Each feature is independently corrupted with probability p=0.2.
Corrupted values are sampled from the empirical marginal distribution of the benign training set.

## Loss
Total Loss = Impute_MSE + 2.0 * Mask_BCE

## Anomaly Scoring
Distance to the benign latent centroid: ||e(x) - c||^2

## Evaluation Metrics
| Metric | Value | Baseline (AE) |
|---|---|---|
| ROC-AUC | 0.7305 | 0.8843 |
| PR-AUC | 0.7558 | 0.9029 |
