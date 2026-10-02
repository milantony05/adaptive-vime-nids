# Corrected Vanilla VIME Baseline — UNSW-NB15

## Overview
Vanilla VIME baseline, corrected to use the original VIME loss weighting:
L_total = L_mask + alpha * L_impute, alpha = 2.0.

## Input Data
| File | Shape | Contents |
|---|---|---|
| `preprocessed_clean/X_train_normal.npy` | (56 000, 65) | Benign training features |
| `preprocessed_clean/X_test.npy` | (82 332, 65) | Test features (benign + attack) |
| `preprocessed_clean/y_test_binary.npy` | (82 332,) | Test labels (0=benign, 1=attack) |

## Architecture
- The encoder architecture is an adaptation for UNSW-NB15 (65 -> 32 -> 16).
- The imputation head is linear because the input representation is standardized and can contain negative values.
- Mask head uses sigmoid activation.

## Corruption Mechanism
- Corruption uses empirical marginal replacement.
- Mask probability p = 0.2.
- Corrupted values are sampled from the empirical marginal distribution of the benign training set.

## Loss
Total Loss = Mask_BCE + 2.0 * Impute_MSE

## Anomaly Scoring
The primary anomaly score is latent distance to the benign centroid: ||e(x) - c||^2

## Evaluation Metrics
| Metric | Value | DAE Baseline | Previous VIME (Incorrect) |
|---|---|---|---|
| ROC-AUC | 0.7188 | 0.8843 | 0.7305 |
| PR-AUC | 0.7488 | 0.9029 | 0.7558 |
