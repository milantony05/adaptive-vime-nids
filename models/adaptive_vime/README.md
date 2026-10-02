# Adaptive VIME Baseline — UNSW-NB15

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
final_s_mask: 0.0030
final_s_impute: -2.4051

## 10. Primary anomaly scoring method
Latent distance to benign centroid: ||e(x) - c||^2.

## 11. Evaluation metrics
ROC-AUC and Average Precision (PR-AUC).

## 12. Comparison
| Metric | Adaptive VIME | Vanilla VIME (Corrected) | DAE |
|---|---|---|---|
| ROC-AUC | 0.7527 | 0.7188 | 0.8843 |
| PR-AUC | 0.7534 | 0.7488 | 0.9029 |

## 13. Limitations
The learned scalars are described as learned task-difficulty or loss-balancing
parameters, not direct physical aleatoric uncertainty measurements.
