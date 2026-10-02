# Denoising Autoencoder Baseline — UNSW-NB15

## Overview
Unsupervised anomaly detection via reconstruction error.
The autoencoder is trained exclusively on benign network traffic.
At inference time, anomalous (attack) traffic is expected to yield
higher reconstruction MSE than benign traffic.

## Input Data
| File | Shape | Contents |
|---|---|---|
| `preprocessed_clean/X_train_normal.npy` | (56 000, 65) | Benign training features |
| `preprocessed_clean/X_test.npy` | (82 332, 65) | Test features (benign + attack) |
| `preprocessed_clean/y_test_binary.npy` | (82 332,) | Test labels (0=benign, 1=attack) |

**y_test_binary was NOT used during training or model selection.**

## Architecture
```
Input          (65)
MaskingNoise   (p=0.1, training only)
Dense          (32, ReLU)    -- Encoder layer 1
Dense          (16, ReLU)    -- Latent representation
Dense          (32, ReLU)    -- Decoder layer 1
Dense          (65, linear)  -- Reconstruction output
```
Total parameters: 5,329

## Corruption Procedure
Random masking: each feature is independently zeroed with probability
`mask_prob = 0.1` during training (Bernoulli mask).
The decoder target is always the **original clean vector**.
At inference, the masking layer is bypassed — inputs are passed clean.

## Training Configuration
| Parameter | Value |
|---|---|
| Random seed | 42 |
| Corruption probability | 0.1 |
| Batch size | 256 |
| Max epochs | 200 |
| Learning rate | 0.001 (Adam) |
| Loss | MSE |
| Val fraction | 0.1 (from benign train rows) |
| Early stopping patience | 15 |

## Training Results
| Metric | Value |
|---|---|
| Epochs trained | 200 |
| Best epoch | 192 |
| Best val MSE | 0.011246 |
| Final train MSE | 0.022420 |
| Final val MSE | 0.011247 |

## Evaluation Metrics
| Metric | Value |
|---|---|
| ROC-AUC | 0.8843 |
| PR-AUC (Avg Prec.) | 0.9029 |

## Score Statistics
| | Benign | Attack |
|---|---|---|
| Count | 37,000 | 45,332 |
| Mean | 0.0195 | 0.5700 |
| Std | 0.0697 | 17.9703 |
| Median | 0.0062 | 0.0876 |
| P95 | 0.0579 | 0.9043 |

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
All randomness is controlled by `SEED = 42`.
The preprocessing outputs in `preprocessed_clean/` must already exist.
