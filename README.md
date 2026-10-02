# Adaptive Multi-Task Self-Supervised Learning for Network Intrusion Detection

This repository studies self-supervised representation learning for network intrusion and anomaly detection using benign UNSW-NB15 traffic for representation learning. It is based on VIME, with a controlled extension that replaces VIME's manually selected fixed task weighting with learned task-difficulty weighting.

The evaluation also isolates VIME's two pretext tasks: mask prediction and feature imputation. The experiments concern unseen attack-family detection and cross-attack generalization on this benchmark; they do not claim literal real-world zero-day detection.

## Research Questions

- How does VIME-style self-supervised learning compare with a denoising autoencoder on this tabular NIDS setup?
- How much do mask prediction and feature imputation contribute individually?
- Does learned task weighting improve the fixed-weight VIME formulation?

## Method

The VIME variants use a shared encoder `65 -> 32 -> 16`, with a sigmoid mask head and a linear imputation head where applicable. Corruption uses feature-wise empirical marginal replacement with probability `p=0.20`; replacement values come only from benign training traffic.

Adaptive VIME uses:

```text
L = exp(-s_mask) * L_mask
  + 0.5 * exp(-s_impute) * L_impute
  + 0.5 * (s_mask + s_impute)
```

Here `s_mask` and `s_impute` are learned task-difficulty or loss-balancing parameters. They are not interpreted as literal physical aleatoric uncertainty measurements.

All VIME variants use a benign latent centroid for anomaly scoring:

```text
score(x) = ||encoder(x) - mean(encoder(X_train_normal))||^2
```

The DAE baseline instead uses clean reconstruction MSE as its anomaly score.

## Dataset and Preprocessing

The project uses UNSW-NB15. Raw data are intentionally not part of the Git release. See [data/README.md](data/README.md) for acquisition and placement instructions.

The validated preprocessing pipeline separates labels before fitting, fits imputers/scalers/encoders only on benign training rows, uses one-hot encoding with unknown-category handling, removes identifiers, and writes 65-dimensional clean features under `preprocessing/clean/`.

Final arrays:

- `X_train_normal.npy`: `(56000, 65)` benign training traffic
- `X_test.npy`: `(82332, 65)` mixed test traffic
- `y_test_binary.npy`: 37,000 benign and 45,332 attack labels, used only for final evaluation

## Experimental Setup

- Random seed: `42`
- Benign train/validation split: `90/10`
- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `256`
- Maximum epochs: `200`
- Early-stopping patience: `15`
- Test scoring: clean test inputs, with no test-time corruption
- VIME centroid: benign training samples only

## Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Denoising Autoencoder | 0.884313 | 0.902925 |
| Mask-only VIME | 0.883745 | 0.896206 |
| Imputation-only VIME | 0.689168 | 0.713966 |
| Corrected Vanilla VIME | 0.718812 | 0.748780 |
| Adaptive VIME | 0.752678 | 0.753441 |

Mask-only VIME performed very similarly to the denoising autoencoder in this experimental setup. Imputation-only VIME performed substantially worse. Adaptive task weighting improved the fixed-weight VIME formulation on both ROC-AUC and PR-AUC, although it did not reach the performance of the mask-only formulation.

These are controlled observations on UNSW-NB15, not claims of universal superiority or causality beyond the ablation design.

## Ablation Study

The pretext-task ablations are stored separately:

- `models/vime_mask_only/`: mask BCE only
- `models/vime_impute_only/`: imputation MSE only
- `models/vime_baseline_corrected/`: `L_mask + 2.0 * L_impute`
- `models/adaptive_vime/`: learned task weighting

The denoising autoencoder baseline is stored in `models/autoencoder_baseline/`.

## Reproducibility

Create an environment from [requirements.txt](requirements.txt), place the raw CSVs as described in [data/README.md](data/README.md), and run:

```powershell
python src/preprocess_clean.py
python src/train_autoencoder.py
python src/train_vime_corrected.py
python src/train_adaptive_vime.py
python src/train_vime_ablations.py
python src/audit_experiments.py
```

The commands above reproduce the project workflow; the committed model and result artifacts are the validated runs and are not regenerated as part of repository cleanup.

## Repository Structure

```text
src/                    Final preprocessing, training, and audit scripts
data/                   Dataset instructions; raw CSVs are locally ignored
preprocessing/clean/    Validated preprocessing artifacts
models/                 Validated checkpoints and per-experiment artifacts
results/                Consolidated tables, audit report, and figures
docs/                   Additional project documentation
```

## Limitations

UNSW-NB15 contains public historical attack families, so these experiments should not be described as guaranteed zero-day detection. Results depend on the preprocessing, split, corruption process, architecture, and scoring protocol documented here. The adaptive model improves fixed-weight VIME in this setup but does not recover mask-only performance.

## Citation

The baseline method is:

> Jinsung Yoon, Kihyuk Sohn, and Jaewoo Kang. “VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain.” NeurIPS, 2020.

Please cite the original VIME paper when using the baseline methodology. Dataset terms and attribution remain governed by the UNSW-NB15 dataset source.

## License

The code license has not yet been selected. See [LICENSE](LICENSE) before publishing or redistributing this repository.