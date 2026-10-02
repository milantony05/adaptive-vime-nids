# Dataset Instructions

This project uses the UNSW-NB15 network intrusion-detection dataset.

Obtain the dataset from the official UNSW Canberra Cyber dataset distribution or the source specified by your course/research supervisor. This repository does not redistribute the raw CSV files and does not invent a download mirror.

Place the files here with these exact names:

```text
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
```

Then run the leakage-aware preprocessing pipeline:

```powershell
python src/preprocess_clean.py
```

The pipeline writes validated features and preprocessing objects to `preprocessing/clean/`. It separates labels before fitting and fits learned preprocessing statistics only on benign training traffic.

The raw CSV files are ignored by Git because they are large and may have separate dataset distribution terms.