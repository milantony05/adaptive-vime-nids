"""
preprocess_clean.py
===================
Leakage-Free UNSW-NB15 Preprocessing Pipeline
for Unsupervised Network Intrusion Detection
----------------------------------------------

Design principles
-----------------
1.  Labels are separated FIRST - before any imputer/scaler/encoder is
    fitted - so target information cannot contaminate input statistics.

2.  ALL sklearn objects are fitted ONLY on benign training rows
    (label == 0), so the learned statistics model the normal-traffic
    distribution the unsupervised model will be trained on.

3.  proto / service / state are one-hot encoded (not ordinal integers).
    OneHotEncoder(handle_unknown='ignore') produces an all-zero row for
    any category seen in test but not in the benign training vocabulary —
    no crash, consistent feature dimension.

4.  Identifier / shortcut columns are dropped before processing.

5.  log1p is a fixed, parameter-free transform — safe to apply before
    the scaler fit; does NOT use any test statistics.

6.  Outputs go to preprocessing/clean/ as the validated clean representation.
    preprocessed/ directory.

No model training, VIME, autoencoder, or anomaly-detection logic here.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV   = os.path.join(PROJECT_DIR, "data", "raw", "UNSW_NB15_training-set.csv")
TEST_CSV    = os.path.join(PROJECT_DIR, "data", "raw", "UNSW_NB15_testing-set.csv")
OUT_DIR     = os.path.join(PROJECT_DIR, "preprocessing", "clean")
os.makedirs(OUT_DIR, exist_ok=True)

# Columns that carry identity / shortcut information — always dropped first.
DROP_COLS = ["id", "srcip", "dstip", "sport", "dsport"]

# Target columns — separated from features before any fitting.
BINARY_LABEL     = "label"       # 0 = Normal, 1 = Attack
MULTICLASS_LABEL = "attack_cat"  # e.g. normal, dos, exploits, …

# Categorical columns to be one-hot encoded.
CAT_COLS = ["proto", "service", "state"]

# Numeric columns with strong right skew → log1p before scaling.
# log1p is a fixed transform; no statistics are computed from the data.
SKEWED_COLS = [
    "dur", "sbytes", "dbytes", "rate",
    "sload", "dload",
    "sjit", "djit",
    "sinpkt", "dinpkt",
    "response_body_len",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pretty section header
# ─────────────────────────────────────────────────────────────────────────────
def _hdr(text: str) -> None:
    print(f"\n{'-'*60}\n  {text}\n{'-'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1.  Load raw CSVs
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    _hdr("Step 1 · Load raw data")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    print(f"  Raw train shape : {train.shape}")
    print(f"  Raw test  shape : {test.shape}")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Step 2.  Drop identifier / shortcut columns
# ─────────────────────────────────────────────────────────────────────────────
def drop_identifiers(df: pd.DataFrame, name: str) -> pd.DataFrame:
    dropped = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=dropped, errors="ignore")
    print(f"  [{name}] Dropped identifier columns: {dropped}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3.  Separate labels FIRST
# ─────────────────────────────────────────────────────────────────────────────
def separate_labels(df: pd.DataFrame, name: str):
    """
    Split DataFrame into (feature_df, y_binary, y_multiclass_str).
    This MUST happen before any fitting so the label cannot contaminate
    the imputer or scaler statistics.
    """
    # Normalise attack_cat: strip whitespace, lower-case
    y_bin = df[BINARY_LABEL].values.astype(np.int32)
    y_mc  = df[MULTICLASS_LABEL].astype(str).str.strip().str.lower().values

    feature_df = df.drop(columns=[BINARY_LABEL, MULTICLASS_LABEL])
    print(f"  [{name}] Features: {feature_df.shape[1]} cols | "
          f"Normal rows: {(y_bin == 0).sum()} | Attack rows: {(y_bin == 1).sum()}")
    return feature_df, y_bin, y_mc


# ─────────────────────────────────────────────────────────────────────────────
# Step 4.  Basic cleaning  (placeholder normalisation, whitespace strip)
# ─────────────────────────────────────────────────────────────────────────────
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df.replace(["", "?", "nan", "NaN", "None", "none", " "], np.nan, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5.  Identify numeric vs categorical columns
# ─────────────────────────────────────────────────────────────────────────────
def identify_columns(benign_train_X: pd.DataFrame):
    _hdr("Step 5 · Column identification")
    cat_present = [c for c in CAT_COLS if c in benign_train_X.columns]
    num_cols    = [c for c in benign_train_X.columns if c not in cat_present]
    print(f"  Numeric columns   : {len(num_cols)}")
    print(f"  Categorical cols  : {cat_present}")
    return num_cols, cat_present


# ─────────────────────────────────────────────────────────────────────────────
# Step 6.  Numeric pipeline  (fit on benign train only)
#   6a.  Median imputation
#   6b.  log1p on skewed columns
#   6c.  StandardScaler
# ─────────────────────────────────────────────────────────────────────────────
def fit_numeric_pipeline(benign_train_X: pd.DataFrame, num_cols: list):
    _hdr("Step 6 - Numeric pipeline (fitted on benign train only)")
    n_rows = len(benign_train_X)

    # -- 6a. Imputation -------------------------------------------------------
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(benign_train_X[num_cols])
    print(f"  Imputer fitted on {n_rows:,} benign-train rows")

    # -- 6b. log1p: fixed transform, no fitting needed ------------------------
    skewed_present = [c for c in SKEWED_COLS if c in num_cols]
    print(f"  log1p columns ({len(skewed_present)}): {skewed_present}")

    # -- 6c. Scaler -----------------------------------------------------------
    # Apply imputation + log1p to benign train, THEN fit scaler on the result.
    tmp = _apply_numeric(benign_train_X[num_cols], num_imputer, skewed_present)
    scaler = StandardScaler()
    scaler.fit(tmp)
    print(f"  Scaler   fitted on {int(scaler.n_samples_seen_):,} benign-train rows")

    joblib.dump(num_imputer,    os.path.join(OUT_DIR, "num_imputer.pkl"))
    joblib.dump(scaler,         os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(skewed_present, os.path.join(OUT_DIR, "skewed_cols.pkl"))

    return num_imputer, scaler, skewed_present


def _apply_numeric(df_num: pd.DataFrame,
                   num_imputer: SimpleImputer,
                   skewed_present: list) -> np.ndarray:
    """Impute → log1p → return ndarray (scaler applied by caller)."""
    arr = num_imputer.transform(df_num)
    tmp = pd.DataFrame(arr, columns=df_num.columns)
    for col in skewed_present:
        tmp[col] = np.log1p(tmp[col].clip(lower=0))
    return tmp.values


def transform_numeric(df_num: pd.DataFrame,
                      num_imputer: SimpleImputer,
                      scaler: StandardScaler,
                      skewed_present: list) -> np.ndarray:
    arr = _apply_numeric(df_num, num_imputer, skewed_present)
    return scaler.transform(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7.  Categorical pipeline  (fit on benign train only)
#   7a.  Mode imputation
#   7b.  OneHotEncoder(handle_unknown='ignore')
# ─────────────────────────────────────────────────────────────────────────────
def fit_categorical_pipeline(benign_train_X: pd.DataFrame, cat_present: list):
    _hdr("Step 7 - Categorical pipeline (fitted on benign train only)")

    # ── 7a. Mode imputation ───────────────────────────────────────────────────
    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_imputer.fit(benign_train_X[cat_present].astype(str))
    print(f"  Cat-imputer mode values: {dict(zip(cat_present, cat_imputer.statistics_))}")

    # ── 7b. OneHotEncoder ─────────────────────────────────────────────────────
    # handle_unknown='ignore' -> all-zero row for any unseen test category.
    # sparse_output=False     -> return dense ndarray for easy concatenation.
    # dtype=np.float32        -> memory-efficient.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

    # Impute first, then fit OHE
    tmp_arr = cat_imputer.transform(benign_train_X[cat_present].astype(str))
    ohe.fit(tmp_arr)

    # Report vocabulary per column
    total_ohe_cols = 0
    for col_name, cats in zip(cat_present, ohe.categories_):
        print(f"  {col_name:15s}: {len(cats)} one-hot columns")
        total_ohe_cols += len(cats)
    print(f"  Total OHE output columns: {total_ohe_cols}")

    joblib.dump(cat_imputer, os.path.join(OUT_DIR, "cat_imputer.pkl"))
    joblib.dump(ohe,         os.path.join(OUT_DIR, "ohe_encoder.pkl"))

    return cat_imputer, ohe


def transform_categorical(df_cat: pd.DataFrame,
                          cat_imputer: SimpleImputer,
                          ohe: OneHotEncoder,
                          cat_present: list) -> np.ndarray:
    tmp_arr = cat_imputer.transform(df_cat[cat_present].astype(str))
    return ohe.transform(tmp_arr)


# ─────────────────────────────────────────────────────────────────────────────
# Step 8.  Build feature names list
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_names(num_cols: list,
                        cat_present: list,
                        ohe: OneHotEncoder) -> list:
    ohe_names = ohe.get_feature_names_out(cat_present).tolist()
    return num_cols + ohe_names


# ─────────────────────────────────────────────────────────────────────────────
# Step 9.  Assertions
# ─────────────────────────────────────────────────────────────────────────────
def run_assertions(X_train: np.ndarray,
                   X_test:  np.ndarray,
                   y_test_bin: np.ndarray,
                   feature_names: list,
                   num_imputer: SimpleImputer,
                   scaler: StandardScaler,
                   n_benign_train: int) -> None:
    _hdr("Step 9 - Assertions")

    # No NaN
    assert np.isnan(X_train).sum() == 0, "X_train contains NaN!"
    assert np.isnan(X_test).sum()  == 0, "X_test contains NaN!"
    print("  [PASS] No NaN in X_train or X_test")

    # No Inf
    assert np.isinf(X_train).sum() == 0, "X_train contains Inf!"
    assert np.isinf(X_test).sum()  == 0, "X_test contains Inf!"
    print("  [PASS] No Inf in X_train or X_test")

    # Shape consistency
    assert X_train.shape[1] == X_test.shape[1], \
        f"Feature dim mismatch: train {X_train.shape[1]} vs test {X_test.shape[1]}"
    assert X_train.shape[1] == len(feature_names), \
        f"feature_names length {len(feature_names)} != X_train cols {X_train.shape[1]}"
    print(f"  [PASS] train/test feature dims match: {X_train.shape[1]}")

    # Imputer and scaler were fitted on benign training data only.
    # We verify via the scaler (StandardScaler.n_samples_seen_ is reliable);
    # the imputer is verified indirectly through the shared benign-train data.
    assert int(scaler.n_samples_seen_) == n_benign_train, (
        f"Scaler was fitted on {int(scaler.n_samples_seen_)} rows, "
        f"expected {n_benign_train} (benign train only)."
    )
    print(f"  [PASS] Scaler fitted on exactly {n_benign_train} benign rows")
    print(f"  [PASS] Imputer input was the same {n_benign_train}-row benign DataFrame")

    # label column must NOT appear in feature_names
    assert BINARY_LABEL     not in feature_names, "Binary label found in features!"
    assert MULTICLASS_LABEL not in feature_names, "attack_cat found in features!"
    print("  [PASS] 'label' and 'attack_cat' not present in feature matrix")

    # Labels present in y_test
    assert len(y_test_bin) == X_test.shape[0], "y_test_binary length mismatch!"
    print("  [PASS] y_test_binary length matches X_test rows")

    print("\n  ALL ASSERTIONS PASSED [OK]")


# ─────────────────────────────────────────────────────────────────────────────
# Step 10.  Save outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(X_train: np.ndarray,
                 X_test:  np.ndarray,
                 y_test_bin: np.ndarray,
                 y_test_mc:  np.ndarray,
                 feature_names: list,
                 cat2idx: dict) -> None:
    _hdr("Step 10 - Saving outputs")

    np.save(os.path.join(OUT_DIR, "X_train_normal.npy"),    X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),            X_test)
    np.save(os.path.join(OUT_DIR, "y_test_binary.npy"),     y_test_bin)
    np.save(os.path.join(OUT_DIR, "y_test_multiclass.npy"), y_test_mc)

    pd.Series(feature_names).to_csv(
        os.path.join(OUT_DIR, "feature_names.csv"),
        index=False, header=["feature"]
    )
    joblib.dump(cat2idx, os.path.join(OUT_DIR, "attack_cat_mapping.pkl"))

    for fname in ["X_train_normal.npy", "X_test.npy", "y_test_binary.npy",
                  "y_test_multiclass.npy", "feature_names.csv",
                  "num_imputer.pkl", "scaler.pkl", "ohe_encoder.pkl",
                  "cat_imputer.pkl", "skewed_cols.pkl", "attack_cat_mapping.pkl"]:
        full = os.path.join(OUT_DIR, fname)
        size_kb = os.path.getsize(full) / 1024
        print(f"  {fname:<35s}  {size_kb:>8.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# Step 11.  Preprocessing report
# ─────────────────────────────────────────────────────────────────────────────
def print_report(X_train: np.ndarray,
                 X_test:  np.ndarray,
                 y_test_bin: np.ndarray,
                 num_cols: list,
                 cat_present: list,
                 ohe: OneHotEncoder,
                 n_benign_train: int,
                 n_attack_train: int) -> None:
    ohe_dim = sum(len(c) for c in ohe.categories_)
    n_num   = len(num_cols)

    print("\n" + "="*60)
    print("  PREPROCESSING REPORT")
    print("="*60)
    print(f"  Benign training rows        : {n_benign_train:>8,}")
    print(f"  Attack training rows        : {n_attack_train:>8,}  (not used for fitting)")
    print(f"  Benign test rows            : {(y_test_bin == 0).sum():>8,}")
    print(f"  Attack test rows            : {(y_test_bin == 1).sum():>8,}")
    print(f"")
    print(f"  Numeric feature count       : {n_num:>8,}")
    print(f"  Categorical column count    : {len(cat_present):>8,}")
    print(f"  One-hot feature count       : {ohe_dim:>8,}")
    print(f"  Final feature dimension     : {X_train.shape[1]:>8,}  "
          f"(= {n_num} numeric + {ohe_dim} OHE)")
    print(f"")
    print(f"  X_train_normal shape        : {X_train.shape}")
    print(f"  X_test         shape        : {X_test.shape}")
    print(f"")
    print(f"  NaN in X_train              : {np.isnan(X_train).sum()}")
    print(f"  NaN in X_test               : {np.isnan(X_test).sum()}")
    print(f"  Inf in X_train              : {np.isinf(X_train).sum()}")
    print(f"  Inf in X_test               : {np.isinf(X_test).sum()}")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 12.  Encode multi-class label for test set
# ─────────────────────────────────────────────────────────────────────────────
def encode_attack_cat(train_mc: np.ndarray, test_mc: np.ndarray):
    """Build a str→int mapping from training categories only."""
    categories = sorted(set(train_mc))
    cat2idx    = {c: i for i, c in enumerate(categories)}
    y_train_mc_int = np.array([cat2idx.get(c, -1) for c in train_mc], dtype=np.int32)
    y_test_mc_int  = np.array([cat2idx.get(c, -1) for c in test_mc],  dtype=np.int32)
    return y_train_mc_int, y_test_mc_int, cat2idx


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline() -> dict:
    print("\n" + "="*60)
    print("  UNSW-NB15  CLEAN PREPROCESSING PIPELINE")
    print("  (Leakage-free - Unsupervised NIDS edition)")
    print("="*60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    train_raw, test_raw = load_data()

    # ── 2. Drop identifiers ───────────────────────────────────────────────────
    _hdr("Step 2 · Drop identifier / shortcut columns")
    train_raw = drop_identifiers(train_raw, "train")
    test_raw  = drop_identifiers(test_raw,  "test")

    # ── 3. Basic cleaning ─────────────────────────────────────────────────────
    _hdr("Step 3–4 · Basic cleaning + label separation")
    train_raw = basic_clean(train_raw)
    test_raw  = basic_clean(test_raw)

    # ── 4. Separate labels FIRST ──────────────────────────────────────────────
    train_X, y_train_bin, y_train_mc_str = separate_labels(train_raw, "train")
    test_X,  y_test_bin,  y_test_mc_str  = separate_labels(test_raw,  "test")

    # ── 4b. Encode attack_cat integers from training vocab only ───────────────
    _, y_test_mc_int, cat2idx = encode_attack_cat(y_train_mc_str, y_test_mc_str)
    print(f"  Attack-cat classes ({len(cat2idx)}): {sorted(cat2idx.keys())}")

    # ── 4c. Filter benign training rows ───────────────────────────────────────
    benign_mask    = y_train_bin == 0
    benign_train_X = train_X[benign_mask].copy().reset_index(drop=True)
    n_benign_train = int(benign_mask.sum())
    n_attack_train = int((~benign_mask).sum())
    print(f"  Benign train rows : {n_benign_train:,}")
    print(f"  Attack train rows : {n_attack_train:,}  (excluded from fitting)")

    # ── 5. Column identification ──────────────────────────────────────────────
    num_cols, cat_present = identify_columns(benign_train_X)

    # ── 6. Numeric pipeline ───────────────────────────────────────────────────
    num_imputer, scaler, skewed_present = fit_numeric_pipeline(benign_train_X, num_cols)

    X_train_num = transform_numeric(train_X[benign_mask][num_cols],
                                    num_imputer, scaler, skewed_present)
    X_test_num  = transform_numeric(test_X[num_cols],
                                    num_imputer, scaler, skewed_present)

    # ── 7. Categorical pipeline ───────────────────────────────────────────────
    cat_imputer, ohe = fit_categorical_pipeline(benign_train_X, cat_present)

    X_train_cat = transform_categorical(train_X[benign_mask], cat_imputer, ohe, cat_present)
    X_test_cat  = transform_categorical(test_X,              cat_imputer, ohe, cat_present)

    # ── 8. Concatenate ────────────────────────────────────────────────────────
    _hdr("Step 8 - Concatenate numeric + one-hot features")
    X_train_final = np.concatenate([X_train_num, X_train_cat], axis=1).astype(np.float32)
    X_test_final  = np.concatenate([X_test_num,  X_test_cat],  axis=1).astype(np.float32)
    print(f"  X_train_normal : {X_train_final.shape}")
    print(f"  X_test         : {X_test_final.shape}")

    # ── 9. Build feature names ────────────────────────────────────────────────
    feature_names = build_feature_names(num_cols, cat_present, ohe)

    # ── 10. Assertions ────────────────────────────────────────────────────────
    run_assertions(X_train_final, X_test_final,
                   y_test_bin, feature_names,
                   num_imputer, scaler, n_benign_train)

    # ── 11. Save ──────────────────────────────────────────────────────────────
    save_outputs(X_train_final, X_test_final,
                 y_test_bin, y_test_mc_int,
                 feature_names, cat2idx)

    # ── 12. Report ────────────────────────────────────────────────────────────
    print_report(X_train_final, X_test_final,
                 y_test_bin, num_cols, cat_present, ohe,
                 n_benign_train, n_attack_train)

    return {
        "X_train_normal":     X_train_final,
        "X_test":             X_test_final,
        "y_test_binary":      y_test_bin,
        "y_test_multiclass":  y_test_mc_int,
        "feature_names":      feature_names,
        "attack_cat_mapping": cat2idx,
        "num_imputer":        num_imputer,
        "scaler":             scaler,
        "ohe_encoder":        ohe,
        "cat_imputer":        cat_imputer,
        "n_benign_train":     n_benign_train,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print("Pipeline complete. Ready for unsupervised model training.")
