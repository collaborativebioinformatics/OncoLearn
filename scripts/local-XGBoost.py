"""
NVFlare XGBoost Federated Learning (multi-class) - robust CSV parsing + numeric features + metrics

Key fixes vs your previous version:
1) DO NOT drop first row via iloc[1:,...]. Instead handle header rows correctly via skip_rows.
2) Force all feature columns to numeric (coerce errors to NaN) then fill NaNs with train median.
3) Ensure labels are encoded to contiguous ints 0..K-1 (and warn if sites might mismatch).
4) evaluate_model returns dict; metrics are sent correctly; no dict formatted as float.
5) data_num matches actual rows used.
"""

import argparse
import csv
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nvflare import client as flare
from nvflare.app_opt.xgboost.tree_based.shareable_generator import update_model


# ---------------------------
# Data helpers
# ---------------------------

def load_features(feature_data_path: str) -> list:
    """Load column names from header file (single-line CSV)."""
    try:
        with open(feature_data_path, "r") as file:
            csv_reader = csv.reader(file)
            features = next(csv_reader)
        print(f"Loaded {len(features)} column names from header")
        print(f"First column (ID): {features[0]}")
        print(f"Last column (Label): {features[-1]}")
        return features
    except Exception as e:
        raise Exception(f"Load header for path '{feature_data_path}' failed! {e}")


def load_data(
    data_path: str,
    data_features: list,
    random_state: int,
    test_size: float,
    skip_rows: int = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load CSV and split into train/test DataFrames.

    NOTE:
      - If your data file already has a header row, pass --skip_rows 1 from federate script.
      - We pass names=data_features so the header file defines column names consistently.
    """
    try:
        print(f"Loading data from: {data_path}")
        df: pd.DataFrame = pd.read_csv(
            data_path,
            names=data_features,
            sep=r"\s*,\s*",
            engine="python",
            na_values="?",
            skiprows=skip_rows,
        )

        print(f"Loaded dataframe shape: {df.shape}")
        print(f"Columns (first 5): {df.columns.tolist()[:5]}")

        # Basic sanity checks
        if df.shape[1] < 3:
            raise ValueError("Expected at least 3 columns: ID, >=1 feature, label.")

        # Split
        label_col = df.columns[-1]
        class_counts = df[label_col].value_counts(dropna=False)
        print(f"Class distribution (raw): {class_counts.to_dict()}")

        min_class_count = class_counts.min()
        if min_class_count < 2:
            print(f"WARNING: Minimum class has only {min_class_count} sample(s). Not stratifying.")
            stratify_param = None
        else:
            stratify_param = df[label_col]

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )

        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        print(f"Train label distribution: {train_df[label_col].value_counts().to_dict()}")
        print(f"Test label distribution: {test_df[label_col].value_counts().to_dict()}")

        return {"train": train_df, "test": test_df}

    except Exception as e:
        raise Exception(f"Load data for path '{data_path}' failed! {e}")


def encode_labels_consistently(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode labels to contiguous ints 0..K-1 using mapping derived from TRAIN labels.
    This prevents weird label types and improves XGBoost/metrics stability.

    IMPORTANT (Federated note):
      If different sites have different sets/order of label strings, this can cause mismatch.
      Ideally all sites share the same global label mapping. If label values are already integers 0..K-1
      and consistent across sites, you can skip this step.
    """
    label_col = df_train.columns[-1]

    # Build mapping from train labels (sorted for determinism)
    unique_train = pd.Series(df_train[label_col].unique()).dropna().tolist()
    # Make deterministic ordering:
    try:
        unique_train_sorted = sorted(unique_train)
    except TypeError:
        # Mixed types (e.g. str and int) can't be sorted; fall back to string sort
        unique_train_sorted = sorted(unique_train, key=lambda x: str(x))

    mapping = {v: i for i, v in enumerate(unique_train_sorted)}

    # Apply mapping; unseen labels in test become NaN -> we error out
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[label_col] = df_train[label_col].map(mapping)
    df_test[label_col] = df_test[label_col].map(mapping)

    if df_train[label_col].isna().any():
        bad = df_train[df_train[label_col].isna()]
        raise ValueError(f"Found unmapped labels in TRAIN after mapping. Example rows:\n{bad.head()}")

    if df_test[label_col].isna().any():
        bad = df_test[df_test[label_col].isna()]
        raise ValueError(
            "Found labels in TEST that do not exist in TRAIN for this site. "
            f"Example rows:\n{bad.head()}"
        )

    print(f"Label mapping (train-derived): {mapping}")
    return df_train, df_test


def coerce_features_numeric_and_impute(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert feature columns to numeric (coerce errors -> NaN) and impute NaNs using TRAIN medians.
    This fixes errors like: could not convert string to float: 'SCGB2A2'
    """
    id_col = train_df.columns[0]
    label_col = train_df.columns[-1]
    feat_cols = train_df.columns[1:-1]

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Convert to numeric; strings become NaN
    train_df[feat_cols] = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    test_df[feat_cols] = test_df[feat_cols].apply(pd.to_numeric, errors="coerce")

    # Compute medians on train only
    medians = train_df[feat_cols].median(numeric_only=True)

    # Fill missing
    train_na = int(train_df[feat_cols].isna().sum().sum())
    test_na = int(test_df[feat_cols].isna().sum().sum())
    if train_na > 0 or test_na > 0:
        print(f"Imputing NaNs with TRAIN medians: train_NaNs={train_na}, test_NaNs={test_na}")

    train_df[feat_cols] = train_df[feat_cols].fillna(medians)
    test_df[feat_cols] = test_df[feat_cols].fillna(medians)

    # Final check: still non-numeric?
    # (If some columns are all-NaN in train, median is NaN and fill won't help.)
    remaining_train_na = int(train_df[feat_cols].isna().sum().sum())
    remaining_test_na = int(test_df[feat_cols].isna().sum().sum())
    if remaining_train_na > 0 or remaining_test_na > 0:
        raise ValueError(
            f"After numeric coercion + median impute, NaNs remain. "
            f"train_NaNs={remaining_train_na}, test_NaNs={remaining_test_na}. "
            "This usually means some feature columns are entirely non-numeric or entirely missing in train."
        )

    return train_df, test_df


def df_to_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    """First column: ID (ignored), last column: label, middle: numeric features."""
    x = df.iloc[:, 1:-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return x, y, x.shape[0]


def to_dataset_tuple(data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    return {name: df_to_xy(df) for name, df in data.items()}


def transform_data(data: Dict[str, Tuple[np.ndarray, np.ndarray, int]]) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Standardize features using train stats only."""
    scaler = StandardScaler()
    scaled = {}

    train_x, train_y, train_n = data["train"]
    scaler.fit(train_x)

    for name, (x, y, n) in data.items():
        x_scaled = scaler.transform(x)
        scaled[name] = (x_scaled, y, n)
        print(f"{name}: scaled features {x_scaled.shape}")

    return scaled


# ---------------------------
# Evaluation
# ---------------------------

def evaluate_model(x_test: np.ndarray, model: xgb.Booster, y_test: np.ndarray, n_features: int) -> Dict[str, float]:
    dtest = xgb.DMatrix(x_test, feature_names=[f"f{i}" for i in range(n_features)])
    y_pred = model.predict(dtest)  # (N, K) for multi:softprob

    y_pred_class = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred_class)

    auc_val = None
    try:
        auc_val = roc_auc_score(y_test, y_pred, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"  AUC failed: {e}")

    if auc_val is None:
        print(f"  Accuracy: {acc:.5f}")
    else:
        print(f"  Accuracy: {acc:.5f}, AUC: {auc_val:.5f}")

    return {"accuracy": float(acc), "auc": float(auc_val) if auc_val is not None else None}


# ---------------------------
# Feature importance
# ---------------------------

def save_feature_importance(model, feature_names, site_name, output_dir, curr_round):
    try:
        os.makedirs(output_dir, exist_ok=True)
        importance_types = ["weight", "gain", "cover"]

        all_importance = {}
        for imp_type in importance_types:
            try:
                all_importance[imp_type] = model.get_score(importance_type=imp_type)
                print(f"  Got {len(all_importance[imp_type])} features with {imp_type} importance")
            except Exception as e:
                print(f"  Warning: Could not get {imp_type} importance: {e}")

        if not all_importance:
            print("  Warning: No feature importance scores available yet")
            return None

        rows = []
        for i, fname in enumerate(feature_names):
            fkey = f"f{i}"
            row = {"feature_name": fname, "feature_index": i}
            for t in importance_types:
                row[f"{t}_importance"] = all_importance.get(t, {}).get(fkey, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("gain_importance", ascending=False)
        out = f"{output_dir}/{site_name}_feature_importance_round_{curr_round}.csv"
        df.to_csv(out, index=False)
        print(f"Feature importance saved to: {out}")
        return df
    except Exception as e:
        print(f"Error saving feature importance: {e}")
        return None


def save_final_feature_importance_summary(all_importance_dfs, feature_names, site_name, output_dir):
    try:
        if not all_importance_dfs:
            print("No importance data to summarize")
            return None

        # Aggregate mean/std/max over rounds for gain
        summary_rows = []
        for i, fname in enumerate(feature_names):
            gains = []
            weights = []
            covers = []
            for df in all_importance_dfs:
                r = df[df["feature_name"] == fname]
                if not r.empty:
                    gains.append(float(r["gain_importance"].values[0]))
                    weights.append(float(r["weight_importance"].values[0]))
                    covers.append(float(r["cover_importance"].values[0]))

            if gains:
                summary_rows.append(
                    {
                        "feature_name": fname,
                        "feature_index": i,
                        "avg_gain_importance": float(np.mean(gains)),
                        "std_gain_importance": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
                        "max_gain_importance": float(np.max(gains)),
                        "avg_weight_importance": float(np.mean(weights)) if weights else 0.0,
                        "avg_cover_importance": float(np.mean(covers)) if covers else 0.0,
                    }
                )
            else:
                summary_rows.append(
                    {
                        "feature_name": fname,
                        "feature_index": i,
                        "avg_gain_importance": 0.0,
                        "std_gain_importance": 0.0,
                        "max_gain_importance": 0.0,
                        "avg_weight_importance": 0.0,
                        "avg_cover_importance": 0.0,
                    }
                )

        summary_df = pd.DataFrame(summary_rows).sort_values("avg_gain_importance", ascending=False)
        out = f"{output_dir}/{site_name}_feature_importance_FINAL_SUMMARY.csv"
        summary_df.to_csv(out, index=False)
        print(f"Final feature importance summary saved to: {out}")
        return summary_df
    except Exception as e:
        print(f"Error creating feature importance summary: {e}")
        return None


# ---------------------------
# Args
# ---------------------------

def define_args_parser():
    parser = argparse.ArgumentParser(description="XGBoost Federated Learning with NVFLARE (multi-class)")
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/ec2-user/nvflare/outputs")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--num_client_bagging", type=int, default=2)
    parser.add_argument(
        "--skip_rows",
        type=int,
        default=None,
        help="Rows to skip at beginning of data file. If the CSV already has a header row, set to 1.",
    )
    parser.add_argument(
        "--encode_labels",
        action="store_true",
        help="Encode labels to 0..K-1 based on train labels. Use only if labels are not already ints.",
    )
    return parser


# ---------------------------
# Main
# ---------------------------

def main():
    parser = define_args_parser()
    args = parser.parse_args()

    data_root_dir = args.data_root_dir
    output_dir = args.output_dir
    random_state = args.random_state
    test_size = args.test_size
    num_client_bagging = args.num_client_bagging
    skip_rows = args.skip_rows

    print("=" * 80)
    print("NVFLARE XGBoost Federated Learning - Training Script")
    print("=" * 80)
    print(f"Data root: {data_root_dir}")
    print(f"Output dir: {output_dir}")
    print(f"random_state: {random_state}, test_size: {test_size}")
    print(f"num_client_bagging: {num_client_bagging}, skip_rows: {skip_rows}")
    print("=" * 80)

    flare.init()
    site_name = flare.get_site_name()
    print(f"Starting training for site: {site_name}")

    feature_data_path = f"{data_root_dir}/{site_name}_header.csv"
    all_features = load_features(feature_data_path)

    feature_names = all_features[1:-1]
    n_features = len(feature_names)
    print(f"n_features = {n_features}")

    data_path = f"{data_root_dir}/{site_name}.csv"
    data = load_data(
        data_path=data_path,
        data_features=all_features,
        random_state=random_state,
        test_size=test_size,
        skip_rows=skip_rows,
    )

    # Clean features (numeric coercion + impute) BEFORE converting to numpy
    train_df, test_df = data["train"], data["test"]
    train_df, test_df = coerce_features_numeric_and_impute(train_df, test_df)

    # Optional label encoding (use if labels are strings or not 0..K-1)
    if args.encode_labels:
        train_df, test_df = encode_labels_consistently(train_df, test_df)

    data = {"train": train_df, "test": test_df}

    # Convert to numpy tuples and scale
    data_tuples = to_dataset_tuple(data)
    dataset = transform_data(data_tuples)

    x_train, y_train, _ = dataset["train"]
    x_test, y_test, _ = dataset["test"]

    # Ensure integer labels for XGBoost
    if not np.issubdtype(y_train.dtype, np.integer):
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    num_class = int(len(np.unique(y_train)))
    print(f"Detected num_class = {num_class}")

    dmat_train = xgb.DMatrix(x_train, label=y_train, feature_names=[f"f{i}" for i in range(n_features)])
    dmat_test = xgb.DMatrix(x_test, label=y_test, feature_names=[f"f{i}" for i in range(n_features)])

    xgb_params = {
        "eta": 0.1 / num_client_bagging,
        "objective": "multi:softprob",
        "num_class": num_class,
        "max_depth": 8,
        "eval_metric": "mlogloss",
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1.0,
        "tree_method": "hist",
    }

    print("XGBoost params:")
    for k, v in xgb_params.items():
        print(f"  {k}: {v}")

    global_model_as_dict = None
    config = None
    all_importance_dfs = []

    print("=" * 80)
    print("Starting federated learning loop...")
    print("=" * 80)

    while flare.is_running():
        input_model = flare.receive()
        global_params = input_model.params
        curr_round = input_model.current_round

        print("\n" + ("=" * 80))
        print(f"Round {curr_round} - Site: {site_name}")
        print("=" * 80)

        if curr_round == 0:
            print("Training initial local model...")
            model = xgb.train(
                xgb_params,
                dmat_train,
                num_boost_round=1,
                evals=[(dmat_train, "train"), (dmat_test, "test")],
            )
            config = model.save_config()

        else:
            print("Updating model with global parameters...")
            model_updates = global_params["model_data"]
            for upd in model_updates:
                # if update_model doesn't like None, initialize:
                if global_model_as_dict is None:
                    global_model_as_dict = {}
                global_model_as_dict = update_model(global_model_as_dict, json.loads(upd))

            loadable_model = bytearray(json.dumps(global_model_as_dict), "utf-8")
            model.load_model(loadable_model)
            if config is not None:
                model.load_config(config)

            # Eval before local update
            print(model.eval_set(evals=[(dmat_train, "train"), (dmat_test, "test")],
                                 iteration=max(model.num_boosted_rounds() - 1, 0)))

            print("Training local update (one more boosting round)...")
            model.update(dmat_train, model.num_boosted_rounds())

        # Evaluate
        eval_metrics = evaluate_model(x_test, model, y_test, n_features)

        # Save feature importance
        print(f"Saving feature importance for round {curr_round}...")
        imp_df = save_feature_importance(model, feature_names, site_name, output_dir, curr_round)
        if imp_df is not None:
            all_importance_dfs.append(imp_df)

        # Send newly added tree
        bst_new = model[model.num_boosted_rounds() - 1: model.num_boosted_rounds()]
        local_model_update = bst_new.save_raw("json")
        params = {"model_data": local_model_update}

        output_model = flare.FLModel(params=params, metrics=eval_metrics)

        auc_str = f"{eval_metrics['auc']:.5f}" if eval_metrics.get("auc") is not None else "NA"
        print(f"{site_name}: Sending local update back (acc={eval_metrics['accuracy']:.5f}, auc={auc_str})")
        flare.send(output_model)

    print("=" * 80)
    print("Training completed! Generating final feature importance summary...")
    print("=" * 80)

    save_final_feature_importance_summary(all_importance_dfs, feature_names, site_name, output_dir)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
