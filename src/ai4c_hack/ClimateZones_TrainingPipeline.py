# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
"""Train and evaluate climate-zone classifiers from ML-ready CSV data.

This module exposes a command-line interface and separates the workflow into:
- configuration and argument parsing
- data loading and preprocessing
- model training
- model evaluation
"""

# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
from typing import Any

import pandas as pd
from sklearn import ensemble, metrics, neural_network, preprocessing, tree

ALLOWED_RESOLUTIONS = (0.1, 0.5, 1.0)
DEFAULT_TARGET_COLUMN = "climate_group"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train climate-zone classifiers.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("config.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        choices=ALLOWED_RESOLUTIONS,
        default=1.0,
        help="Input data resolution to use.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Optional platform override. If omitted, value is read from config.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=DEFAULT_TARGET_COLUMN,
        help="Target label column.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of rows reserved for test split.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of total rows reserved for validation split.",
    )
    return parser.parse_args()


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    """Load and return the tutorial config JSON."""
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def get_platform_dir(selected_platform: str, config: dict[str, Any]) -> pathlib.Path:
    """Resolve base data directory for a selected platform.

    Falls back to ``$HOME/climate_zones`` if the platform is not found.
    """
    try:
        return pathlib.Path(config["default_dirs"][selected_platform]) / "climate_zones"
    except KeyError:
        return pathlib.Path.home() / "climate_zones"


def get_ml_ready_path(root_data_dir: pathlib.Path, config: dict[str, Any], resolution: float) -> pathlib.Path:
    """Build the ML-ready CSV path for the selected resolution."""
    resolutions_dict = {float(key): value for key, value in config["resolutions_names"].items()}
    resolution_name = resolutions_dict[resolution]
    csv_template = config["csv_out_template"]
    return root_data_dir / "ml_ready" / csv_template.format(resolution=resolution_name)


def load_data(data_path: pathlib.Path) -> pd.DataFrame:
    """Load CSV input data and apply the same memory-saving filter as the original script."""
    print(f"Loading data from: {data_path}")
    zones_df = pd.read_csv(data_path)

    # Keep the historical 1991 period subset to limit memory usage.
    zones_df = zones_df[(zones_df["period_start"] == 1991) & (zones_df["scenario"] == "historic")]
    print(f"Data loaded successfully: {zones_df.shape}")
    return zones_df


def build_predictors(zones_df: pd.DataFrame) -> list[str]:
    """Build predictor columns used for training."""
    precip_mean = [col for col in zones_df.columns if "precipitation" in col and "mean" in col]
    temp_mean = [col for col in zones_df.columns if "air_temperature" in col and "mean" in col]
    predictors = precip_mean + temp_mean
    if not predictors:
        raise ValueError("No predictor columns found. Check input schema.")
    return predictors


def split_data(
    zones_df: pd.DataFrame,
    random_seed: int,
    test_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits using grouped sampling."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1.")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if test_frac + val_frac >= 1.0:
        raise ValueError("test_frac + val_frac must be less than 1.")

    group_cols = ["period_start", "scenario"]
    val_frac_sub = val_frac / (1.0 - test_frac)

    test_df = zones_df.groupby(group_cols).sample(frac=test_frac, random_state=random_seed)
    remain_df = zones_df.drop(test_df.index)

    val_df = remain_df.groupby(group_cols).sample(frac=val_frac_sub, random_state=random_seed)
    train_df = remain_df.drop(val_df.index)

    print(f"Train split: {train_df.shape}")
    print(f"Validation split: {val_df.shape}")
    print(f"Test split: {test_df.shape}")
    return train_df, val_df, test_df


def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list[str],
    target_column: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Scale input features and encode target labels."""
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_df[predictors])

    x_train = scaler.transform(train_df[predictors])
    x_val = scaler.transform(val_df[predictors])
    x_test = scaler.transform(test_df[predictors])

    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[target_column])
    y_val = label_encoder.transform(val_df[target_column])
    y_test = label_encoder.transform(test_df[target_column])

    return x_train, x_val, x_test, y_train, y_val, y_test


def train_models(x_train: Any, y_train: Any, random_seed: int) -> dict[str, Any]:
    """Train all configured classifiers and return them by name."""
    classifiers_params = {
        "decision_tree": {
            "class": tree.DecisionTreeClassifier,
            "opts": {"max_depth": 10, "class_weight": "balanced", "random_state": random_seed},
        },
        "random_forest": {
            "class": ensemble.RandomForestClassifier,
            "opts": {
                "max_depth": 10,
                "class_weight": "balanced",
                "n_estimators": 10,
                "random_state": random_seed,
            },
        },
        "ann_3_100": {
            "class": neural_network.MLPClassifier,
            "opts": {"hidden_layer_sizes": (100, 100, 100), "random_state": random_seed},
        },
    }

    print("Starting training")
    models: dict[str, Any] = {}
    for name, params in classifiers_params.items():
        print(f"Training algorithm: {name}")
        train_start = datetime.datetime.now()
        model = params["class"](**params["opts"])
        model.fit(x_train, y_train)
        models[name] = model
        print(f"Training time ({name}): {datetime.datetime.now() - train_start}")

    print("Training complete")
    return models


def evaluate_models(models: dict[str, Any], split_data_dict: dict[str, tuple[Any, Any]]) -> dict[str, dict[str, float]]:
    """Evaluate trained models on each split and return weighted metrics."""
    print("Starting evaluation")
    results: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        model_results: dict[str, float] = {}
        for split_name, (x_data, y_true) in split_data_dict.items():
            y_pred = model.predict(x_data)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
            accuracy = metrics.accuracy_score(y_true, y_pred)

            model_results[f"{split_name}_accuracy"] = accuracy
            model_results[f"{split_name}_precision_weighted"] = precision
            model_results[f"{split_name}_recall_weighted"] = recall
            model_results[f"{split_name}_f1_weighted"] = fscore

        results[model_name] = model_results

    print("Evaluation complete")
    return results


def main() -> None:
    """Execute the training pipeline from command-line arguments."""
    args = parse_args()
    print("Start of training script")

    config = load_config(args.config)
    platform_name = args.platform or config["platform"]
    random_seed = config["random_seed"]

    print(f"Config loaded from: {args.config}")
    print(f"Platform: {platform_name}")
    print(f"Resolution: {args.resolution}")

    root_data_dir = get_platform_dir(platform_name, config)
    ml_ready_path = get_ml_ready_path(root_data_dir, config, args.resolution)

    zones_df = load_data(ml_ready_path)
    predictors = build_predictors(zones_df)

    train_df, val_df, test_df = split_data(
        zones_df=zones_df,
        random_seed=random_seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )

    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        predictors=predictors,
        target_column=args.target_column,
    )

    models = train_models(x_train=x_train, y_train=y_train, random_seed=random_seed)

    evaluation_inputs = {
        "train": (x_train, y_train),
        "validation": (x_val, y_val),
        "test": (x_test, y_test),
    }
    results = evaluate_models(models=models, split_data_dict=evaluation_inputs)

    print("\nModel metrics:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

