#!/usr/bin/env python
"""Train a climate-zone classifier with PyTorch.
This script loads the preprocessed climate-zone tabular dataset, prepares train,
validation, and test splits, builds a feed-forward neural network, trains it
with PyTorch, and reports evaluation metrics for each split.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ALLOWED_PLATFORMS = ("mo_linux", "jasmin", "local")
ALLOWED_RESOLUTIONS = (0.1, 0.5, 1.0)
DEFAULT_TARGET_COLUMN = "climate_group"
DEFAULT_TEST_FRAC = 0.1
DEFAULT_VAL_FRAC = 0.1
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_DIMS = (60, 60, 60)
DEFAULT_FILTER_PERIOD_START = 1991
DEFAULT_FILTER_SCENARIO = "historic"


@dataclass(frozen=True)
class SplitData:
    """Container for the prepared train/validation/test data sets."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class ClimateZonesDataset(Dataset):
    """PyTorch dataset for the climate-zone tabular data."""

    def __init__(
            self,
            df: pd.DataFrame,
            feature_columns: list[str],
            target_column: str,
            feature_scaler: sklearn.preprocessing.StandardScaler,
            target_encoder: sklearn.preprocessing.LabelEncoder,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._feature_columns = feature_columns
        self._target_column = target_column
        # Standardise numeric inputs using statistics learned from the training split.
        features = feature_scaler.transform(self._df[self._feature_columns])
        targets = target_encoder.transform(self._df[self._target_column])
        self._features = torch.tensor(features, dtype=torch.float32)
        self._targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features[index], self._targets[index]


class ClimateZoneClassifier(nn.Module):
    """Simple fully-connected classifier for climate-zone prediction."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        # CrossEntropyLoss expects raw logits, so we do not add a softmax layer here.
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train a climate-zone classifier with PyTorch.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=ALLOWED_PLATFORMS,
        required=True,
        help="Platform name used to resolve the data directory.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        choices=ALLOWED_RESOLUTIONS,
        required=True,
        help="Resolution to use when loading the ML-ready CSV file.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=DEFAULT_TARGET_COLUMN,
        help="Target label column to predict.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Fraction of the data reserved for testing.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Fraction of the data reserved for validation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate used by the Adam optimiser.",
    )
    parser.add_argument(
        "--model-out-dir",
        dest="model_out_dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help='Root directory for saving model weights and other outputs.',
    )
    return parser.parse_args()


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    """Load the JSON configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def get_platform_dir(platform: str, config: dict[str, Any]) -> pathlib.Path:
    """Resolve the base climate-zones directory for a platform.
    The configuration file may define platform-specific base directories. If a
    platform cannot be found there, the function falls back to ``$HOME``.
    """
    default_dirs = config.get("default_dirs", {})
    try:
        return pathlib.Path(default_dirs[platform]) / "climate_zones"
    except KeyError:
        return pathlib.Path(os.environ["HOME"]) / "climate_zones"


def get_ml_ready_path(root_data_dir: pathlib.Path, config: dict[str, Any], resolution: float) -> pathlib.Path:
    """Construct the path to the ML-ready CSV for the selected resolution."""
    resolutions_dict = {float(key): value for key, value in config["resolutions_names"].items()}
    resolution_name = resolutions_dict[resolution]
    csv_template = config["csv_out_template"]
    return root_data_dir / "ml_ready" / csv_template.format(resolution=resolution_name)


def load_data(data_path: pathlib.Path) -> pd.DataFrame:
    """Load the CSV data and apply the same historical subset used in the notebook."""
    if not data_path.is_file():
        raise FileNotFoundError(f"ML-ready data file not found: {data_path}")
    print(f"Loading data from: {data_path}")
    zones_df = pd.read_csv(data_path)
    # Reduce the data volume to keep memory use manageable for this tutorial.
    zones_df = zones_df[
        (zones_df["period_start"] == DEFAULT_FILTER_PERIOD_START) & (zones_df["scenario"] == DEFAULT_FILTER_SCENARIO)]
    print(f"Data loaded successfully: {zones_df.shape}")
    return zones_df


def select_feature_columns(zones_df: pd.DataFrame) -> list[str]:
    """Select the predictor columns used by the tutorial model."""
    precip_mean = [column for column in zones_df.columns if "precipitation" in column and "mean" in column]
    temp_mean = [column for column in zones_df.columns if "air_temperature" in column and "mean" in column]
    feature_columns = precip_mean + temp_mean
    if not feature_columns:
        raise ValueError("No feature columns were found in the input data.")
    return feature_columns


def split_data(zones_df: pd.DataFrame, random_seed: int, test_frac: float, val_frac: float) -> SplitData:
    """Create train, validation, and test splits using grouped sampling."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1.")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if test_frac + val_frac >= 1.0:
        raise ValueError("test_frac + val_frac must be less than 1.")
    group_cols = ["period_start", "scenario"]
    val_frac_sub = val_frac / (1.0 - test_frac)
    test_df = zones_df.groupby(group_cols).sample(frac=test_frac, random_state=random_seed)
    remaining_df = zones_df.drop(test_df.index)
    validation_df = remaining_df.groupby(group_cols).sample(frac=val_frac_sub, random_state=random_seed)
    train_df = remaining_df.drop(validation_df.index)
    print(f"Train split: {train_df.shape}")
    print(f"Validation split: {validation_df.shape}")
    print(f"Test split: {test_df.shape}")
    return SplitData(train=train_df, validation=validation_df, test=test_df)


def fit_preprocessors(train_df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[
    sklearn.preprocessing.StandardScaler, sklearn.preprocessing.LabelEncoder]:
    """Fit preprocessing objects using the training split only."""
    feature_scaler = sklearn.preprocessing.StandardScaler()
    feature_scaler.fit(train_df[feature_columns])
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_encoder.fit(train_df[target_column])
    return feature_scaler, target_encoder


def build_datasets(
        splits: SplitData,
        feature_columns: list[str],
        target_column: str,
        feature_scaler: sklearn.preprocessing.StandardScaler,
        target_encoder: sklearn.preprocessing.LabelEncoder,
) -> tuple[ClimateZonesDataset, ClimateZonesDataset, ClimateZonesDataset]:
    """Create PyTorch datasets for the three data splits."""
    train_dataset = ClimateZonesDataset(splits.train, feature_columns, target_column, feature_scaler, target_encoder)
    validation_dataset = ClimateZonesDataset(splits.validation, feature_columns, target_column, feature_scaler,
                                             target_encoder)
    test_dataset = ClimateZonesDataset(splits.test, feature_columns, target_column, feature_scaler, target_encoder)
    return train_dataset, validation_dataset, test_dataset


def build_dataloaders(
        train_dataset: ClimateZonesDataset,
        validation_dataset: ClimateZonesDataset,
        test_dataset: ClimateZonesDataset,
        batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch data loaders for the train, validation, and test datasets."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, validation_loader, test_loader


def create_model(input_dim: int, num_classes: int, hidden_dims: tuple[int, ...]) -> ClimateZoneClassifier:
    """Instantiate the neural network used for climate-zone classification."""
    return ClimateZoneClassifier(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)


def train_model(
        model: ClimateZoneClassifier,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        device: torch.device,
        epochs: int,
        learning_rate: float,
) -> dict[str, list[float]]:
    """Train the model and record loss values for each epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "validation_loss": []}
    print("Starting training")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        average_train_loss = running_train_loss / max(len(train_loader), 1)
        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in validation_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                logits = model(batch_features)
                running_validation_loss += criterion(logits, batch_targets).item()
        average_validation_loss = running_validation_loss / max(len(validation_loader), 1)
        history["train_loss"].append(average_train_loss)
        history["validation_loss"].append(average_validation_loss)
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"train_loss={average_train_loss:.4f} | validation_loss={average_validation_loss:.4f}"
        )
    print("Training complete")
    return history


def evaluate_model(
        model: ClimateZoneClassifier,
        data_loader: DataLoader,
        device: torch.device,
        target_encoder: sklearn.preprocessing.LabelEncoder,
        split_name: str,
) -> dict[str, Any]:
    """Evaluate the model on one split and return summary metrics."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(batch_targets.tolist())
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    return {
        "split": split_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classes": list(target_encoder.classes_),
    }


def main() -> None:
    """Run the end-to-end climate-zone training workflow."""
    args = parse_args()
    config = load_config(args.config)
    print("Starting climate-zone training script")
    print(f"Config path: {args.config}")
    print(f"Platform: {args.platform}")
    print(f"Resolution: {args.resolution}")
    root_data_dir = get_platform_dir(args.platform, config)
    ml_ready_path = get_ml_ready_path(root_data_dir, config, args.resolution)
    zones_df = load_data(ml_ready_path)
    feature_columns = select_feature_columns(zones_df)
    splits = split_data(
        zones_df=zones_df,
        random_seed=int(config["random_seed"]),
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )
    feature_scaler, target_encoder = fit_preprocessors(splits.train, feature_columns, args.target_column)
    train_dataset, validation_dataset, test_dataset = build_datasets(
        splits=splits,
        feature_columns=feature_columns,
        target_column=args.target_column,
        feature_scaler=feature_scaler,
        target_encoder=target_encoder,
    )
    train_loader, validation_loader, test_loader = build_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = create_model(
        input_dim=len(feature_columns),
        num_classes=len(target_encoder.classes_),
        hidden_dims=DEFAULT_HIDDEN_DIMS,
    ).to(device)

    exp_name = 'climate_zones_train_torch'
    exp_dir = args.model_out_dir / exp_name
    if not exp_dir.is_dir():
        exp_dir.mkdir(parents=True)
        print(f'created experiment directory {exp_dir}')
    else:
        print(f'experiment directory {exp_dir}')

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    results = {
        "history": history,
        "metrics": {
            "train": evaluate_model(model, train_loader, device, target_encoder, "train"),
            "validation": evaluate_model(model, validation_loader, device, target_encoder, "validation"),
            "test": evaluate_model(model, test_loader, device, target_encoder, "test"),
        },
    }

    model_name = f'model_{exp_name}.pth'
    model_out_path = exp_dir / model_name

    print(f'saving model to {model_out_path}')
    torch.save(model.state_dict(), model_out_path)

    print("\nEvaluation results:")
    print(json.dumps(results, indent=2))

    results_path = exp_dir / 'climate_zones_torch_results.json'
    with open(results_path, 'w') as results_file:
        json.dump(results, results_file, indent=2)


if __name__ == "__main__":
    main()
