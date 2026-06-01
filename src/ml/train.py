"""
train.py - Model Training Module for Credit Risk Intelligence Platform

Handles LightGBM model training, saving, and loading for credit default prediction.
Uses class-weighted binary classification with GBDT boosting and early stopping.
"""

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    val_size: float = 0.2,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier for credit default prediction.

    Uses GBDT with class-weight adjustment (scale_pos_weight=11.4) to handle
    the ~92/8 class imbalance. Training uses a stratified validation split
    with early stopping to prevent overfitting.

    Args:
        X: Training feature matrix.
        y: Binary target series (0=repaid, 1=default).
        feature_names: List of feature column names to use.
        val_size: Fraction of data reserved for early-stopping validation.
        early_stopping_rounds: Patience for early stopping on validation AUC.

    Returns:
        Trained LGBMClassifier instance.
    """
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "scale_pos_weight": 11.4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)

    # Ensure we only use the specified features
    X_train_feat = X[feature_names]

    # Stratified split for early stopping validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_feat, y, test_size=val_size, random_state=42, stratify=y
    )

    print(f"Training set:   {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Features:       {len(feature_names)}")
    print(f"Default rate (train): {y_train.mean():.4f}")
    print(f"Default rate (val):   {y_val.mean():.4f}")
    print("-" * 50)

    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = model.best_iteration_
    best_score = model.best_score_["valid_0"]["auc"]
    print(f"\nBest iteration: {best_iter}")
    print(f"Best validation AUC: {best_score:.6f}")

    return model


def save_model(
    model: lgb.LGBMClassifier,
    feature_names: list,
    model_dir: str,
) -> None:
    """
    Persist trained model and feature metadata to disk.

    Saves:
        - lgbm_model.pkl: Serialized LGBMClassifier via joblib.
        - feature_names.json: Ordered list of feature names used during training.

    Args:
        model: Trained LGBMClassifier to save.
        feature_names: List of feature names the model expects at inference.
        model_dir: Directory path where model artifacts will be saved.
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    model_file = model_path / "lgbm_model.pkl"
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")

    # Save feature names for inference
    features_file = model_path / "feature_names.json"
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {features_file}")

    # Save model metadata
    metadata = {
        "n_features": len(feature_names),
        "n_estimators": model.n_estimators,
        "best_iteration": getattr(model, "best_iteration_", None),
        "best_score": None,
    }
    if hasattr(model, "best_score_") and model.best_score_:
        try:
            metadata["best_score"] = model.best_score_["valid_0"]["auc"]
        except (KeyError, TypeError):
            pass

    metadata_file = model_path / "model_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")


def load_model(model_dir: str) -> tuple[lgb.LGBMClassifier, list]:
    """
    Load a previously saved model and its feature names from disk.

    Args:
        model_dir: Directory containing lgbm_model.pkl and feature_names.json.

    Returns:
        Tuple of (trained LGBMClassifier, list of feature names).

    Raises:
        FileNotFoundError: If the model or feature names file is missing.
    """
    model_path = Path(model_dir)

    model_file = model_path / "lgbm_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    features_file = model_path / "feature_names.json"
    if not features_file.exists():
        raise FileNotFoundError(f"Feature names file not found: {features_file}")

    model = joblib.load(model_file)
    with open(features_file, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"Model loaded from: {model_file}")
    print(f"Features: {len(feature_names)}")

    return model, feature_names


if __name__ == "__main__":

    print("=" * 50)
    print("CREDIT RISK MODEL TRAINING")
    print("=" * 50)

    from src.data.loader import load_train_data
    from src.data.preprocessor import CreditDataPreprocessor

    print("Loading dataset...")

    df = load_train_data()

    print(f"Dataset shape: {df.shape}")

    y = df["TARGET"]

    X = df.drop(
        columns=["TARGET"]
    )

    print("Preprocessing features...")

    preprocessor = CreditDataPreprocessor()

    X_processed = preprocessor.fit_transform(X)

    print(
        f"Processed feature shape: {X_processed.shape}"
    )

    print("Training LightGBM model...")

    model = train_model(
        X=X_processed,
        y=y,
        feature_names=list(X_processed.columns)
    )

    print("Saving model...")

    save_model(
        model=model,
        feature_names=list(X_processed.columns),
        model_dir="models"
    )

    preprocessor.save(
        "models/preprocessor.pkl"
    )

    print("\nTraining Complete!")
