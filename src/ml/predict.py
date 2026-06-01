"""
predict.py

Prediction Module for Credit Risk Intelligence Platform.

Loads trained LightGBM model and preprocessor,
generates risk scores and interpretable risk bands.
"""

import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path
from typing import Dict


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"


# =====================================================
# MODEL LOADING
# =====================================================

def load_trained_model():

    model_path = MODELS_DIR / "lgbm_model.pkl"

    model = joblib.load(model_path)

    return model


def load_preprocessor():

    preprocessor_path = MODELS_DIR / "preprocessor.pkl"

    preprocessor = joblib.load(preprocessor_path)

    return preprocessor


def load_feature_names():

    feature_file = MODELS_DIR / "feature_names.json"

    with open(feature_file, "r") as f:

        feature_names = json.load(f)

    return feature_names


# =====================================================
# SINGLE APPLICANT PREDICTION
# =====================================================

def predict_risk(
    model: lgb.LGBMClassifier,
    input_data: pd.DataFrame
) -> Dict:

    probabilities = model.predict_proba(input_data)

    default_probability = float(
        probabilities[0, 1]
    )

    risk_score = int(
        round(default_probability * 100)
    )

    risk_score = max(
        0,
        min(100, risk_score)
    )

    if risk_score < 30:

        risk_band = "Low"
        risk_color = "#4ade80"

    elif risk_score <= 60:

        risk_band = "Medium"
        risk_color = "#fbbf24"

    else:

        risk_band = "High"
        risk_color = "#f87171"

    return {

        "default_probability": round(
            default_probability,
            4
        ),

        "risk_score": risk_score,

        "risk_band": risk_band,

        "risk_color": risk_color
    }


# =====================================================
# BATCH PREDICTION
# =====================================================

def batch_predict_risk(
    model: lgb.LGBMClassifier,
    input_data: pd.DataFrame
) -> pd.DataFrame:

    probabilities = model.predict_proba(
        input_data
    )

    default_probs = probabilities[:, 1]

    risk_scores = np.clip(
        np.round(
            default_probs * 100
        ).astype(int),
        0,
        100
    )

    risk_bands = np.where(
        risk_scores < 30,
        "Low",
        np.where(
            risk_scores <= 60,
            "Medium",
            "High"
        )
    )

    risk_colors = np.where(
        risk_scores < 30,
        "#4ade80",
        np.where(
            risk_scores <= 60,
            "#fbbf24",
            "#f87171"
        )
    )

    return pd.DataFrame(
        {
            "default_probability": default_probs,
            "risk_score": risk_scores,
            "risk_band": risk_bands,
            "risk_color": risk_colors
        }
    )


# =====================================================
# SAMPLE INPUT HELPER
# =====================================================

def get_sample_input(
    df: pd.DataFrame,
    idx: int = 0
):

    sample = df.iloc[[idx]].copy()

    sample = sample.fillna(0)

    return sample


# =====================================================
# END-TO-END PREDICTION
# =====================================================

def predict_from_raw_dataframe(
    raw_df: pd.DataFrame
):

    model = load_trained_model()

    preprocessor = load_preprocessor()

    processed_df = preprocessor.transform(
        raw_df
    )

    result = predict_risk(
        model,
        processed_df
    )

    return result


# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":

    print("=" * 50)
    print("CREDIT RISK PREDICTION TEST")
    print("=" * 50)

    from src.data.loader import load_train_data

    print("Loading model...")

    model = load_trained_model()

    print("Loading preprocessor...")

    preprocessor = load_preprocessor()

    print("Loading dataset...")

    df = load_train_data()

    print(f"Dataset shape: {df.shape}")

    X = df.drop(
        columns=["TARGET"]
    )

    print("Preprocessing sample...")

    processed_X = preprocessor.transform(
        X
    )

    sample = processed_X.iloc[[0]]

    result = predict_risk(
        model,
        sample
    )

    print("\nPrediction Result")
    print("-" * 30)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )