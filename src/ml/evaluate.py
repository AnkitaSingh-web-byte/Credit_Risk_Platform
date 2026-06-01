"""
evaluate.py - Model Evaluation Module for Credit Risk Intelligence Platform

Provides comprehensive model evaluation metrics and Plotly visualizations
for the Streamlit dashboard. All plots use a dark theme consistent with the
platform's UI design.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Dark theme layout defaults shared by all plots
# ─────────────────────────────────────────────────────────────────────────────
_DARK_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#fafafa", family="Inter"),
)


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the standard dark theme to a Plotly figure."""
    fig.update_layout(**_DARK_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute a full suite of classification metrics on held-out test data.

    Metrics returned:
        - roc_auc: Area under the ROC curve (probability-based).
        - accuracy, precision, recall, f1: Threshold-based at 0.5.
        - confusion_matrix: 2×2 array [[TN, FP], [FN, TP]].

    Args:
        model: Trained LGBMClassifier.
        X_test: Test feature matrix.
        y_test: True binary labels for the test set.

    Returns:
        Dictionary of metric names to their float/array values.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> go.Figure:
    """
    Plot the ROC curve with AUC score displayed in the title.

    Args:
        model: Trained LGBMClassifier.
        X_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        Plotly Figure with the ROC curve and diagonal reference line.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc_score:.4f})",
            line=dict(color="#667eea", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.15)",
        )
    )

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="#6b7280", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"ROC Curve (AUC = {auc_score:.4f})",
            font=dict(size=16),
        ),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 1.02],
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        legend=dict(x=0.55, y=0.1),
        height=450,
    )

    return _apply_dark_theme(fig)


def plot_precision_recall_curve(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> go.Figure:
    """
    Plot the Precision-Recall curve with average precision in the title.

    The PR curve is particularly informative for this imbalanced dataset
    (8% positive class).

    Args:
        model: Trained LGBMClassifier.
        X_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        Plotly Figure with the Precision-Recall curve.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR Curve (AP = {ap_score:.4f})",
            line=dict(color="#764ba2", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(118, 75, 162, 0.15)",
        )
    )

    # Baseline: prevalence of positive class
    baseline = float(y_test.mean())
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name=f"Baseline ({baseline:.3f})",
            line=dict(color="#6b7280", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Precision-Recall Curve (AP = {ap_score:.4f})",
            font=dict(size=16),
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 1.02],
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        legend=dict(x=0.05, y=0.1),
        height=450,
    )

    return _apply_dark_theme(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> go.Figure:
    """
    Plot an annotated confusion matrix heatmap showing both counts and percentages.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Plotly Figure with the confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()

    # Build annotation text: count + percentage
    annotations = []
    labels = ["Repaid (0)", "Default (1)"]
    text_matrix = []
    for i in range(2):
        row_texts = []
        for j in range(2):
            count = cm[i][j]
            pct = count / total * 100
            row_texts.append(f"{count:,}<br>({pct:.1f}%)")
        text_matrix.append(row_texts)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#fafafa"),
            colorscale=[
                [0, "#1e1b4b"],
                [0.5, "#667eea"],
                [1, "#764ba2"],
            ],
            showscale=True,
            colorbar=dict(
                title="Count",
                tickfont=dict(color="#fafafa"),
                titlefont=dict(color="#fafafa"),
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Confusion Matrix",
            font=dict(size=16),
        ),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        height=450,
        width=500,
    )

    return _apply_dark_theme(fig)


def plot_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list,
    top_n: int = 20,
) -> go.Figure:
    """
    Plot a horizontal bar chart of the top-N most important features.

    Importance is based on the LightGBM `feature_importances_` attribute
    (split-based importance by default). Bars use a gradient from
    #667eea (indigo) to #764ba2 (purple).

    Args:
        model: Trained LGBMClassifier with feature_importances_.
        feature_names: Ordered list of feature names matching training data.
        top_n: Number of top features to display.

    Returns:
        Plotly Figure with horizontal bar chart.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True)

    # Keep only top N
    importance_df = importance_df.tail(top_n)

    # Generate gradient colors from #667eea to #764ba2
    n = len(importance_df)
    colors = []
    for i in range(n):
        ratio = i / max(n - 1, 1)
        r = int(102 + (118 - 102) * ratio)
        g = int(126 + (75 - 126) * ratio)
        b = int(234 + (162 - 234) * ratio)
        colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure(
        go.Bar(
            x=importance_df["importance"].values,
            y=importance_df["feature"].values,
            orientation="h",
            marker=dict(color=colors),
            text=importance_df["importance"].values,
            textposition="outside",
            textfont=dict(size=10),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Feature Importances",
            font=dict(size=16),
        ),
        xaxis_title="Importance (split count)",
        yaxis_title="",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(450, top_n * 25),
        margin=dict(l=200),
    )

    return _apply_dark_theme(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
def get_cross_val_scores(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict:
    """
    Run stratified k-fold cross-validation and return AUC statistics.

    Uses StratifiedKFold to preserve class balance across folds — important
    given the ~8% default rate.

    Args:
        model: LGBMClassifier (will be cloned internally by sklearn).
        X: Full feature matrix.
        y: Full binary target series.
        cv: Number of folds (default: 5).

    Returns:
        Dictionary with:
            - cv_scores: List of per-fold AUC scores.
            - mean_auc: Mean AUC across folds.
            - std_auc: Standard deviation of AUC across folds.
            - n_folds: Number of folds used.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
    )

    result = {
        "cv_scores": scores.tolist(),
        "mean_auc": float(scores.mean()),
        "std_auc": float(scores.std()),
        "n_folds": cv,
    }

    print(f"Cross-validation AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    for i, score in enumerate(result["cv_scores"]):
        print(f"  Fold {i + 1}: {score:.4f}")

    return result
