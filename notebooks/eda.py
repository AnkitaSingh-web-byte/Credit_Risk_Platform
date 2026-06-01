"""
Exploratory Data Analysis (EDA) module for the Credit Risk Intelligence Platform.

Generates interactive Plotly visualisations for credit-risk data exploration.
Every public function accepts a pandas DataFrame (based on the Home Credit
``application_train`` schema) and returns one or more ``plotly.graph_objects.Figure``
objects that are ready to be rendered in Streamlit via ``st.plotly_chart``.

All charts share a consistent dark theme applied through ``_apply_theme``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_REPAID = "#667eea"   # Soft indigo  – TARGET 0 (repaid)
COLOR_DEFAULT = "#f56565"  # Coral red    – TARGET 1 (default)
PALETTE = [COLOR_REPAID, COLOR_DEFAULT]

# ---------------------------------------------------------------------------
# Theme helper
# ---------------------------------------------------------------------------

def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the platform-wide dark theme to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa", family="Inter"),
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ===================================================================
# 1. Target distribution
# ===================================================================

def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart showing TARGET distribution (0 = Repaid, 1 = Default).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``TARGET`` column with values 0 and 1.

    Returns
    -------
    go.Figure
    """
    counts = df["TARGET"].value_counts().sort_index()
    labels = ["Repaid (0)", "Default (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=[COLOR_REPAID, COLOR_DEFAULT]),
                textinfo="label+percent+value",
                textfont=dict(size=14),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
                hole=0.45,
            )
        ]
    )
    fig.update_layout(title="Loan Repayment Distribution")
    return _apply_theme(fig)


# ===================================================================
# 2. EXT_SOURCE distributions by TARGET
# ===================================================================

def plot_ext_source_distribution(df: pd.DataFrame) -> go.Figure:
    """Overlaid histograms of EXT_SOURCE_1/2/3 coloured by TARGET (3-row subplots).

    Lower EXT_SOURCE scores are associated with higher default risk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``EXT_SOURCE_1``, ``EXT_SOURCE_2``, ``EXT_SOURCE_3``, and
        ``TARGET`` columns.

    Returns
    -------
    go.Figure
    """
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[f"{col} Distribution by Target" for col in ext_cols],
        vertical_spacing=0.10,
    )

    for idx, col in enumerate(ext_cols, start=1):
        for target_val, colour, name in [
            (0, COLOR_REPAID, "Repaid"),
            (1, COLOR_DEFAULT, "Default"),
        ]:
            subset = df.loc[df["TARGET"] == target_val, col].dropna()
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    nbinsx=60,
                    marker_color=colour,
                    opacity=0.65,
                    name=name,
                    legendgroup=name,
                    showlegend=(idx == 1),  # show legend only on first subplot
                    hovertemplate=f"{col}: %{{x:.2f}}<br>Count: %{{y:,}}<extra>{name}</extra>",
                ),
                row=idx,
                col=1,
            )
        fig.update_xaxes(title_text=col, row=idx, col=1)
        fig.update_yaxes(title_text="Count", row=idx, col=1)

    fig.update_layout(
        barmode="overlay",
        height=900,
        title="External Source Scores by Repayment Status",
    )
    return _apply_theme(fig)


# ===================================================================
# 3. Income distribution by TARGET
# ===================================================================

def plot_income_distribution(df: pd.DataFrame) -> go.Figure:
    """Box plot of ``AMT_INCOME_TOTAL`` by TARGET, clipped at 99th percentile.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``AMT_INCOME_TOTAL`` and ``TARGET``.

    Returns
    -------
    go.Figure
    """
    cap = df["AMT_INCOME_TOTAL"].quantile(0.99)
    tmp = df[["AMT_INCOME_TOTAL", "TARGET"]].copy()
    tmp["AMT_INCOME_TOTAL"] = tmp["AMT_INCOME_TOTAL"].clip(upper=cap)
    tmp["TARGET_LABEL"] = tmp["TARGET"].map({0: "Repaid", 1: "Default"})

    fig = go.Figure()
    for target_val, colour, label in [
        (0, COLOR_REPAID, "Repaid"),
        (1, COLOR_DEFAULT, "Default"),
    ]:
        subset = tmp.loc[tmp["TARGET"] == target_val, "AMT_INCOME_TOTAL"]
        fig.add_trace(
            go.Box(
                y=subset,
                name=label,
                marker_color=colour,
                boxmean=True,
                hoverinfo="y+name",
            )
        )

    median_repaid = tmp.loc[tmp["TARGET"] == 0, "AMT_INCOME_TOTAL"].median()
    median_default = tmp.loc[tmp["TARGET"] == 1, "AMT_INCOME_TOTAL"].median()

    fig.update_layout(
        title="Income Distribution by Repayment Status (clipped at 99th pctl)",
        yaxis_title="Income (AMT_INCOME_TOTAL)",
        annotations=[
            dict(
                x=0, y=median_repaid,
                text=f"Median: {median_repaid:,.0f}",
                showarrow=True, arrowhead=2, ax=60, ay=-30,
                font=dict(color=COLOR_REPAID, size=12),
            ),
            dict(
                x=1, y=median_default,
                text=f"Median: {median_default:,.0f}",
                showarrow=True, arrowhead=2, ax=60, ay=-30,
                font=dict(color=COLOR_DEFAULT, size=12),
            ),
        ],
    )
    return _apply_theme(fig)


# ===================================================================
# 4. Age distribution by TARGET
# ===================================================================

def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of applicant age (years) coloured by TARGET.

    ``DAYS_BIRTH`` is converted to positive years (``/ -365``).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``DAYS_BIRTH`` and ``TARGET``.

    Returns
    -------
    go.Figure
    """
    tmp = df[["DAYS_BIRTH", "TARGET"]].copy()
    tmp["AGE_YEARS"] = (tmp["DAYS_BIRTH"] / -365).round(1)

    fig = go.Figure()
    for target_val, colour, label in [
        (0, COLOR_REPAID, "Repaid"),
        (1, COLOR_DEFAULT, "Default"),
    ]:
        subset = tmp.loc[tmp["TARGET"] == target_val, "AGE_YEARS"]
        fig.add_trace(
            go.Histogram(
                x=subset,
                nbinsx=50,
                marker_color=colour,
                opacity=0.65,
                name=label,
                hovertemplate="Age: %{x:.0f} yrs<br>Count: %{y:,}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="overlay",
        title="Age Distribution by Repayment Status",
        xaxis_title="Age (years)",
        yaxis_title="Count",
    )
    return _apply_theme(fig)


# ===================================================================
# 5. Employment analysis (anomaly detection + distribution)
# ===================================================================

def plot_employment_analysis(df: pd.DataFrame) -> go.Figure:
    """Two-panel analysis of DAYS_EMPLOYED.

    * Left panel – bar chart: anomalous value 365 243 count vs. normal.
    * Right panel – histogram of employment years for non-anomalous records.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``DAYS_EMPLOYED`` and ``TARGET``.

    Returns
    -------
    go.Figure
    """
    anomaly_mask = df["DAYS_EMPLOYED"] == 365243
    n_anomaly = int(anomaly_mask.sum())
    n_normal = int((~anomaly_mask).sum())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "DAYS_EMPLOYED Anomaly (365 243)",
            "Employment Years (non-anomalous)",
        ],
        horizontal_spacing=0.15,
    )

    # -- Left: anomaly bar --
    fig.add_trace(
        go.Bar(
            x=["Anomalous (365 243)", "Normal"],
            y=[n_anomaly, n_normal],
            marker_color=[COLOR_DEFAULT, COLOR_REPAID],
            text=[f"{n_anomaly:,}", f"{n_normal:,}"],
            textposition="outside",
            hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # -- Right: employment-years histogram for non-anomalous records --
    normal_employment = df.loc[~anomaly_mask, "DAYS_EMPLOYED"].copy()
    normal_employment = (normal_employment / -365).clip(lower=0)

    for target_val, colour, label in [
        (0, COLOR_REPAID, "Repaid"),
        (1, COLOR_DEFAULT, "Default"),
    ]:
        mask = df.loc[~anomaly_mask, "TARGET"] == target_val
        subset = normal_employment[mask]
        fig.add_trace(
            go.Histogram(
                x=subset,
                nbinsx=50,
                marker_color=colour,
                opacity=0.65,
                name=label,
                hovertemplate="Years: %{x:.1f}<br>Count: %{y:,}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Employment Years", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(
        barmode="overlay",
        height=450,
        title="Employment Duration Analysis",
    )
    return _apply_theme(fig)


# ===================================================================
# 6. Default rate by loan type
# ===================================================================

def plot_loan_type_default_rate(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of default rate by ``NAME_CONTRACT_TYPE``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``NAME_CONTRACT_TYPE`` and ``TARGET``.

    Returns
    -------
    go.Figure
    """
    stats = (
        df.groupby("NAME_CONTRACT_TYPE")["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "default_rate", "count": "total"})
    )
    stats["default_rate_pct"] = (stats["default_rate"] * 100).round(2)

    fig = go.Figure()

    # Default rate bars
    fig.add_trace(
        go.Bar(
            x=stats["NAME_CONTRACT_TYPE"],
            y=stats["default_rate_pct"],
            marker_color=COLOR_DEFAULT,
            text=stats["default_rate_pct"].apply(lambda v: f"{v:.2f}%"),
            textposition="outside",
            name="Default Rate (%)",
            yaxis="y",
            hovertemplate="%{x}<br>Default Rate: %{y:.2f}%<extra></extra>",
        )
    )

    # Total count bars on secondary axis
    fig.add_trace(
        go.Bar(
            x=stats["NAME_CONTRACT_TYPE"],
            y=stats["total"],
            marker_color=COLOR_REPAID,
            text=stats["total"].apply(lambda v: f"{v:,}"),
            textposition="outside",
            name="Total Applications",
            yaxis="y2",
            opacity=0.45,
            hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Default Rate by Loan Contract Type",
        xaxis_title="Contract Type",
        yaxis=dict(title="Default Rate (%)", side="left", showgrid=False),
        yaxis2=dict(
            title="Total Applications",
            side="right",
            overlaying="y",
            showgrid=False,
        ),
        barmode="group",
    )
    return _apply_theme(fig)


# ===================================================================
# 7. Correlation heatmap (top 15 features vs TARGET)
# ===================================================================

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of the 15 features most correlated (absolute value) with TARGET.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``TARGET`` and numeric columns.

    Returns
    -------
    go.Figure
    """
    numeric_df = df.select_dtypes(include="number")
    corr_with_target = numeric_df.corr()["TARGET"].drop("TARGET", errors="ignore")
    top15 = corr_with_target.abs().nlargest(15).index.tolist()

    corr_matrix = numeric_df[top15].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Corr: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Correlation Heatmap – Top 15 Features",
        height=650,
        xaxis=dict(tickangle=-45),
    )
    return _apply_theme(fig)


# ===================================================================
# 8. Missing values
# ===================================================================

def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the 20 columns with the most missing values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    go.Figure
    """
    missing = df.isnull().mean().sort_values(ascending=False).head(20) * 100
    missing = missing.sort_values(ascending=True)  # ascending for horizontal bars

    fig = go.Figure(
        go.Bar(
            x=missing.values,
            y=missing.index,
            orientation="h",
            marker_color=COLOR_REPAID,
            text=[f"{v:.1f}%" for v in missing.values],
            textposition="outside",
            hovertemplate="%{y}<br>Missing: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top 20 Columns with Missing Values",
        xaxis_title="Missing (%)",
        yaxis_title="",
        height=600,
    )
    return _apply_theme(fig)


# ===================================================================
# 9. Feature distributions (multiple figures)
# ===================================================================

def plot_feature_distributions(df: pd.DataFrame) -> list[go.Figure]:
    """Distribution plots for key numeric features.

    Features: ``AMT_CREDIT``, ``AMT_ANNUITY``, ``AMT_GOODS_PRICE``,
    ``AMT_INCOME_TOTAL``.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[go.Figure]
        One figure per feature.
    """
    features = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]
    figures: list[go.Figure] = []

    for feat in features:
        if feat not in df.columns:
            continue

        cap = df[feat].quantile(0.99)
        clipped = df[[feat, "TARGET"]].copy()
        clipped[feat] = clipped[feat].clip(upper=cap)

        fig = go.Figure()
        for target_val, colour, label in [
            (0, COLOR_REPAID, "Repaid"),
            (1, COLOR_DEFAULT, "Default"),
        ]:
            subset = clipped.loc[clipped["TARGET"] == target_val, feat].dropna()
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    nbinsx=60,
                    marker_color=colour,
                    opacity=0.65,
                    name=label,
                    hovertemplate=f"{feat}: %{{x:,.0f}}<br>Count: %{{y:,}}<extra></extra>",
                )
            )

        fig.update_layout(
            barmode="overlay",
            title=f"{feat} Distribution by Target (clipped at 99th pctl)",
            xaxis_title=feat,
            yaxis_title="Count",
        )
        figures.append(_apply_theme(fig))

    return figures


# ===================================================================
# 10. Categorical analysis – education type
# ===================================================================

def plot_categorical_analysis(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of default rate by ``NAME_EDUCATION_TYPE``,
    sorted from highest to lowest default rate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``NAME_EDUCATION_TYPE`` and ``TARGET``.

    Returns
    -------
    go.Figure
    """
    stats = (
        df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "default_rate", "count": "total"})
    )
    stats["default_rate_pct"] = (stats["default_rate"] * 100).round(2)
    stats = stats.sort_values("default_rate_pct", ascending=True)

    # Colour gradient: higher default rate → redder
    max_rate = stats["default_rate_pct"].max()
    colours = [
        f"rgba({int(245 * (r / max_rate))}, {int(101 * (1 - r / max_rate) + 100)}, "
        f"{int(101 * (1 - r / max_rate) + 130)}, 0.85)"
        if max_rate > 0
        else COLOR_REPAID
        for r in stats["default_rate_pct"]
    ]

    fig = go.Figure(
        go.Bar(
            x=stats["default_rate_pct"],
            y=stats["NAME_EDUCATION_TYPE"],
            orientation="h",
            marker_color=colours,
            text=stats.apply(
                lambda row: f"{row['default_rate_pct']:.2f}%  (n={row['total']:,})",
                axis=1,
            ),
            textposition="outside",
            hovertemplate="%{y}<br>Default Rate: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Default Rate by Education Type",
        xaxis_title="Default Rate (%)",
        yaxis_title="",
        height=450,
    )
    return _apply_theme(fig)


# ===================================================================
# 11. Business insights (non-visual)
# ===================================================================

def get_business_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute high-level business insights from the dataset.

    Returns
    -------
    list[dict]
        Each dict has keys ``title``, ``description``, ``metric``, ``icon``.
    """
    total_apps = len(df)
    default_rate = df["TARGET"].mean() * 100
    avg_income = df["AMT_INCOME_TOTAL"].mean()
    avg_credit = df["AMT_CREDIT"].mean()

    # Most common contract type
    top_contract = df["NAME_CONTRACT_TYPE"].value_counts().idxmax()
    top_contract_pct = (
        df["NAME_CONTRACT_TYPE"].value_counts(normalize=True).iloc[0] * 100
    )

    # Gender split
    gender_counts = df["CODE_GENDER"].value_counts(normalize=True) * 100
    female_pct = gender_counts.get("F", 0)
    male_pct = gender_counts.get("M", 0)

    # Average age
    avg_age = (df["DAYS_BIRTH"] / -365).mean()

    # Anomalous employment count
    anomaly_pct = (df["DAYS_EMPLOYED"] == 365243).mean() * 100

    insights: list[dict[str, Any]] = [
        {
            "title": "Total Applicants",
            "description": f"The dataset contains {total_apps:,} loan applications.",
            "metric": f"{total_apps:,}",
            "icon": "👥",
        },
        {
            "title": "Default Rate",
            "description": (
                f"{default_rate:.1f}% of applicants defaulted on their loans, "
                "indicating a significant class imbalance."
            ),
            "metric": f"{default_rate:.1f}%",
            "icon": "⚠️",
        },
        {
            "title": "Average Income",
            "description": (
                f"Mean total income across all applicants is "
                f"{avg_income:,.0f}."
            ),
            "metric": f"{avg_income:,.0f}",
            "icon": "💰",
        },
        {
            "title": "Most Common Loan Type",
            "description": (
                f"'{top_contract}' loans dominate at "
                f"{top_contract_pct:.1f}% of all applications."
            ),
            "metric": f"{top_contract} ({top_contract_pct:.1f}%)",
            "icon": "📄",
        },
        {
            "title": "Average Credit Amount",
            "description": (
                f"The average requested credit amount is "
                f"{avg_credit:,.0f}."
            ),
            "metric": f"{avg_credit:,.0f}",
            "icon": "🏦",
        },
        {
            "title": "Gender Distribution",
            "description": (
                f"Female applicants: {female_pct:.1f}%, "
                f"Male applicants: {male_pct:.1f}%."
            ),
            "metric": f"F {female_pct:.1f}% / M {male_pct:.1f}%",
            "icon": "👫",
        },
        {
            "title": "Average Applicant Age",
            "description": (
                f"The average applicant age is {avg_age:.1f} years."
            ),
            "metric": f"{avg_age:.1f} yrs",
            "icon": "🎂",
        },
        {
            "title": "Employment Anomaly",
            "description": (
                f"{anomaly_pct:.1f}% of records have the DAYS_EMPLOYED "
                "anomalous value of 365 243, likely encoding a special status."
            ),
            "metric": f"{anomaly_pct:.1f}%",
            "icon": "🔍",
        },
    ]
    return insights
