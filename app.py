import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.data.loader import load_train_data

from src.ml.predict import (
    load_trained_model,
    load_preprocessor,
    predict_risk
)

from src.talk_to_data.chat_with_data import (
    ask_question
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    layout="wide"
)

# ==================================================
# TITLE
# ==================================================

st.title("🏦 Credit Risk Intelligence Platform")

st.markdown("""
Predict customer default risk using a LightGBM model trained on the
Home Credit Default Risk dataset.
""")

# ==================================================
# MODEL METRICS
# ==================================================

st.subheader("Model Performance Summary")

m1, m2, m3 = st.columns(3)

m1.metric("ROC AUC", "0.760")
m2.metric("Accuracy", "70.7%")
m3.metric("Recall", "67.4%")

st.markdown("---")

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "Risk Prediction",
    "Talk To Data",
    "Model Performance"
])

# ==================================================
# TAB 1 - RISK PREDICTION
# ==================================================

with tab1:

    st.header("Credit Risk Prediction")

    df = load_train_data()

    row_id = st.number_input(
        "Applicant Row Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    if st.button("Predict Risk"):

        model = load_trained_model()
        preprocessor = load_preprocessor()

        applicant = df.drop(
            columns=["TARGET"]
        ).iloc[[row_id]]

        applicant_processed = (
            preprocessor.transform(applicant)
        )

        result = predict_risk(
            model,
            applicant_processed
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Risk Score",
            result["risk_score"]
        )

        c3.metric(
            "Default Probability",
            f"{result['default_probability']:.2%}"
        )

        risk_band = result["risk_band"]

        with c2:

            if risk_band == "Low":
                st.success("🟢 LOW RISK")

            elif risk_band == "Medium":
                st.warning("🟡 MEDIUM RISK")

            else:
                st.error("🔴 HIGH RISK")

        # Gauge Chart

        probability = (
            result["default_probability"] * 100
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                title={
                    "text": "Default Probability (%)"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Applicant Details

        st.subheader(
            "Applicant Details"
        )

        display_cols = [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY"
        ]

        available_cols = [
            c for c in display_cols
            if c in applicant.columns
        ]

        st.dataframe(
            applicant[available_cols]
        )

# ==================================================
# TAB 2 - TALK TO DATA
# ==================================================

with tab2:

    st.header("Talk To Data")

    question = st.text_input(
        "Ask a question about the dataset"
    )

    if st.button("Run Question"):

        response = ask_question(
            question
        )

        st.subheader(
            "Generated SQL"
        )

        st.code(
            response["sql"],
            language="sql"
        )

        st.subheader(
            "Result"
        )

        st.dataframe(
            response["result"]
        )

# ==================================================
# TAB 3 - MODEL PERFORMANCE
# ==================================================

with tab3:

    st.header("Model Evaluation")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ROC AUC",
        "0.760"
    )

    c2.metric(
        "Accuracy",
        "70.7%"
    )

    c3.metric(
        "Recall",
        "67.4%"
    )

    c4, c5 = st.columns(2)

    c4.metric(
        "Precision",
        "16.9%"
    )

    c5.metric(
        "F1 Score",
        "27.0%"
    )

    st.markdown("---")

    st.subheader(
        "Confusion Matrix"
    )

    cm = [
        [40414, 16397],
        [1621, 3344]
    ]

    fig, ax = plt.subplots()

    ax.imshow(
        cm,
        cmap="Blues"
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    st.markdown("---")

    st.write("""
    LightGBM model trained on the
    Home Credit Default Risk dataset.

    This model predicts the probability
    of customer default and assigns a
    corresponding risk score.
    """)