# 🏦 Credit Risk Intelligence Platform

## Overview

The Credit Risk Intelligence Platform is an end-to-end Machine Learning application built using the Home Credit Default Risk dataset.

The platform predicts the probability of loan default, provides risk scoring, enables model evaluation, and allows users to query the dataset using natural language through Gemini AI-powered SQL generation.

The application is designed as a complete ML solution with:

* Data Preprocessing Pipeline
* LightGBM Risk Prediction Model
* Model Evaluation Dashboard
* Natural Language to SQL Querying
* Streamlit Interactive UI
* Dockerized Deployment Support

---

# Problem Statement

Financial institutions need reliable methods to assess the risk associated with loan applicants.

This project predicts whether a customer is likely to default on a loan using historical customer information and provides an intelligent interface for exploring the data.

---

# Dataset

Dataset Used:

**Home Credit Default Risk Dataset**

Target Variable:

| Value | Meaning                  |
| ----- | ------------------------ |
| 0     | Customer did not default |
| 1     | Customer defaulted       |

Dataset Statistics:

* Records: 307,511
* Features: 122
* Target Column: TARGET

---

# Key Features

## 1. Credit Risk Prediction

Predicts:

* Default Probability
* Risk Score
* Risk Band

Risk Categories:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

---

## 2. Talk To Data

Users can ask questions in plain English.

Examples:

* How many customers defaulted?
* What is the average income of customers who defaulted?
* Show top 5 highest credit amounts.

Workflow:

Question
↓
Gemini AI
↓
SQL Generation
↓
SQLite Execution
↓
Results

---

## 3. Model Performance Dashboard

Provides:

* ROC AUC
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Technology Stack

## Backend

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM

## Frontend

* Streamlit
* Plotly
* Matplotlib

## AI Integration

* Google Gemini API

## Data Query Layer

* SQLite

## Deployment

* Docker
* Docker Compose

---

# Project Structure

```text
credit-risk-platform/

├── app.py

├── data/
│   ├── application_train.csv
│   └── application_test.csv

├── documents/

├── models/
│   ├── lgbm_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.json
│   └── model_metadata.json

├── notebooks/
│   └── eda.py

├── sql/

├── src/
│
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│
│   ├── ml/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│
│   ├── talk_to_data/
│   │   ├── prompt_templates.py
│   │   ├── nl_to_sql.py
│   │   ├── query_runner.py
│   │   └── chat_with_data.py
│
│   ├── utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── helpers.py
│   │   └── docker_utils.py
│
│   └── __init__.py

├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

# Machine Learning Pipeline

## Data Loading

Dataset loading and validation.

## Data Preprocessing

The preprocessing pipeline performs:

* Missing Value Imputation
* Categorical Encoding
* Feature Cleaning
* Data Transformation

## Model Training

Algorithm:

**LightGBM Classifier**

Advantages:

* Fast Training
* High Accuracy
* Handles Large Datasets Efficiently
* Feature Importance Support

---

# Model Performance

| Metric    | Value |
| --------- | ----- |
| ROC AUC   | 0.760 |
| Accuracy  | 70.7% |
| Precision | 16.9% |
| Recall    | 67.4% |
| F1 Score  | 27.0% |

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd credit-risk-platform
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key
```

A template is provided:

```text
.env.example
```

---

# Dataset Setup

Download the Home Credit Default Risk dataset and place:

```text
application_train.csv
application_test.csv
```

inside:

```text
data/
```

---

# Model Training (Required Before Running App)

The trained model files are not included in the repository.

Generate them by running:

```bash
python -m src.ml.train
```

This creates:

```text
models/
├── lgbm_model.pkl
├── preprocessor.pkl
├── feature_names.json
└── model_metadata.json
```

---

# Running the Application

Launch Streamlit:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Example Questions for Talk To Data

Try:

```text
How many customers defaulted?
```

```text
What is the average income of customers who defaulted?
```

```text
Show top 5 highest credit amounts.
```

```text
What is the maximum credit amount?
```

```text
What is the average income of customers?
```

---

# Docker Deployment

## Build Image

```bash
docker build -t credit-risk-platform .
```

## Run Container

```bash
docker run -p 8501:8501 credit-risk-platform
```

## Using Docker Compose

```bash
docker compose up --build
```

Application:

```text
http://localhost:8501
```

---

# Future Enhancements

* SHAP Explainability Dashboard
* Cloud Deployment
* Automated Model Retraining
* REST API Integration
* User Authentication
* Real-Time Credit Scoring

---

# Author

Credit Risk Intelligence Platform

Developed using Machine Learning, Streamlit, Gemini AI, SQLite, and Docker as part of a Credit Risk Assessment and Data Intelligence solution.
