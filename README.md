Explainable Credit Risk Scoring Dashboard
roject Overview

In the modern financial ecosystem, deploying opaque “black-box” AI models is no longer sufficient. Regulatory bodies and stakeholders require transparency in every decision.

This project is an end-to-end, interactive machine learning web application that predicts credit risk while providing clear, human-understandable explanations using Explainable AI (XAI).

Users can input an applicant’s financial profile and instantly receive:

A probability score indicating credit risk (Approved vs. Declined)

A SHAP waterfall plot showing how each feature influenced the prediction

A downloadable PDF audit report summarizing the decision

Tech Stack

Machine Learning: XGBoost Classifier (xgboost)

Explainable AI: SHAP (SHapley Additive exPlanations)

Frontend: Streamlit

Data Processing: Pandas, NumPy

Reporting: FPDF

Model Persistence: Joblib

Key Features

Interactive UI with real-time inputs using Streamlit sliders and dropdowns

Instant inference with dynamic encoding via pre-trained LabelEncoders

Transparent predictions using SHAP for feature-level explanations

One-click PDF generation for audit-ready reports

How to Run Locally
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/explainable-credit-risk-scoring.git
cd explainable-credit-risk-scoring
2. Environment Setup (Important)

Due to compatibility issues between newer versions of XGBoost and SHAP, this project relies on specific library versions to avoid runtime errors.

It is strongly recommended to use a dedicated Conda environment with Python 3.10:

conda create -n credit_env python=3.10 -y
conda activate credit_env
pip install -r requirements.txt

Key dependencies:

xgboost==2.0.3

shap==0.45.1

numpy==1.26.4

Launch the Application
streamlit run app.py
📁 Project Structure
├── app.py                   # Main Streamlit dashboard script
├── requirements.txt         # Locked dependency versions
├── README.md                # Project documentation
└── models/                  # Serialized artifacts
    ├── xgb_constrained_model.json
    ├── Sex_encoder.pkl
    ├── Housing_encoder.pkl
    ├── Checking_account_encoder.pkl
    └── Saving_accounts_encoder.pkl
Technical Challenges Overcome
1. Dependency Compatibility

Resolved version conflicts between XGBoost and SHAP that caused metadata-related errors. This was handled by isolating the environment and locking stable versions.

2. Model Persistence

Designed a robust pipeline for loading serialized models and encoders, ensuring seamless integration with the Streamlit frontend without state inconsistencies.