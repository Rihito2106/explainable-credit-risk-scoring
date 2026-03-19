# Explainable Credit Risk Scoring Dashboard

## Project Overview
In the modern financial ecosystem, deploying opaque "black-box" AI models is no longer sufficient. Regulatory bodies and stakeholders require transparency in every decision. 

This project is an end-to-end, interactive machine learning web application that predicts credit risk while providing clear, human-understandable explanations using **Explainable AI (XAI)**.

Users can input an applicant's financial profile and instantly receive:
* **A Probability Score** indicating credit risk (Approved vs. Declined).
* **A SHAP Waterfall Plot** showing exactly how each feature influenced the prediction.
* **A Downloadable PDF Audit Report** summarizing the decision for stakeholders.

---

## Tech Stack
* **Machine Learning:** XGBoost Classifier (`xgboost`)
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Frontend UI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Automated Reporting:** FPDF
* **Model Persistence:** Joblib

---

## Key Features
1. **Interactive UI:** Real-time adjustments using Streamlit sliders and dropdowns.
2. **Instant Inference:** Dynamic encoding via pre-trained `scikit-learn` LabelEncoders.
3. **Transparency:** Local SHAP explanations breaking down the exact push-and-pull of features (e.g., "Checking Account Status").
4. **Exportable Reports:** One-click generation of a professional PDF detailing the risk factors.

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Rihito2106/explainable-credit-risk-scoring.git
cd explainable-credit-risk-scoring
```

### 2. Environment Setup (Important!)
To avoid metadata compatibility issues between newer versions of XGBoost and SHAP, this project relies on a specific Conda environment:
```bash
conda create -n credit_env python=3.10 -y
conda activate credit_env
pip install -r requirements.txt
```

### 3. Launch the App
```bash
streamlit run app.py
```

---

## Technical Challenges Overcome
* **Dependencies:** Diagnosed and resolved a complex compatibility bug between XGBoost's JSON tree metadata and SHAP's TreeExplainer by isolating the project in a strict Conda environment using stable LTS versions (`xgboost==2.0.3`, `shap==0.45.1`).
* **Model Persistence:** Engineered a clean artifact-loading pipeline to ensure the Streamlit frontend can seamlessly deserialize the trained model and multiple categorical encoders without state-loss.
