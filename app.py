import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64


# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Credit Risk Explainer", layout="wide")
st.title("🏦 Explainable AI: Credit Risk Scoring")
st.markdown("This tool predicts credit risk and uses **SHAP** to explain exactly *why* a decision was made.")

# --- 2. LOAD MODELS & ENCODERS ---
# We cache this so it only loads once when the app starts
@st.cache_resource
@st.cache_resource
def load_artifacts():
    model = xgb.Booster()
    # LOAD THE NEW FILENAME HERE
    model.load_model("models/xgb_constrained_model.json") 
    
    # We keep this as a 'safety belt' just in case
    model.set_param({"base_score": 0.5}) 
    
    # 3. Load the encoders
    encoders = {
        "Sex": joblib.load("models/Sex_encoder.pkl"),
        "Housing": joblib.load("models/Housing_encoder.pkl"),
        "Saving accounts": joblib.load("models/Saving accounts_encoder.pkl"),
        "Checking account": joblib.load("models/Checking account_encoder.pkl")
    }
    return model, encoders

bst_cons, encoders = load_artifacts()

# --- 3. SIDEBAR USER INPUTS ---
st.sidebar.header("Applicant Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 30)
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    job = st.sidebar.selectbox("Job (0=Unskilled/Non-resident, 3=Highly Skilled)", [0, 1, 2, 3], index=2)
    housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])
    saving_accounts = st.sidebar.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "NaN"])
    checking_account = st.sidebar.selectbox("Checking account", ["little", "moderate", "rich", "NaN"])
    credit_amount = st.sidebar.number_input("Credit amount ($)", 100, 20000, 2500)
    duration = st.sidebar.slider("Duration (months)", 6, 72, 24)
    
    # Create dictionary matching exact column names
    data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. PREPROCESS INPUTS ---
# Make a copy for encoding
encoded_df = input_df.copy()

# Apply the loaded encoders to the categorical columns
for col in encoders.keys():
    # If the user selected NaN string, we might need to handle it depending on how you trained it
    # Assuming your encoder learned these exact string categories during prep
    encoded_df[col] = encoders[col].transform(encoded_df[col])

# --- 5. PREDICTION & EXPLANATION ---
col1, col2 = st.columns([1, 2])

# Convert to DMatrix for native XGBoost prediction
dmatrix = xgb.DMatrix(encoded_df)
prediction_prob = bst_cons.predict(dmatrix)[0]

with col1:
    st.subheader("Prediction")
    st.write("Based on the financial profile:")
    
    if prediction_prob > 0.5:
        st.success(f"✅ **GOOD RISK**")
        st.metric(label="Approval Probability", value=f"{prediction_prob * 100:.1f}%")
    else:
        st.error(f"❌ **BAD RISK**")
        st.metric(label="Approval Probability", value=f"{prediction_prob * 100:.1f}%")

with col2:
    st.subheader("Why was this decision made?")
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(bst_cons)
    shap_values = explainer(encoded_df)
    
    # Generate Waterfall Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)


# --- Automated Insights Section ---
st.markdown("---")
st.subheader("📋 Key Insights for this Applicant")

# We can pull the top positive and negative contributors automatically
# shap_values[0].values contains the "impact" of each feature
top_features = pd.Series(shap_values[0].values, index=encoded_df.columns).sort_values()

# 1. Main Negative Factor (Biggest Blue Bar)
worst_feature = top_features.index[0]
worst_val = top_features.iloc[0]

# 2. Main Positive Factor (Biggest Red Bar)
best_feature = top_features.index[-1]
best_val = top_features.iloc[-1]

col_a, col_b = st.columns(2)

with col_a:
    st.error(f"**Primary Risk Factor:** {worst_feature}")
    st.write(f"This factor pushed the risk score down by **{abs(worst_val):.2f}**. Improving this status would most likely change the decision.")

with col_b:
    st.success(f"**Primary Strength:** {best_feature}")
    st.write(f"This factor added **{best_val:.2f}** to the approval score, helping balance out other risk factors.")

with st.expander("💡 How to read this chart?"):
    st.write("""
    * **The Center Line:** Represents the average risk score for all applicants.
    * **Blue Bars (Left):** Factors that made the applicant look **riskier** to the model.
    * **Red Bars (Right):** Factors that increased the model's **confidence** in the applicant.
    * **The Top Number (f(x)):** The final calculated score. High scores = Low Risk.
    """)



def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download Report</a>'

# Inside your app, after the plot is generated:
if st.button("Generate Professional Report"):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header & Branding
    pdf.set_fill_color(30, 30, 30)  # Dark theme header
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "CREDIT RISK AUDIT REPORT", ln=True, align='C')
    
    # 2. Summary Section
    pdf.set_text_color(0, 0, 0)
    pdf.ln(25)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Decision: {'APPROVED' if prediction_prob > 0.5 else 'DECLINED'}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Approval Probability: {prediction_prob*100:.1f}%", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Divider
    
    # 3. Applicant Profile
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Applicant Profile:", ln=True)
    pdf.set_font("Arial", "", 11)
    # This pulls data from your encoded_df/input data
    for col in input_df.columns:
        pdf.cell(100, 8, f"{col}: {input_df[col].values[0]}", ln=True)
    
    # 4. Explainable AI (SHAP) Insights
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Key Decision Drivers (XAI Analysis):", ln=True)
    pdf.set_font("Arial", "", 11)
    
    # Add the automated insights we calculated
    pdf.multi_cell(0, 8, f"- Primary Risk Factor: {worst_feature}. This had a negative impact of {abs(worst_val):.2f}.")
    pdf.multi_cell(0, 8, f"- Primary Strength: {best_feature}. This added {best_val:.2f} to the score.")
    
    # Footer
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, "This report was generated using an XGBoost Classifier and SHAP Explainer.", align='C')

    # Output
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Detailed_Credit_Report")
    st.markdown(html, unsafe_allow_html=True)