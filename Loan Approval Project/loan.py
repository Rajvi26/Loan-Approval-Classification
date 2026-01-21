import streamlit as st
import pandas as pd
import joblib
import base64   # ✅ only extra import

st.set_page_config(page_title="Loan Prediction App", layout="centered")

# ================= LOGO (LEFT TOP YELLOW BOX) =================
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("data_vidwan_logo.png")

st.markdown(
    f"""
    <style>
    .dv-logo {{
        position: fixed;
        top: 95px;     /* yellow box vertical position */
        left: 20px;    /* yellow box horizontal position */
        z-index: 9999;
    }}
    </style>

    <div class="dv-logo">
        <img src="data:image/png;base64,{logo_base64}" width="250">
    </div>
    """,
    unsafe_allow_html=True
)
# =============================================================

st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: white;
        color: black;
    }

    /* Text color black */
    h1, h2, h3, h4, h5, h6, p, label {
        color: black !important;
    }

    /* Input boxes background */
    input, textarea {
        background-color: #f5f5f5 !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Loan Approval Prediction")
st.write("Random Forest Model Inference")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()
st.success("✅ Model loaded successfully")

# ---------------- INPUTS ----------------
st.subheader("🧾 Enter Applicant Details")

age = st.number_input("Age", 18, 70, 30)
annual_income = st.number_input("Annual Income", 0, value=500000)
loan_amount = st.number_input("Loan Amount", 0, value=200000)
credit_score = st.slider("Credit Score", 300, 900, 650)
interest_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 10.0)
dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])

employment_type = st.selectbox(
    "Employment Type",
    ["Salaried", "Self-Employed", "Unemployed"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Post Graduate", "Under Graduate"]
)

default_history = st.selectbox(
    "Default History",
    ["Yes", "No"]
)

employment_length = st.selectbox(
    "Employment Length",
    ["01-Mar", "03-May", "05-Oct", "10+ Years"]
)

# ---------------- RAW INPUT ----------------
raw_input = pd.DataFrame({
    "age": [age],
    "annual_income": [annual_income],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "interest_rate": [interest_rate],
    "dependents": [dependents],
    "employment_type": [employment_type],
    "education": [education],
    "default_history": [default_history],
    "employment_length": [employment_length]
})

# ---------------- ALIGN FEATURES ----------------
raw_encoded = pd.get_dummies(raw_input)

final_input = pd.DataFrame(
    0,
    index=[0],
    columns=model.feature_names_in_
)

for col in raw_encoded.columns:
    if col in final_input.columns:
        final_input[col] = raw_encoded[col].values

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(final_input)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
