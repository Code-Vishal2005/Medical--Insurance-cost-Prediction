import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(
    page_title="🏥 Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
)

# -----------------------------------------------------
# CUSTOM CSS - PREMIUM DEEP BLUE THEME
# -----------------------------------------------------
st.markdown("""
    <style>
    /* Background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #001F3F 0%, #001233 50%, #000814 100%);
        color: #F1FAEE !important;
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001845, #001233);
        color: white;
    }

    /* Main Title */
    .title {
        text-align: center;
        color: #00B4D8;
        font-size: 50px;
        font-weight: 900;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        text-shadow: 2px 2px 8px #023E8A;
    }

    .subtitle {
        text-align: center;
        color: #ADE8F4;
        font-size: 18px;
        margin-bottom: 30px;
        font-weight: 500;
    }

    /* Stylish Divider */
    hr {
        border: 1px solid #00B4D8;
        border-radius: 5px;
        margin: 25px 0;
    }

    /* Result Box */
    .result-box {
        background: linear-gradient(145deg, #023E8A, #0077B6);
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
        transition: transform 0.2s ease;
    }

    .result-box:hover {
        transform: scale(1.05);
    }

    /* Info Card */
    .info-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-left: 5px solid #00B4D8;
        padding: 15px;
        margin-top: 15px;
        border-radius: 10px;
        font-size: 16px;
        color: #CAF0F8;
    }

    label, .stNumberInput label, .stSelectbox label {
        color: #CAF0F8 !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00B4D8, #0077B6);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 12px 40px;
        font-weight: 600;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 12px rgba(0,180,216,0.3);
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #48CAE4, #00B4D8);
        transform: scale(1.05);
    }

    /* Sidebar Enhancements */
    .sidebar-title {
        color: #90E0EF;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
    }

    .sidebar-section {
        background-color: rgba(255,255,255,0.05);
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 10px;
        color: #CAF0F8;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: #B0BEC5;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR DESIGN
# -----------------------------------------------------
st.sidebar.markdown("<h1 class='sidebar-title'>🎀 Medical Insurance Predictor</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class='sidebar-section'>
✨ <b>Welcome!</b><br>
This AI-powered app predicts your <b>Medical Insurance Cost</b> 💰 using ML.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class='sidebar-section'>
🧠 <b>How to Use:</b><br>
1️⃣ Fill all input fields<br>
2️⃣ Click <b>Predict Now 🚀</b><br>
3️⃣ Instantly view your cost prediction
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class='sidebar-section'>
📊 <b>Dataset:</b> Trained on <i>insurance.csv</i><br>
⚙️ <b>Model:</b> Linear Regression
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------
model_path = r"E:\Machine Learning Projects\insurance_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.sidebar.success("✅ Model Loaded Successfully!")
else:
    st.sidebar.error("❌ Model not found. Please check the path.")
    st.stop()

# -----------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------
st.markdown("<h1 class='title'>💰 Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Fill in your details to estimate your insurance cost instantly ⚡</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------------------------------
# INPUT SECTION - TWO COLUMNS × SIX ROWS
# -----------------------------------------------------
st.markdown("### 🧾 Enter Your Details Below")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Enter Your Age", min_value=0, max_value=120, value=30)
    bmi = st.number_input("⚖️ Enter Your BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.5, step=0.1)
    children = st.number_input("👶 Enter Number of Dependents", min_value=0, max_value=10, value=1)

with col2:
    sex = st.selectbox("🧬 Gender", options=["Male", "Female"])
    smoker = st.selectbox("🚬 Smoker", options=["Yes", "No"])
    region = st.selectbox("🌍 Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------------------------------
# ENCODING FUNCTION
# -----------------------------------------------------
def encode_input(age, sex, bmi, children, smoker, region):
    sex_encoded = 1 if sex.lower() == "male" else 0
    smoker_encoded = 1 if smoker.lower() == "yes" else 0
    region_encoding = {
        "northeast": [1, 0, 0, 0],
        "northwest": [0, 1, 0, 0],
        "southeast": [0, 0, 1, 0],
        "southwest": [0, 0, 0, 1],
    }
    region_encoded = region_encoding.get(region.lower(), [0, 0, 0, 0])
    return [age, sex_encoded, bmi, children, smoker_encoded] + region_encoded

# -----------------------------------------------------
# PREDICTION SECTION
# -----------------------------------------------------
st.markdown("### 🚀 Predict Your Cost")

if st.button("🔮 Predict Now"):
    try:
        input_data = np.array([encode_input(age, sex, bmi, children, smoker, region)])
        prediction = model.predict(input_data)[0]

        st.markdown(
            f"<div class='result-box'>💵 <b>Estimated Medical Cost:</b><br><br> 🩺 ${prediction:,.2f}</div>",
            unsafe_allow_html=True,
        )

        st.balloons()

        # Input Summary
        st.markdown("### 📋 Input Summary")
        st.markdown(f"""
        <div class='info-card'>
        👤 <b>Age:</b> {age} years <br>
        ⚖️ <b>BMI:</b> {bmi} <br>
        👶 <b>Children:</b> {children} <br>
        🧬 <b>Gender:</b> {sex} <br>
        🚬 <b>Smoker:</b> {smoker} <br>
        🌍 <b>Region:</b> {region}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------------------------------
# 🌟 BEAUTIFUL FOOTER SECTION
# -----------------------------------------------------




st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
    <div style="
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(90deg, #001233, #001F3F, #003566);
        border-radius: 15px;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
        margin-top: 40px;
        color: #E0F7FA;
    ">
        <h3 style="color: #00B4D8; font-weight: 800; margin-bottom: 10px;">
            💻 Built by Vishal Kumar ❤️ using Streamlit
        </h3>
        <p style="font-size: 16px; color: #CAF0F8; margin: 0;">
            👨‍💻 Developed by <b style="color:#90E0EF;">Vishal Kumar</b>  
            <br>
            📊 Powered by <i style="color:#ADE8F4;">insurance.csv</i> dataset
        </p>
        <div style="margin-top: 15px;">
            <a href='https://github.com/' target='_blank' style='text-decoration:none; margin: 0 10px;'>
                🌐 <span style='color:#48CAE4;'>GitHub</span>
            </a> |
            <a href='https://www.linkedin.com/' target='_blank' style='text-decoration:none; margin: 0 10px;'>
                💼 <span style='color:#48CAE4;'>LinkedIn</span>
            </a> |
            <a href='mailto:vishalkumar@gmail.com' style='text-decoration:none; margin: 0 10px;'>
                📧 <span style='color:#48CAE4;'>Contact Me</span>
            </a>
        </div>
        <p style="font-size: 12px; color: #B0BEC5; margin-top: 25px;">
            © 2025 Vishal Kumar | All Rights Reserved 🌍
        </p>
    </div>
""", unsafe_allow_html=True)
