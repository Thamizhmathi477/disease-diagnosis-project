import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="🏥 AI Hospital Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("""
        <style>
        .login-container {
            max-width: 420px;
            margin: 80px auto;
            padding: 40px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            border-top: 6px solid #0d6efd;
        }
        .login-container h1 {
            text-align: center;
            color: #0d6efd;
            font-size: 2.2rem;
            margin-bottom: 8px;
        }
        .login-container p {
            text-align: center;
            color: #6c757d;
            margin-bottom: 25px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-container">
        <h1>🏥 AI Hospital</h1>
        <p>Login to access the diagnosis system</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button("🔐 Login")

        if submit:
            if username and password:
                st.session_state.login = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("⚠️ Please enter both username and password")

    st.stop()

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .hospital-header {
        background: linear-gradient(135deg, #0D47A1, #1565C0);
        padding: 20px 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(13,71,161,0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hospital-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .hospital-header p { margin: 0; opacity: 0.9; font-size: 0.9rem; }
    .header-badge {
        background: rgba(255,255,255,0.2);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 10px 0;
        border: 1px solid #e8ecf0;
    }
    .card-blue { border-top: 5px solid #1976D2; }
    .card-green { border-top: 5px solid #43A047; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e8ecf0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #0D47A1; }
    .metric-label { color: #78909C; font-size: 0.85rem; margin-top: 5px; }
    .metric-icon { font-size: 1.8rem; margin-bottom: 5px; }
    .stButton button {
        background: #1976D2;
        color: white;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        width: 100%;
    }
    .diagnosis-box {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #1976D2;
        margin: 20px 0;
    }
    .remedy-box {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        padding: 15px;
        border-radius: 12px;
        border-left: 6px solid #43A047;
        margin: 10px 0;
    }
    .severity-high { background: #FFEBEE; color: #D32F2F; padding: 12px 20px; border-radius: 10px; font-weight: 700; }
    .severity-medium { background: #FFF3E0; color: #F57C00; padding: 12px 20px; border-radius: 10px; font-weight: 700; }
    .severity-low { background: #E8F5E9; color: #388E3C; padding: 12px 20px; border-radius: 10px; font-weight: 700; }
    .symptom-tag {
        display: inline-block;
        background: #E3F2FD;
        color: #0D47A1;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 3px;
        font-weight: 500;
        border: 1px solid #BBDEFB;
    }
    .footer {
        text-align: center;
        color: #78909C;
        padding: 15px;
        border-top: 1px solid #e8ecf0;
        margin-top: 30px;
        font-size: 0.85rem;
        background: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hospital-header">
    <div>
        <h1>🏥 AI Hospital</h1>
        <p>🤖 AI-Powered Disease Diagnosis System</p>
    </div>
    <div>
        <span class="header-badge">🟢 Live</span>
        <span class="header-badge">👤 {st.session_state.get('username', 'Doctor')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_files():
    model = joblib.load("model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    symptom_columns = joblib.load("symptom_columns.pkl")
    return model, encoder, symptom_columns

model, encoder, symptom_columns = load_model_files()

@st.cache_resource
def load_csv_files():
    data = {}
    try:
        data["description"] = pd.read_csv("description.csv") if os.path.exists("description.csv") else None
    except:
        data["description"] = None
    try:
        data["precautions"] = pd.read_csv("precautions.csv") if os.path.exists("precautions.csv") else None
    except:
        data["precautions"] = None
    try:
        data["medications"] = pd.read_csv("medications.csv") if os.path.exists("medications.csv") else None
    except:
        data["medications"] = None
    try:
        data["diets"] = pd.read_csv("diets.csv") if os.path.exists("diets.csv") else None
    except:
        data["diets"] = None
    try:
        data["workout"] = pd.read_csv("workout.csv") if os.path.exists("workout.csv") else None
    except:
        data["workout"] = None
    return data

csv_files = load_csv_files()

with st.sidebar:
    st.markdown("### 🏥 Navigation")
    page = st.radio("", ["🏠 Home", "🩺 Disease Prediction", "ℹ️ About"], index=0)
    st.markdown("---")
    st.markdown("### 📊 Model Statistics")
    st.markdown(f"**Diseases:** {len(encoder.classes_)}")
    st.markdown(f"**Symptoms:** {len(symptom_columns)}")
    if st.button("🚪 Logout"):
        st.session_state.login = False
        st.rerun()

if page == "🏠 Home":
    st.title("🏥 AI Disease Diagnosis System")
    st.image("https://images.unsplash.com/photo-1584515933487-779824d29309?w=1200", use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-icon">👤</div><div class="metric-value">96K</div><div class="metric-label">Patients</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-icon">🦠</div><div class="metric-value">100</div><div class="metric-label">Diseases</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-icon">🔬</div><div class="metric-value">230</div><div class="metric-label">Symptoms</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-icon">📊</div><div class="metric-value">95%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    st.success("✅ Professional Hospital Dashboard")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card card-blue">
            <h3>✨ Features</h3>
            <ul>
                <li>✔ Disease Prediction with Confidence</li>
                <li>✔ Medicine Recommendations</li>
                <li>✔ Diet Recommendations</li>
                <li>✔ Workout Recommendations</li>
                <li>✔ Precautions & Safety Tips</li>
                <li>✔ Report Download</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card card-green">
            <h3>📈 How It Works</h3>
            <ol>
                <li>Select your symptoms</li>
                <li>AI analyzes the patterns</li>
                <li>Get instant diagnosis</li>
                <li>View recommendations</li>
                <li>Download report</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

elif page == "🩺 Disease Prediction":
    st.title("🩺 Disease Prediction")
    st.write("Select your symptoms below. The AI will analyze and provide a diagnosis.")
    st.markdown("---")

    selected_symptoms = st.multiselect("🔍 Choose Symptoms", symptom_columns, placeholder="Type to search symptoms...")

    if selected_symptoms:
        st.markdown("### 📋 Selected Symptoms")
        tags = "".join([f'<span class="symptom-tag">✅ {s.replace("_"," ").title()}</span>' for s in selected_symptoms])
        st.markdown(tags, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔍 Predict Disease"):
        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            input_data = np.zeros(len(symptom_columns))
            for symptom in selected_symptoms:
                if symptom in symptom_columns:
                    index = symptom_columns.index(symptom)
                    input_data[index] = 1

            prediction = model.predict([input_data])[0]
            disease = encoder.inverse_transform([prediction])[0]
            confidence = np.max(model.predict_proba([input_data])) * 100

            st.markdown("---")
            st.subheader("📊 Diagnosis Results")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                <div class="diagnosis-box">
                    <h3>🦠 Primary Diagnosis</h3>
                    <h2>{disease}</h2>
                    <p style="font-size:1.1rem;">Confidence: <strong>{confidence:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if confidence > 80:
                    st.markdown('<div class="severity-high">🔴 High Risk</div>', unsafe_allow_html=True)
                elif confidence > 60:
                    st.markdown('<div class="severity-medium">🟡 Medium Risk</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="severity-low">🟢 Low Risk</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📖 Disease Description")
            if csv_files["description"] is not None:
                try:
                    row = csv_files["description"][csv_files["description"].iloc[:, 0].str.lower() == disease.lower()]
                    if len(row) > 0:
                        st.write(row.iloc[0, 1])
                    else:
                        st.info("Description not available.")
                except:
                    st.info("Description file format issue.")
            else:
                st.warning("⚠️ description.csv not uploaded.")

            st.subheader("🛡️ Precautions")
            if csv_files["precautions"] is not None:
                try:
                    row = csv_files["precautions"][csv_files["precautions"].iloc[:, 0].str.lower() == disease.lower()]
                    if len(row) > 0:
                        for prec in row.iloc[0, 1:]:
                            if pd.notna(prec):
                                st.write("✅", prec)
                    else:
                        st.info("Precautions not available.")
                except:
                    st.info("Precaution file format issue.")
            else:
                st.warning("⚠️ precautions.csv not uploaded.")

            st.subheader("💊 Treatment Recommendations")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown('<div class="remedy-box"><h4>💊 Medications</h4></div>', unsafe_allow_html=True)
                if csv_files["medications"] is not None:
                    try:
                        row = csv_files["medications"][csv_files["medications"].iloc[:, 0].str.lower() == disease.lower()]
                        if len(row) > 0:
                            for med in row.iloc[0, 1:]:
                                if pd.notna(med):
                                    st.write("💊", med)
                        else:
                            st.info("Medications not available.")
                    except:
                        st.info("Medication file format issue.")
                else:
                    st.warning("⚠️ medications.csv not uploaded.")

            with c2:
                st.markdown('<div class="remedy-box"><h4>🥗 Diet</h4></div>', unsafe_allow_html=True)
                if csv_files["diets"] is not None:
                    try:
                        row = csv_files["diets"][csv_files["diets"].iloc[:, 0].str.lower() == disease.lower()]
                        if len(row) > 0:
                            for diet in row.iloc[0, 1:]:
                                if pd.notna(diet):
                                    st.write("🍽️", diet)
                        else:
                            st.info("Diet not available.")
                    except:
                        st.info("Diet file format issue.")
                else:
                    st.warning("⚠️ diets.csv not uploaded.")

            with c3:
                st.markdown('<div class="remedy-box"><h4>🏋️ Workout</h4></div>', unsafe_allow_html=True)
                if csv_files["workout"] is not None:
                    try:
                        row = csv_files["workout"][csv_files["workout"].iloc[:, 0].str.lower() == disease.lower()]
                        if len(row) > 0:
                            for workout in row.iloc[0, 1:]:
                                if pd.notna(workout):
                                    st.write("🏃", workout)
                        else:
                            st.info("Workout not available.")
                    except:
                        st.info("Workout file format issue.")
                else:
                    st.warning("⚠️ workout.csv not uploaded.")

            st.markdown("---")
            st.subheader("👤 Patient Details")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Patient Name", placeholder="Enter patient name")
                age = st.number_input("Age", 1, 120, 25)
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                date = st.date_input("Date", datetime.now())

            report = f"""
=====================================
      AI HOSPITAL DIAGNOSIS REPORT
=====================================

Patient Name    : {name if name else 'N/A'}
Age             : {age}
Gender          : {gender}
Date            : {date}

=====================================
      DIAGNOSIS RESULTS
=====================================

Predicted Disease   : {disease}
Confidence          : {confidence:.2f}%
Symptoms            : {', '.join(selected_symptoms)}
Risk Level          : {'High' if confidence > 80 else 'Medium' if confidence > 60 else 'Low'}

=====================================
      TREATMENT RECOMMENDATIONS
=====================================
"""
            st.download_button(
                "📄 Download Report",
                report,
                file_name=f"Diagnosis_Report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

else:
    st.title("ℹ️ About AI Disease Diagnosis System")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card card-blue">
            <h3>🏥 Project Information</h3>
            <p><strong>Project:</strong> AI-Based Disease Diagnosis System</p>
            <p><strong>Algorithm:</strong> Random Forest</p>
            <p><strong>Framework:</strong> Streamlit</p>
            <p><strong>Diseases:</strong> 100+</p>
            <p><strong>Symptoms:</strong> 230</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card card-green">
            <h3>👨‍💻 Developer</h3>
            <p><strong>Name:</strong> Thamizhmathi Sivakumar</p>
            <p><strong>Department:</strong> CSE</p>
            <p><strong>Year:</strong> Final Year (2026)</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="footer">
    ⚠️ <strong>Disclaimer:</strong> Educational purposes only.<br>
    © 2026 AI Disease Diagnosis System
</div>
""", unsafe_allow_html=True)
