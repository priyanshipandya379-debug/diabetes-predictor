import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time

@st.cache_resource
def load_model():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
               'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(url, names=columns)
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        df[col] = df[col].replace(0, df[col].median())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #ffffff !important; }
    .stApp * { color: #000000; }
    h1 { color: #0077b6 !important; text-align: center !important; }
    h2 { color: #0077b6 !important; }
    h3 { color: #0096c7 !important; }
    [data-testid="stSubheader"] { color: #0077b6 !important; }
    label { color: #000000 !important; font-weight: bold !important; }
    p { color: #000000 !important; }
    div.streamlit-expanderContent p {
        color: #000000 !important;
    }
    
    /* NAYA CODE YAHAN */
    [data-testid="stSlider"] label {
        color: #000000 !important;
        font-size: 1rem !important;
        font-weight: bold !important;
    }
    [data-testid="stSlider"] p {
        color: #000000 !important;
        font-size: 1rem !important;
        font-weight: bold !important;
    }
    .streamlit-expanderHeader {
        color: #000000 !important;
        font-weight: bold !important;
        background-color: #e0f4ff !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent {
        background-color: #ffffff !important;
    }
    .streamlit-expanderContent p {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<h1 style="color:#0077b6; text-align:center;">
🩺 Diabetes Prediction System
</h1>
<h3 style="color:#0096c7; text-align:center;">
your Health Matter For Us
</h3>
<hr style="border:1px solid #0077b6;">
""", unsafe_allow_html=True)

# ===== SLIDERS =====
col1, col2 = st.columns(2)
with col1:
    st.markdown('<p style="color:#0077b6; font-size:1.2rem; font-weight:bold;">👤 Patient Information</p>', unsafe_allow_html=True)
    pregnancies = st.slider("🤰 Pregnancies",       0,    17,   2)
    glucose     = st.slider("🩸 Glucose Level",     50,   250,  120)
    blood_press = st.slider("💓 Blood Pressure",    30,   130,  80)
    skin_thick  = st.slider("📏 Skin Thickness",    5,    100,  25)

with col2:
    st.markdown('<p style="color:#0077b6; font-size:1.2rem; font-weight:bold;">🔬 Medical Values</p>', unsafe_allow_html=True)
    insulin     = st.slider("💉 Insulin Level",     10,   900,  85)
    bmi         = st.slider("⚖️ BMI",               10.0, 70.0, 28.5)
    dpf         = st.slider("🧬 Diabetes Pedigree", 0.0,  2.5,  0.45)
    age         = st.slider("🎂 Age",               10,   100,  35)

st.markdown('<hr style="border:1px solid #0077b6;">', unsafe_allow_html=True)

# ===== PREDICT BUTTON =====
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    with st.spinner("🔄 Analyzing patient data..."):
        time.sleep(1)

    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_press,
        skin_thick, insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies','Glucose','BloodPressure',
        'SkinThickness','Insulin','BMI',
        'DiabetesPedigreeFunction','Age'
    ])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.markdown('<hr style="border:1px solid #0077b6;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#0077b6;">📊 Prediction Result</h2>', unsafe_allow_html=True)

    # ===== DIABETIC =====
    if prediction == 1:
        risk = "HIGH" if probability[1] > 0.7 else "MODERATE"

        # Main Result Box
        st.markdown(f"""
        <div style="background-color:#ff4444; color:white; padding:25px;
        border-radius:15px; text-align:center; font-size:2rem;
        font-weight:bold; margin-bottom:15px;">
        🔴 DIABETIC — {risk} RISK
        </div>
        """, unsafe_allow_html=True)

        # Type 1 or Type 2
        if age < 30 and bmi < 25:
            st.markdown("""
            <div style="background-color:#e65c00; color:white; padding:20px;
            border-radius:15px; text-align:center; margin-bottom:15px;">
            <p style="font-size:1.5rem; font-weight:bold; margin:0;">
            🔸 TYPE 1 DIABETES</p>
            <p style="font-size:1rem; margin:5px 0 0 0;">
            Immune system related — See doctor immediately!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:#cc5500; color:white; padding:20px;
            border-radius:15px; text-align:center; margin-bottom:15px;">
            <p style="font-size:1.5rem; font-weight:bold; margin:0;">
            🔶 TYPE 2 DIABETES</p>
            <p style="font-size:1rem; margin:5px 0 0 0;">
            Lifestyle related — Diet and exercise needed!</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability Boxes
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
            <div style="background-color:#fff0f0; border:3px solid #ff4444;
            border-radius:15px; padding:20px; text-align:center;
            margin-bottom:10px;">
            <p style="color:#000000; font-size:1rem;
            font-weight:bold; margin:0;">🔴 Diabetes Probability</p>
            <p style="color:#ff4444; font-size:2.5rem;
            font-weight:bold; margin:0;">{probability[1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style="background-color:#f0fff0; border:3px solid #28a745;
            border-radius:15px; padding:20px; text-align:center;
            margin-bottom:10px;">
            <p style="color:#000000; font-size:1rem;
            font-weight:bold; margin:0;">🟢 No Diabetes Probability</p>
            <p style="color:#28a745; font-size:2.5rem;
            font-weight:bold; margin:0;">{probability[0]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # ===== NON DIABETIC =====
    else:
        # Main Result Box
        st.markdown("""
        <div style="background-color:#28a745; color:white; padding:25px;
        border-radius:15px; text-align:center; font-size:2rem;
        font-weight:bold; margin-bottom:15px;">
        🟢 NON-DIABETIC — LOW RISK
        </div>
        """, unsafe_allow_html=True)

        # Borderline or Healthy
        if probability[1] > 0.35:
            st.markdown("""
            <div style="background-color:#ffc107; color:#000000; padding:20px;
            border-radius:15px; text-align:center; margin-bottom:15px;">
            <p style="font-size:1.5rem; font-weight:bold; margin:0;">
            ⚠️ BORDERLINE RISK</p>
            <p style="font-size:1rem; margin:5px 0 0 0;">
            Monitor glucose regularly!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:#20c997; color:white; padding:20px;
            border-radius:15px; text-align:center; margin-bottom:15px;">
            <p style="font-size:1.5rem; font-weight:bold; margin:0;">
            ✅ PERFECTLY HEALTHY</p>
            <p style="font-size:1rem; margin:5px 0 0 0;">
            Keep up the good work!</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

        # Probability Boxes
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
            <div style="background-color:#f0fff0; border:3px solid #28a745;
            border-radius:15px; padding:20px; text-align:center;
            margin-bottom:10px;">
            <p style="color:#000000; font-size:1rem;
            font-weight:bold; margin:0;">🟢 No Diabetes Probability</p>
            <p style="color:#28a745; font-size:2.5rem;
            font-weight:bold; margin:0;">{probability[0]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style="background-color:#fff0f0; border:3px solid #ff4444;
            border-radius:15px; padding:20px; text-align:center;
            margin-bottom:10px;">
            <p style="color:#000000; font-size:1rem;
            font-weight:bold; margin:0;">🔴 Diabetes Probability</p>
            <p style="color:#ff4444; font-size:2.5rem;
            font-weight:bold; margin:0;">{probability[1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # ===== PATIENT SUMMARY =====
    st.markdown('<hr style="border:1px solid #0077b6;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color:#e0f4ff; border:2px solid #0077b6;
    border-radius:15px; padding:20px;">
    <p style="color:#0077b6; font-size:1.3rem;
    font-weight:bold; margin-bottom:10px;">📋 Patient Summary</p>
    <p style="color:#000000; font-size:1rem; margin:5px 0;">
    🩸 <b>Glucose</b> : {glucose} mg/dL</p>
    <p style="color:#000000; font-size:1rem; margin:5px 0;">
    💓 <b>Blood Pressure</b> : {blood_press} mm Hg</p>
    <p style="color:#000000; font-size:1rem; margin:5px 0;">
    ⚖️ <b>BMI</b> : {bmi}</p>
    <p style="color:#000000; font-size:1rem; margin:5px 0;">
    🎂 <b>Age</b> : {age} years</p>
    <p style="color:#000000; font-size:1rem; margin:5px 0;">
    💉 <b>Insulin</b> : {insulin}</p>
    </div>
    """, unsafe_allow_html=True)

    # ===== HEALTH GUIDE =====
    st.markdown('<hr style="border:1px solid #0077b6;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#0077b6;">🥗 Health Guide</h2>',
    unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("""
        <div style="background-color:#ff4444; color:white; padding:15px;
        border-radius:10px; font-weight:bold; font-size:1.1rem;
        margin-bottom:15px;">
        ⚠️ Please consult a doctor immediately!
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🥗 Foods to EAT ✅"):
            st.markdown('<p style="color:#000000 !important;">✅ <b>Karela</b> — controls blood sugar</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ <b>Methi</b> — reduces glucose</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ <b>Jamun</b> — lowers sugar levels</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ <b>Leafy greens</b> — spinach, palak</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ <b>Whole grains</b> — brown rice, oats</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ <b>Nuts</b> — almonds, walnuts</p>', unsafe_allow_html=True)

        with st.expander("🚫 Foods to AVOID ❌"):
            st.markdown('<p style="color:#000000 !important;">❌ <b>White rice</b>, maida, white bread</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">❌ <b>Sugar</b>, mithai, cold drinks</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">❌ <b>Fried foods</b> — samosa, pakoda</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">❌ <b>Alcohol</b> and smoking</p>', unsafe_allow_html=True)

        with st.expander("🌿 Home Remedies"):
            st.markdown('<p style="color:#000000 !important;">🌱 <b>Karela juice</b> — 30ml every morning</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">🌱 <b>Methi seeds</b> — soak overnight</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">🌱 <b>Jamun seed powder</b> — 1 tsp daily</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">🌱 <b>Amla juice</b> — 20ml daily morning</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">🌱 <b>Cinnamon</b> — half tsp warm water</p>', unsafe_allow_html=True)

        with st.expander("💊 Medicines — Doctor Only!"):
            st.markdown('<p style="color:#000000 !important;">• <b>Metformin</b> — most common first medicine</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">• <b>Glipizide</b> — stimulates insulin release</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">• <b>Sitagliptin</b> — controls sugar after meal</p>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background-color:#fff3cd; color:#856404 !important;
            padding:10px; border-radius:8px; font-weight:bold;">
            ⚠️ NEVER take medicines without doctor advice!
            </div>""", unsafe_allow_html=True)

        with st.expander("🏃 Lifestyle Tips"):
            st.markdown('<p style="color:#000000 !important;">• <b>Walk</b> 30 minutes daily after meals</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">• <b>Yoga</b> — Surya Namaskar daily</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">• <b>Sleep</b> 7-8 hours every night</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">• <b>Drink</b> 2-3 litres water daily</p>', unsafe_allow_html=True)

    else:
        with st.expander("💪 Prevention Tips"):
            st.markdown('<p style="color:#000000 !important;">✅ Maintain healthy weight — BMI 18.5 to 24.9</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ Exercise 30 min daily</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ Eat more vegetables and fruits</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ Avoid sugar and processed food</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#000000 !important;">✅ Drink 2-3 litres of water daily</p>', unsafe_allow_html=True)

st.markdown('<hr style="border:1px solid #0077b6;">', unsafe_allow_html=True)
st.markdown('<p style="color:#666666; text-align:center; font-size:0.9rem;">⚠️ Educational purposes only. Not a substitute for medical advice.</p>', unsafe_allow_html=True)
