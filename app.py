import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Train model directly — no pkl files needed!
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
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Prediction System")
st.markdown("### College Project — Machine Learning")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Information")
    pregnancies = st.slider("Pregnancies",       0,    17,   2)
    glucose     = st.slider("Glucose Level",     50,   250,  120)
    blood_press = st.slider("Blood Pressure",    30,   130,  80)
    skin_thick  = st.slider("Skin Thickness",    5,    100,  25)

with col2:
    st.subheader("🔬 Medical Values")
    insulin     = st.slider("Insulin Level",     10,   900,  85)
    bmi         = st.slider("BMI",               10.0, 70.0, 28.5)
    dpf         = st.slider("Diabetes Pedigree", 0.0,  2.5,  0.45)
    age         = st.slider("Age",               10,   100,  35)

st.markdown("---")

if st.button("🔍 Predict Diabetes", use_container_width=True):

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

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col3, col4 = st.columns(2)

    if prediction == 1:
        risk = "HIGH" if probability[1] > 0.7 else "MODERATE"
        with col3:
            st.error(f"🔴 DIABETIC — {risk} RISK")
            st.metric("Diabetes Probability",    f"{probability[1]*100:.1f}%")
            st.metric("No Diabetes Probability", f"{probability[0]*100:.1f}%")
    else:
        with col3:
            st.success("🟢 NON-DIABETIC")
            st.metric("No Diabetes Probability", f"{probability[0]*100:.1f}%")
            st.metric("Diabetes Probability",    f"{probability[1]*100:.1f}%")

    st.markdown("---")
    st.subheader("🥗 Health Guide")

    if prediction == 1:
        st.error("⚠️ Please consult a doctor immediately!")

        with st.expander("🥗 Foods to EAT"):
            st.write("✅ Karela (bitter gourd) — controls blood sugar")
            st.write("✅ Methi (fenugreek)     — reduces glucose")
            st.write("✅ Jamun                 — lowers sugar levels")
            st.write("✅ Leafy greens          — spinach, palak")
            st.write("✅ Whole grains          — brown rice, oats")
            st.write("✅ Nuts                  — almonds, walnuts")

        with st.expander("🚫 Foods to AVOID"):
            st.write("❌ White rice, maida, white bread")
            st.write("❌ Sugar, mithai, cold drinks")
            st.write("❌ Fried foods — samosa, pakoda")
            st.write("❌ Alcohol and smoking")

        with st.expander("🌿 Home Remedies"):
            st.write("🌱 Karela juice      — 30ml every morning")
            st.write("🌱 Methi seeds       — soak overnight, eat empty stomach")
            st.write("🌱 Jamun seed powder — 1 tsp with water daily")
            st.write("🌱 Amla juice        — 20ml daily morning")
            st.write("🌱 Cinnamon          — half tsp in warm water")

        with st.expander("💊 Medicines — Doctor Only!"):
            st.write("• Metformin     — most common first medicine")
            st.write("• Glipizide     — stimulates insulin release")
            st.write("• Sitagliptin   — controls sugar after meal")
            st.warning("⚠️ NEVER take medicines without doctor advice!")

        with st.expander("🏃 Lifestyle Tips"):
            st.write("• Walk 30 minutes daily after meals")
            st.write("• Yoga — Surya Namaskar, Halasana daily")
            st.write("• Sleep 7-8 hours every night")
            st.write("• Drink 2-3 litres of water daily")
    else:
        st.success("✅ You are healthy! Keep it up!")

        with st.expander("💪 Prevention Tips"):
            st.write("✅ Maintain healthy weight — BMI 18.5 to 24.9")
            st.write("✅ Exercise 30 min daily")
            st.write("✅ Eat more vegetables and fruits")
            st.write("✅ Avoid sugar and processed food")
            st.write("✅ Drink 2-3 litres of water daily")

st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Not a substitute for medical advice.")
