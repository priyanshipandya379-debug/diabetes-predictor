import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesSense AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #0a0f1e;
    --card:      #111827;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --red:       #ff4757;
    --green:     #00e676;
    --yellow:    #ffd600;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px; }

/* All text white */
.stApp * { color: var(--text) !important; }

/* Slider */
.stSlider > div > div > div { background: var(--border) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 2rem !important;
    letter-spacing: 1px !important;
    transition: all 0.3s !important;
    font-family: 'Space Mono', monospace !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* HR */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOAD / TRAIN
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = "diabetes_model.pkl"
    scaler_path = "diabetes_scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model  = pickle.load(open(model_path,  "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
        acc = None
    else:
        url = ('https://raw.githubusercontent.com/jbrownlee/Datasets'
               '/master/pima-indians-diabetes.data.csv')
        cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
        df = pd.read_csv(url, names=cols)
        for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
            df[c] = df[c].replace(0, df[c].median())

        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_tr_s, y_tr)
        acc = round(accuracy_score(y_te, model.predict(X_te_s)) * 100, 2)

        pickle.dump(model,  open(model_path,  "wb"))
        pickle.dump(scaler, open(scaler_path, "wb"))

    return model, scaler

model, scaler = load_model()


# ─────────────────────────────────────────────
#  HELPER: PROGRESS BAR
# ─────────────────────────────────────────────
def progress_bar(label, value, min_val, max_val, color="#00d4ff", unit=""):
    pct = min(max((value - min_val) / (max_val - min_val), 0), 1) * 100
    st.markdown(f"""
    <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between;
                    font-size:0.85rem; margin-bottom:4px;">
            <span style="color:#94a3b8; font-weight:500;">{label}</span>
            <span style="color:{color}; font-weight:700;
                         font-family:'Space Mono',monospace;">
                {value}{unit}
            </span>
        </div>
        <div style="background:#1e2d45; border-radius:999px; height:8px;">
            <div style="width:{pct}%; background:{color};
                        border-radius:999px; height:8px;
                        transition:width 0.6s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER: HEALTH SCORE
# ─────────────────────────────────────────────
def health_score(glucose, bp, bmi, age, insulin, prob_diabetic):
    score = 100
    if glucose > 140: score -= 25
    elif glucose > 110: score -= 12
    if bmi > 30: score -= 20
    elif bmi > 25: score -= 10
    if bp > 90: score -= 10
    elif bp > 80: score -= 5
    if age > 60: score -= 10
    elif age > 45: score -= 5
    if insulin > 200: score -= 10
    score -= int(prob_diabetic * 25)
    return max(score, 0)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
    <div style="font-family:'Space Mono',monospace; font-size:0.85rem;
                letter-spacing:4px; color:#00d4ff; margin-bottom:0.5rem;">
        AI · POWERED · HEALTH
    </div>
    <h1 style="font-size:3rem; font-weight:700; margin:0;
               background:linear-gradient(135deg,#00d4ff,#7c3aed);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        DiabetesSense AI
    </h1>
    <p style="color:#64748b; font-size:1rem; margin-top:0.5rem;">
        Advanced Diabetes Risk Prediction System
    </p>
</div>
<div style="height:1px; background:linear-gradient(90deg,transparent,#1e2d45,transparent);
            margin-bottom:2rem;"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  INPUT SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:#111827; border:1px solid #1e2d45; border-radius:16px;
            padding:1.5rem 2rem; margin-bottom:1.5rem;">
    <p style="font-family:'Space Mono',monospace; color:#00d4ff;
              font-size:0.8rem; letter-spacing:3px; margin:0 0 1rem;">
        PATIENT DATA INPUT
    </p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p style="color:#7c3aed; font-weight:600; font-size:0.9rem; letter-spacing:1px;">👤 PERSONAL INFO</p>', unsafe_allow_html=True)
    pregnancies = st.slider("Pregnancies",          0,    17,    2)
    age         = st.slider("Age (years)",          10,   100,   35)
    bmi         = st.slider("BMI",                  10.0, 70.0,  28.5, step=0.1)
    dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.45, step=0.01)

with col2:
    st.markdown('<p style="color:#7c3aed; font-weight:600; font-size:0.9rem; letter-spacing:1px;">🔬 MEDICAL VALUES</p>', unsafe_allow_html=True)
    glucose     = st.slider("Glucose Level (mg/dL)", 50,  250,  120)
    blood_press = st.slider("Blood Pressure (mmHg)", 30,  130,  80)
    insulin     = st.slider("Insulin Level (μU/mL)", 10,  900,  85)
    skin_thick  = st.slider("Skin Thickness (mm)",   5,   100,  25)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BMI CATEGORY DISPLAY
# ─────────────────────────────────────────────
if bmi < 18.5:
    bmi_cat, bmi_col = "Underweight", "#ffd600"
elif bmi < 25:
    bmi_cat, bmi_col = "Normal", "#00e676"
elif bmi < 30:
    bmi_cat, bmi_col = "Overweight", "#ffd600"
else:
    bmi_cat, bmi_col = "Obese", "#ff4757"

st.markdown(f"""
<div style="background:#111827; border:1px solid #1e2d45; border-radius:12px;
            padding:1rem 1.5rem; margin-bottom:1.5rem; display:flex;
            align-items:center; gap:1rem;">
    <span style="font-size:0.85rem; color:#64748b;">⚖️ BMI Category:</span>
    <span style="background:{bmi_col}22; color:{bmi_col}; font-weight:700;
                 padding:4px 14px; border-radius:999px; font-size:0.9rem;
                 border:1px solid {bmi_col}55;">
        {bmi_cat} — {bmi}
    </span>
    <span style="color:#64748b; font-size:0.8rem; margin-left:auto;">
        Normal: 18.5 – 24.9
    </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREDICT BUTTON
# ─────────────────────────────────────────────
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predict_btn = st.button("🔍 ANALYZE DIABETES RISK", use_container_width=True)


# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running AI analysis..."):
        time.sleep(1.2)

    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_press,
        skin_thick, insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age'
    ])

    scaled      = scaler.transform(input_df)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]
    score       = health_score(glucose, blood_press, bmi, age, insulin, probability[1])

    # ── Result Banner ──────────────────────────────────
    if prediction == 1:
        risk_level = "HIGH RISK" if probability[1] > 0.7 else "MODERATE RISK"
        banner_color = "#ff4757"
        banner_bg    = "#ff475715"
        icon = "🔴"
        result_text = f"DIABETIC — {risk_level}"
    else:
        if probability[1] > 0.35:
            result_text = "NON-DIABETIC — BORDERLINE"
            banner_color = "#ffd600"
            banner_bg    = "#ffd60015"
            icon = "⚠️"
        else:
            result_text = "NON-DIABETIC — HEALTHY"
            banner_color = "#00e676"
            banner_bg    = "#00e67615"
            icon = "🟢"
            st.balloons()

    st.markdown(f"""
    <div style="background:{banner_bg}; border:2px solid {banner_color};
                border-radius:16px; padding:2rem; text-align:center;
                margin:1.5rem 0;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">{icon}</div>
        <div style="font-family:'Space Mono',monospace; font-size:1.6rem;
                    font-weight:700; color:{banner_color}; letter-spacing:2px;">
            {result_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 Metric Cards ────────────────────────────────
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1e2d45;
                    border-radius:14px; padding:1.5rem; text-align:center;">
            <div style="color:#64748b; font-size:0.8rem;
                        letter-spacing:2px; margin-bottom:0.5rem;">
                DIABETES PROBABILITY
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:2.2rem;
                        font-weight:700; color:#ff4757;">
                {probability[1]*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        score_color = "#00e676" if score >= 70 else "#ffd600" if score >= 45 else "#ff4757"
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1e2d45;
                    border-radius:14px; padding:1.5rem; text-align:center;">
            <div style="color:#64748b; font-size:0.8rem;
                        letter-spacing:2px; margin-bottom:0.5rem;">
                HEALTH SCORE
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:2.2rem;
                        font-weight:700; color:{score_color};">
                {score}/100
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1e2d45;
                    border-radius:14px; padding:1.5rem; text-align:center;">
            <div style="color:#64748b; font-size:0.8rem;
                        letter-spacing:2px; margin-bottom:0.5rem;">
                SAFE PROBABILITY
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:2.2rem;
                        font-weight:700; color:#00e676;">
                {probability[0]*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Parameter Analysis ────────────────────────────
    st.markdown("""
    <div style="margin-top:1.5rem; margin-bottom:0.5rem;">
        <span style="font-family:'Space Mono',monospace; color:#00d4ff;
                     font-size:0.8rem; letter-spacing:3px;">
            PARAMETER ANALYSIS
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="background:#111827; border:1px solid #1e2d45; border-radius:14px; padding:1.5rem;">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        progress_bar("🩸 Glucose",        glucose,     50,  250, "#ff4757" if glucose > 140 else "#00d4ff", " mg/dL")
        progress_bar("💓 Blood Pressure", blood_press, 30,  130, "#ff4757" if blood_press > 90 else "#00d4ff", " mmHg")
        progress_bar("⚖️ BMI",            bmi,         10,  70,  "#ff4757" if bmi > 30 else "#00e676")
        progress_bar("🎂 Age",            age,         10,  100, "#00d4ff", " yrs")
    with c2:
        progress_bar("💉 Insulin",        insulin,     10,  900, "#ffd600" if insulin > 200 else "#00d4ff", " μU/mL")
        progress_bar("📏 Skin Thickness", skin_thick,  5,   100, "#00d4ff", " mm")
        progress_bar("🤰 Pregnancies",    pregnancies, 0,   17,  "#7c3aed")
        progress_bar("🧬 Pedigree",       dpf,         0.0, 2.5, "#7c3aed")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Health Guide ──────────────────────────────────
    st.markdown("""
    <div style="margin-top:1.5rem; margin-bottom:0.5rem;">
        <span style="font-family:'Space Mono',monospace; color:#00d4ff;
                     font-size:0.8rem; letter-spacing:3px;">
            HEALTH GUIDE
        </span>
    </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("""
        <div style="background:#ff475715; border:1px solid #ff475755;
                    border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem;">
            ⚠️ <strong>Please consult a doctor immediately!</strong>
            This is an AI prediction — not a medical diagnosis.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🥗 Foods to EAT ✅"):
            items = ["Karela — controls blood sugar","Methi — reduces glucose",
                     "Jamun — lowers sugar levels","Leafy greens (spinach, palak)",
                     "Whole grains — brown rice, oats","Nuts — almonds, walnuts"]
            for i in items:
                st.markdown(f"✅ **{i}**")

        with st.expander("🚫 Foods to AVOID ❌"):
            items = ["White rice, maida, white bread","Sugar, mithai, cold drinks",
                     "Fried foods — samosa, pakoda","Alcohol and smoking"]
            for i in items:
                st.markdown(f"❌ **{i}**")

        with st.expander("🌿 Home Remedies"):
            items = ["Karela juice — 30ml every morning","Methi seeds — soak overnight, eat in morning",
                     "Jamun seed powder — 1 tsp daily","Amla juice — 20ml daily morning",
                     "Cinnamon — half tsp in warm water"]
            for i in items:
                st.markdown(f"🌱 **{i}**")

        with st.expander("💊 Medicines — Doctor Prescribed Only"):
            st.markdown("• **Metformin** — most common first medicine")
            st.markdown("• **Glipizide** — stimulates insulin release")
            st.markdown("• **Sitagliptin** — controls sugar after meals")
            st.warning("⚠️ NEVER take medicines without doctor advice!")

        with st.expander("🏃 Lifestyle Tips"):
            st.markdown("• **Walk** 30 minutes daily after meals")
            st.markdown("• **Yoga** — Surya Namaskar daily")
            st.markdown("• **Sleep** 7–8 hours every night")
            st.markdown("• **Drink** 2–3 litres of water daily")

    else:
        with st.expander("💪 Prevention & Stay Healthy Tips"):
            st.markdown("✅ Maintain healthy weight — BMI 18.5 to 24.9")
            st.markdown("✅ Exercise 30 min daily")
            st.markdown("✅ Eat more vegetables and fruits")
            st.markdown("✅ Avoid sugar and processed food")
            st.markdown("✅ Drink 2–3 litres of water daily")
            st.markdown("✅ Get annual health checkups done")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="height:1px; background:linear-gradient(90deg,transparent,#1e2d45,transparent);
            margin:2rem 0 1rem;"></div>
<p style="text-align:center; color:#334155; font-size:0.8rem;">
    
