import streamlit as st
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #0a1628 0%, #0d2240 50%, #0a1628 100%);
    color: #e8e4d8;
}

#MainMenu, footer, header { visibility: hidden; }

.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #e8e4d8;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.hero p {
    font-size: 1rem;
    color: #8a9ab5;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}
.year-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c4a882;
    border: 1px solid rgba(196,168,130,0.3);
    border-radius: 999px;
    padding: 3px 12px;
    margin-bottom: 1rem;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8a9ab5;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #c8d3e0 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e8e4d8 !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: rgba(196,168,130,0.5) !important;
}

.stSlider > div > div > div > div {
    background: #c4a882 !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #c4a882, #a8835a) !important;
    color: #0a1628 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.result-survived {
    background: linear-gradient(135deg, rgba(26,90,58,0.4), rgba(20,70,45,0.2));
    border: 1px solid rgba(74,200,120,0.35);
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-died {
    background: linear-gradient(135deg, rgba(90,26,26,0.4), rgba(70,20,20,0.2));
    border: 1px solid rgba(200,74,74,0.35);
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-icon  { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
}
.result-survived .result-title { color: #6ee89a; }
.result-died    .result-title  { color: #e87a7a; }
.result-prob   { font-size: 0.9rem; color: #8a9ab5; }
.result-prob span { font-weight: 600; color: #c4a882; }

.prob-bar-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 6px;
    margin: 1rem auto 0;
    max-width: 260px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 999px; }

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 2rem 0;
}

.factor-card {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.factor-label {
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8a9ab5;
    margin-bottom: 4px;
}
.factor-val  { font-size: 1rem; font-weight: 600; color: #e8e4d8; }
.factor-sig  { font-size: 0.78rem; margin-top: 2px; }

.footnote {
    text-align: center;
    font-size: 0.78rem;
    color: #4a5a70;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load("titanic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="year-badge">April 15, 1912</div>
    <h1>🚢 Titanic Survival<br>Predictor</h1>
    <p>Enter a passenger's details to predict whether they would have survived the sinking of the RMS Titanic.</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Passenger details</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox(
        "Passenger Class", [1, 2, 3],
        format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class"
    )
    age = st.slider("Age", min_value=1, max_value=80, value=25)
with col2:
    sex  = st.selectbox("Sex", ["Male", "Female"])
    fare = st.number_input("Fare (£)", min_value=0.0,
                           max_value=600.0, value=50.0, step=1.0)

col3, col4 = st.columns(2)
with col3:
    sibsp = st.slider("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
    parch = st.slider("Parents / Children Aboard", min_value=0, max_value=6, value=0)
with col4:
    embarked = st.selectbox("Port of Embarkation",
                            ["Southampton", "Cherbourg", "Queenstown"])

# ── Encode ────────────────────────────────────────────────────
sex_encoded      = 0 if sex == "Male" else 1
# The model and scaler were only trained on 4 features: Pclass, Sex, Age, Fare
input_data       = np.array([[pclass, sex_encoded, age, fare]])
input_scaled     = scaler.transform(input_data)

# ── Predict button ────────────────────────────────────────────
if st.button("Predict survival"):
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    surv_pct    = probability * 100

    # Result card
    if prediction == 1:
        st.markdown(f"""
        <div class="result-survived">
            <div class="result-icon">✦</div>
            <div class="result-title">Survived</div>
            <div class="result-prob">
                Survival probability: <span>{surv_pct:.1f}%</span>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill"
                     style="width:{surv_pct}%; background:#6ee89a"></div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-died">
            <div class="result-icon">✦</div>
            <div class="result-title">Did Not Survive</div>
            <div class="result-prob">
                Survival probability: <span>{surv_pct:.1f}%</span>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill"
                     style="width:{surv_pct}%; background:#e87a7a"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Key factors
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Key factors</div>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        sig   = "↑ Positive" if sex == "Female" else "↓ Negative"
        color = "#6ee89a"    if sex == "Female" else "#e87a7a"
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-label">Sex</div>
            <div class="factor-val">{sex}</div>
            <div class="factor-sig" style="color:{color}">{sig}</div>
        </div>""", unsafe_allow_html=True)

    with f2:
        sig   = "↑ Positive" if pclass == 1 else ("→ Neutral" if pclass == 2 else "↓ Negative")
        color = "#6ee89a"    if pclass == 1 else ("#c4a882"   if pclass == 2 else "#e87a7a")
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-label">Class</div>
            <div class="factor-val">{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}</div>
            <div class="factor-sig" style="color:{color}">{sig}</div>
        </div>""", unsafe_allow_html=True)

    with f3:
        sig   = "↑ Positive" if age < 15 else ("→ Neutral" if age < 40 else "↓ Negative")
        color = "#6ee89a"    if age < 15 else ("#c4a882"   if age < 40 else "#e87a7a")
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-label">Age</div>
            <div class="factor-val">{age} yrs</div>
            <div class="factor-sig" style="color:{color}">{sig}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footnote">
    Trained on 891 passengers · Kaggle Titanic dataset
</div>
""", unsafe_allow_html=True)