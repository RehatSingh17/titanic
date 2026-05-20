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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f5f0;
    color: #1a1a1a;
}

#MainMenu, footer, header { visibility: hidden; }

.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #e8e2da;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    color: #1a1a1a;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.hero p {
    font-size: 1rem;
    color: #7a6e5f;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}
.year-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7a6e5f;
    border: 1px solid #d9d2c8;
    border-radius: 999px;
    padding: 3px 12px;
    margin-bottom: 1rem;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9e9080;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #7a6e5f !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #fff !important;
    border: 1px solid #d9d2c8 !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: #1a1a1a !important;
    box-shadow: none !important;
}

.stSlider > div > div > div > div {
    background: #1a1a1a !important;
}

.stButton > button {
    width: 100%;
    background: #1a1a1a !important;
    color: #f7f5f0 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
    transition: background 0.18s !important;
}
.stButton > button:hover { background: #2e2e2e !important; }

.result-survived {
    background: #edf7ee;
    border: 1px solid #b6deb9;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-died {
    background: #fdf0f0;
    border: 1px solid #f0bcbc;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-icon  { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 400;
    margin-bottom: 0.4rem;
}
.result-survived .result-title { color: #2e7d32; }
.result-died    .result-title  { color: #c62828; }
.result-prob   { font-size: 0.9rem; color: #7a6e5f; }
.result-prob span { font-weight: 500; color: #1a1a1a; }

.prob-bar-wrap {
    background: #e8e2da;
    border-radius: 999px;
    height: 6px;
    margin: 1rem auto 0;
    max-width: 260px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 999px; }

.divider {
    border: none;
    border-top: 1px solid #e8e2da;
    margin: 2rem 0;
}

.factor-card {
    background: #fff;
    border: 1px solid #e8e2da;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.factor-label {
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9e9080;
    margin-bottom: 4px;
}
.factor-val  { font-size: 1rem; font-weight: 500; color: #1a1a1a; }
.factor-sig  { font-size: 0.78rem; margin-top: 2px; }

.footnote {
    text-align: center;
    font-size: 0.78rem;
    color: #b0a898;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e8e2da;
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

embarked_encoded = {
    "Southampton": 0,
    "Cherbourg": 1,
    "Queenstown": 2
}[embarked]

input_data = np.array([
    [
        pclass,
        sex_encoded,
        age,
        fare,
        sibsp,
        parch,
        embarked_encoded
    ]
])
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
                     style="width:{surv_pct}%; background:#4caf50"></div>
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
                     style="width:{surv_pct}%; background:#e53935"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Key factors
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Key factors</div>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        sig   = "↑ Positive" if sex == "Female" else "↓ Negative"
        color = "#2e7d32"    if sex == "Female" else "#c62828"
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-label">Sex</div>
            <div class="factor-val">{sex}</div>
            <div class="factor-sig" style="color:{color}">{sig}</div>
        </div>""", unsafe_allow_html=True)

    with f2:
        sig   = "↑ Positive" if pclass == 1 else ("→ Neutral" if pclass == 2 else "↓ Negative")
        color = "#2e7d32"    if pclass == 1 else ("#9e9080"   if pclass == 2 else "#c62828")
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-label">Class</div>
            <div class="factor-val">{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}</div>
            <div class="factor-sig" style="color:{color}">{sig}</div>
        </div>""", unsafe_allow_html=True)

    with f3:
        sig   = "↑ Positive" if age < 15 else ("→ Neutral" if age < 40 else "↓ Negative")
        color = "#2e7d32"    if age < 15 else ("#9e9080"   if age < 40 else "#c62828")
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
