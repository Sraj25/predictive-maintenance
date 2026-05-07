"""
================================================
 Predictive Maintenance — Streamlit Web App
 Run: streamlit run app.py
================================================
"""

import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4C72B0;
        margin-bottom: 12px;
    }

    /* Failure alert */
    .failure-box {
        background: linear-gradient(135deg, #3d0000, #5c0000);
        border: 2px solid #ff4b4b;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
    }
    /* Safe alert */
    .safe-box {
        background: linear-gradient(135deg, #003d1a, #005c27);
        border: 2px solid #00cc66;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
    }
    /* Section header */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a0aec0;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    /* Gauge label */
    .gauge-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: -8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path  = os.path.join("model", "random_forest_model.pkl")
    scaler_path = os.path.join("model", "scaler.pkl")
    if not os.path.exists(model_path):
        return None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

model, scaler = load_artifacts()


# ─────────────────────────────────────────────
# HELPER — GAUGE CHART
# ─────────────────────────────────────────────
def draw_gauge(prob):
    """Draw a semicircular gauge showing failure probability."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(polar=False))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.axis("off")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(np.cos(theta)*r, np.sin(theta)*r, color="#2d3748", linewidth=18, solid_capstyle="round")

    # Filled arc (prob portion)
    fill_theta = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    color = "#ff4b4b" if prob > 0.5 else "#00cc66"
    ax.plot(np.cos(fill_theta)*r, np.sin(fill_theta)*r, color=color,
            linewidth=18, solid_capstyle="round")

    # Needle
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(np.cos(angle)*0.85, np.sin(angle)*0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=16))
    ax.plot(0, 0, "o", color="white", markersize=8, zorder=5)

    # Labels
    ax.text(-1.1, -0.15, "0%",  color="#718096", fontsize=9, ha="center")
    ax.text( 1.1, -0.15, "100%", color="#718096", fontsize=9, ha="center")
    ax.text(0, 0.45, f"{prob*100:.1f}%",
            color=color, fontsize=18, ha="center", fontweight="bold")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.2)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Sensor Input Panel")
    st.markdown("---")
    st.markdown('<p class="section-title">Machine Parameters</p>', unsafe_allow_html=True)

    temperature = st.slider(
        "🌡️ Temperature (°C)",
        min_value=40.0, max_value=130.0, value=75.0, step=0.5,
        help="Normal range: 60–90 °C"
    )
    vibration = st.slider(
        "📳 Vibration (mm/s)",
        min_value=0.1, max_value=1.5, value=0.5, step=0.01,
        help="Normal range: 0.3–0.7 mm/s"
    )
    pressure = st.slider(
        "💨 Pressure (bar)",
        min_value=10.0, max_value=60.0, value=30.0, step=0.5,
        help="Normal range: 20–40 bar"
    )
    runtime_hours = st.slider(
        "⏱️ Runtime (hours)",
        min_value=0, max_value=7000, value=1000, step=50,
        help="Cumulative machine runtime"
    )

    st.markdown("---")
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Normal Operating Ranges:**")
    st.markdown("- Temperature: 60–90 °C")
    st.markdown("- Vibration: 0.3–0.7 mm/s")
    st.markdown("- Pressure: 20–40 bar")
    st.markdown("- Runtime: 0–5000 hrs")


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.markdown("# ⚙️ Predictive Maintenance AI")
st.markdown("### Real-time Equipment Failure Detection System")
st.markdown("---")

if model is None:
    st.warning("""
    ⚠️ **No trained model found.**
    
    Please run the training pipeline first:
    ```bash
    python src/generate_data.py
    python src/eda.py
    python src/train_model.py
    ```
    Then relaunch the app.
    """)
    st.stop()

# ── Current readings display ────────────────
col1, col2, col3, col4 = st.columns(4)
readings = [
    ("🌡️", "Temperature", f"{temperature:.1f} °C", temperature, 60, 90),
    ("📳", "Vibration",   f"{vibration:.2f} mm/s", vibration,   0.3, 0.7),
    ("💨", "Pressure",    f"{pressure:.1f} bar",   pressure,    20,  40),
    ("⏱️", "Runtime",     f"{runtime_hours} hrs",  runtime_hours, 0, 5000),
]
for col, (icon, name, val_str, val, lo, hi) in zip([col1,col2,col3,col4], readings):
    status = "🟢" if lo <= val <= hi else "🔴"
    col.metric(label=f"{icon} {name}", value=val_str, delta=f"{status} {'Normal' if lo<=val<=hi else 'Out of range'}")


# ── Prediction ───────────────────────────────
if predict_btn:
    input_arr = np.array([[temperature, vibration, pressure, runtime_hours]])

    # Random Forest doesn't strictly need scaling but apply if scaler exists
    input_pred = input_arr  # RF uses raw features
    prob   = model.predict_proba(input_arr)[0][1]
    result = model.predict(input_arr)[0]

    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")

    res_col, gauge_col = st.columns([1.4, 1])

    with res_col:
        if result == 1:
            st.markdown(f"""
            <div class="failure-box">
                <h1 style="color:#ff4b4b; margin:0; font-size:3rem;">🚨 FAILURE PREDICTED</h1>
                <p style="color:#fca5a5; font-size:1.2rem; margin-top:10px;">
                    Immediate maintenance required!<br>
                    Failure probability: <b>{prob*100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
                <h1 style="color:#00cc66; margin:0; font-size:3rem;">✅ MACHINE HEALTHY</h1>
                <p style="color:#86efac; font-size:1.2rem; margin-top:10px;">
                    Operating within normal parameters.<br>
                    Failure probability: <b>{prob*100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

    with gauge_col:
        st.markdown('<p class="section-title">Failure Probability Gauge</p>', unsafe_allow_html=True)
        fig_gauge = draw_gauge(prob)
        st.pyplot(fig_gauge, use_container_width=True)

    # ── Risk analysis ─────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Risk Analysis")
    r1, r2, r3 = st.columns(3)

    risk_factors = {
        "Temperature Risk": (temperature - 60) / (130 - 60),
        "Vibration Risk":   (vibration  - 0.1) / (1.5  - 0.1),
        "Pressure Risk":    (pressure   - 10)  / (60   - 10),
    }
    for col, (label, risk_val) in zip([r1, r2, r3], risk_factors.items()):
        pct = min(max(risk_val * 100, 0), 100)
        color = "🔴" if pct > 65 else ("🟡" if pct > 40 else "🟢")
        col.metric(label=label, value=f"{pct:.0f}%", delta=f"{color} {'High' if pct>65 else 'Medium' if pct>40 else 'Low'} Risk")

    # ── Recommendations ───────────────────────
    st.markdown("---")
    st.markdown("## 💡 Maintenance Recommendations")
    recs = []
    if temperature > 90:
        recs.append("🌡️ **High Temperature:** Check cooling system, inspect fans and heat exchangers.")
    if vibration > 0.7:
        recs.append("📳 **High Vibration:** Inspect bearings, check for loose components or imbalance.")
    if pressure > 40:
        recs.append("💨 **High Pressure:** Check pressure relief valves and seals immediately.")
    if runtime_hours > 4000:
        recs.append("⏱️ **Long Runtime:** Schedule preventive maintenance — lubrication and part inspection.")
    if not recs:
        recs.append("✅ All parameters normal. Continue routine monitoring per maintenance schedule.")
    for rec in recs:
        st.info(rec)

else:
    # Welcome state
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; opacity:0.6;">
        <h2>👈 Adjust sensor values in the sidebar</h2>
        <p style="font-size:1.1rem;">Then click <b>Run Prediction</b> to analyze equipment health</p>
    </div>
    """, unsafe_allow_html=True)

    # Show sample plots if they exist
    if os.path.exists("plots/feature_importance.png"):
        st.markdown("---")
        st.markdown("## 📈 Model Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.image("plots/feature_importance.png", caption="Feature Importance — Random Forest")
        with c2:
            if os.path.exists("plots/cm_random_forest.png"):
                st.image("plots/cm_random_forest.png", caption="Confusion Matrix — Random Forest")