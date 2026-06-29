import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Dynamically add the 'src' directory to Python's system path to prevent ImportError
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Charts aur auxiliary modules ko import karna
from failure_reason import failure_reason
from evaluation import get_model_metrics
from shap_helper import get_shap_values
from charts import (
    make_prob_chart, make_model_chart, make_feature_chart,
    make_confusion_chart, make_roc_chart, make_shap_chart,
    make_correlation_chart, make_failure_distribution_chart, make_eda_scatter_chart
)

# Streamlit App basic configuration (Wide layout aur tab icon)
st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

# Custom style.css file safely link karna UI styling ke liye
css_file = SRC_DIR / "style.css"
if css_file.exists():
    try:
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")
else:
    st.warning("⚠️ CSS styling file (style.css) not found. Using default Streamlit styles.")

# Machine Learning trained model aur scaler files load karna using dynamic absolute paths
model_path = ROOT_DIR / "models" / "random_forest_model.pkl"
scaler_path = ROOT_DIR / "models" / "scaler.pkl"

if not model_path.exists() or not scaler_path.exists():
    st.error("❌ Model files (random_forest_model.pkl or scaler.pkl) are missing in the models directory.")
    st.info("Please run the training pipeline first or ensure they are pushed to GitHub.")
    st.stop()

try:
    if "model" not in st.session_state:
        st.session_state.model = joblib.load(model_path)
        st.session_state.scaler = joblib.load(scaler_path)
    model = st.session_state.model
    scaler = st.session_state.scaler
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# Model evaluation metrics compute karna aur save karna
try:
    if "metrics" not in st.session_state:
        st.session_state.metrics = get_model_metrics(model, scaler)
    metrics = st.session_state.metrics
except Exception as e:
    st.error(f"❌ Evaluation files or data are missing: {e}")
    st.stop()

# Dataset aur Model Results files read karna page charts ke liye using dynamic absolute paths
model_result_path = ROOT_DIR / "data" / "model_result.csv"
feature_data_path = ROOT_DIR / "data" / "feature_importance.csv"
dataset_path = ROOT_DIR / "data" / "encoded_data.csv"

if not model_result_path.exists() or not feature_data_path.exists() or not dataset_path.exists():
    st.error("❌ CSV files (model results/encoded data) are missing in data directory.")
    st.stop()

try:
    if "model_result" not in st.session_state:
        st.session_state.model_result = pd.read_csv(model_result_path)
    if "feature_data" not in st.session_state:
        st.session_state.feature_data = pd.read_csv(feature_data_path)
    if "dataset" not in st.session_state:
        st.session_state.dataset = pd.read_csv(dataset_path)
    
    model_result = st.session_state.model_result
    feature_data = st.session_state.feature_data
    dataset = st.session_state.dataset
except Exception as e:
    st.error(f"❌ Error loading CSV files: {e}")
    st.stop()

# Sidebar menu tabs navigation
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio("Go to Page", ["Predictive Maintenance Dashboard", "Exploratory Data Analysis (EDA)"])

# ==================== PAGE 1: PREDICTIVE MAINTENANCE DASHBOARD ====================
if app_mode == "Predictive Maintenance Dashboard":
    # Teal Gradient Banner header
    st.markdown("""
    <div class="header-container">
        <h1>🛠️ Predictive Maintenance System</h1>
        <p style="color: #e0f2fe; margin: 5px 0 0 0; font-weight: 500; font-size: 15px;">AI-Powered Machine Failure Prediction Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Input elements aur AI outputs side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Machine Parameters Input")
        machine_type = st.selectbox("Machine Type", ["H", "L", "M"])
        air_temp = st.number_input("Air Temperature (K)", value=0.0)
        process_temp = st.number_input("Process Temperature (K)", value=0.0)
        speed = st.number_input("Rotational Speed (rpm)", value=0)
        torque = st.number_input("Torque (Nm)", value=0.0)
        tool_wear = st.number_input("Tool Wear (min)", value=0)

        predict_btn = st.button("🔍 Predict Failure")

    # Variables state setup
    pred = 0
    working = 100.0
    failure = 0.0
    reason = ["Machine Parameters are Normal"]

    # Jab Predict button click ho tab machine testing parameters run karna
    if predict_btn:
        # Inputs par dynamic feature engineering calculate karna
        temp_diff = process_temp - air_temp
        power = speed * torque
        wear_ratio = 0.0 if speed == 0 else tool_wear / speed

        # Category mapping for machine type (H/L/M)
        if machine_type == "H":
            type_num = 0
        elif machine_type == "L":
            type_num = 1
        else:
            type_num = 2

        # Dataframe setup inference pipeline ke liye
        input_data = pd.DataFrame({
            "Type": [type_num],
            "Air temperature [K]": [air_temp],
            "Process temperature [K]": [process_temp],
            "Rotational speed [rpm]": [speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear],
            "Temp_Difference": [temp_diff],
            "Power": [power],
            "Wear_Ratio": [wear_ratio]
        })

        # Scaling and ML model prediction probabilities calculate karna
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        working = round(prob[0] * 100, 2)
        failure = round(prob[1] * 100, 2)

        # Rule-based failure diagnostic filter run karna
        reason = failure_reason(tool_wear, torque, speed)
        if pred == 1 and (not reason or reason == ["Machine Parameters are Normal"]):
            # Operational failure physics rules evaluate karna
            power_w = torque * speed * 0.10472
            if power_w < 3500 or power_w > 9000:
                reason = ["Power Failure (PWF)"]
            elif process_temp - air_temp < 8.6 and speed < 1380:
                reason = ["Heat Dissipation Failure (HDF)"]
            else:
                reason = ["Complex Operational Failure"]

        # Local explainability (SHAP values) calculation step
        feature_names = list(input_data.columns)
        st.session_state.shap_df = get_shap_values(model, input_scaled, feature_names)

    with col2:
        st.subheader("📊 AI Prediction Summary")
        if predict_btn:
            # Prediction outcome visual alert blocks
            if pred == 0:
                st.markdown('<div class="result-normal">✅ Machine is Working Normally</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-fail">⚠️ Machine Failure Detected</div>', unsafe_allow_html=True)

            # Probabilities split display
            m1, m2 = st.columns(2)
            m1.metric("Working Probability", f"{working}%")
            m2.metric("Failure Probability", f"{failure}%")

            # Probability bar representation
            st.plotly_chart(make_prob_chart(working, failure), use_container_width=True)
        else:
            st.info("Please enter the machine parameters on the left and click 'Predict Failure'.")

    # -------- SECTION 2: DIAGNOSIS & ACTION RECOMMENDATIONS --------
    if predict_btn:
        st.markdown("---")
        st.subheader("📌 Analysis & Maintenance")
        col_diag, col_rec = st.columns(2)
        
        with col_diag:
            st.markdown("**Failure Reason / Diagnosis**")
            st.markdown('<div class="reason-box">', unsafe_allow_html=True)
            for item in reason:
                st.markdown(f'<p class="reason-text">🔸 {item}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_rec:
            st.markdown("**Recommended Maintenance Steps**")
            if pred == 0:
                has_warnings = any(r in reason for r in ["High Tool Wear", "High Torque", "Low Rotational Speed", "Power Failure (PWF)", "Heat Dissipation Failure (HDF)"])
                if has_warnings:
                    st.info("💡 Pre-emptive maintenance recommended soon to prevent future failure.")
                    if "High Tool Wear" in reason:
                        st.warning("⚠️ Plan Tool Replacement: Tool wear is high.")
                    if "High Torque" in reason:
                        st.warning("⚠️ Monitor Load: Torque is elevated.")
                    if "Low Rotational Speed" in reason:
                        st.warning("⚠️ Monitor Motor: Rotational speed is lower than usual.")
                    if "Power Failure (PWF)" in reason:
                        st.warning("⚠️ Inspect Power Supply: Voltage/power draw is unstable.")
                    if "Heat Dissipation Failure (HDF)" in reason:
                        st.warning("⚠️ Monitor Cooling: Thermal dissipation index is warning.")
                else:
                    st.success("✅ No Maintenance Required. Machine is healthy.")
            else:
                # Custom diagnosis suggestions rendering
                if "High Tool Wear" in reason:
                    st.warning("⚠️ Replace Tool: Wear index exceeded critical limits.")
                if "High Torque" in reason:
                    st.warning("⚠️ Reduce Machine Load: Operating under extreme torque levels.")
                if "Low Rotational Speed" in reason:
                    st.error("🚨 Inspect Motor: Rotational speed dropped below safety threshold.")
                if "Power Failure (PWF)" in reason:
                    st.error("🚨 Check Power Supply: Power draw is unstable.")
                if "Heat Dissipation Failure (HDF)" in reason:
                    st.warning("⚠️ Improve Cooling: Thermal dissipation index is unsafe.")

    # -------- SECTION 3: ML MODEL PERFORMANCE RESULTS --------
    st.markdown("---")
    st.subheader("📋 Model Evaluation Results")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Accuracy", f"{round(metrics['accuracy'] * 100, 2)}%")
    e2.metric("Precision", f"{round(metrics['precision'] * 100, 2)}%")
    e3.metric("Recall", f"{round(metrics['recall'] * 100, 2)}%")
    e4.metric("F1 Score", f"{round(metrics['f1'] * 100, 2)}%")
    e5.metric("ROC-AUC Score", f"{round(metrics['auc'] * 100, 2)}%")

    # -------- SECTION 4: IMPORTANCE & LOCAL SHAP EXPLAINABILITY --------
    if predict_btn or "shap_df" in st.session_state:
        st.markdown("---")
        an1, an2 = st.columns(2)
        
        with an1:
            st.markdown("**Global Feature Importance**")
            st.plotly_chart(make_feature_chart(feature_data), use_container_width=True)
            
        with an2:
            st.markdown("**Local SHAP Explanations**")
            if "shap_df" in st.session_state:
                st.plotly_chart(make_shap_chart(st.session_state.shap_df), use_container_width=True)
                st.dataframe(st.session_state.shap_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run the prediction to display the dynamic SHAP explanations.")

    # -------- SECTION 5: MODEL DIAGNOSTICS (ROC & CONFUSION MATRIX) --------
    st.markdown("---")
    ev1, ev2 = st.columns(2)

    with ev1:
        st.markdown("**ROC Curve**")
        st.plotly_chart(make_roc_chart(metrics["fpr"], metrics["tpr"], metrics["auc"]), use_container_width=True)
        
    with ev2:
        st.markdown("**Confusion Matrix Heatmap**")
        st.plotly_chart(make_confusion_chart(metrics["confusion_matrix"]), use_container_width=True)

    # -------- SECTION 6: MODEL COMPARISON GRAPH --------
    st.markdown("---")
    st.subheader("📈 Model Comparison")
    comp_col1, comp_col2 = st.columns([1, 1.5])

    with comp_col1:
        st.dataframe(model_result, use_container_width=True, hide_index=True)
        
    with comp_col2:
        st.plotly_chart(make_model_chart(model_result), use_container_width=True)

# ==================== PAGE 2: EXPLORATORY DATA ANALYSIS (EDA) ====================
else:
    # Teal Gradient Banner header for EDA
    st.markdown("""
    <div class="header-container">
        <h1>📊 Exploratory Data Analysis (EDA)</h1>
        <p style="color: #e0f2fe; margin: 5px 0 0 0; font-weight: 500; font-size: 15px;">Visualizing Features & Correlations in the Machine Failure Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Preview aur Target distribution rendering
    col_preview, col_dist = st.columns([1.5, 1])

    with col_preview:
        st.subheader("📋 Dataset Preview (Head)")
        st.dataframe(dataset.head(5), use_container_width=True)
        st.markdown(f"**Total Records:** {dataset.shape[0]} | **Total Features:** {dataset.shape[1]}")

    with col_dist:
        st.subheader("⚖️ Target Class Distribution")
        st.plotly_chart(make_failure_distribution_chart(dataset), use_container_width=True)

    # Correlation Matrix Heatmap
    st.markdown("---")
    st.subheader("🔗 Feature Interactions & Heatmap")
    st.plotly_chart(make_correlation_chart(dataset), use_container_width=True)

    # Interactive Scatter Plot selection controls
    st.markdown("---")
    st.subheader("🔍 Interactive Feature Relationship Scatter Plot")
    
    col_feat1, col_feat2 = st.columns(2)
    numeric_cols = [
        "Air temperature [K]", 
        "Process temperature [K]", 
        "Rotational speed [rpm]", 
        "Torque [Nm]", 
        "Tool wear [min]", 
        "Temp_Difference", 
        "Power"
    ]
    
    with col_feat1:
        x_feature = st.selectbox("Select X-Axis Feature", numeric_cols, index=0)
    with col_feat2:
        y_feature = st.selectbox("Select Y-Axis Feature", numeric_cols, index=1)
        
    st.plotly_chart(make_eda_scatter_chart(dataset, x_feature, y_feature), use_container_width=True)

# Page Footer simple layout
st.markdown("""
<div class="footer-text">
    Predictive Maintenance System © 2026 | Project Dashboard
</div>
""", unsafe_allow_html=True)
