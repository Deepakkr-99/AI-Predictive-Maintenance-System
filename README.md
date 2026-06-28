# AI-Powered Predictive Maintenance System

A professional-grade Machine Learning system designed to predict industrial machine failure before it occurs. Built with Python, Scikit-Learn, Plotly, SHAP (Explainable AI), and Streamlit.

---

## 📌 Project Overview
In modern manufacturing, unexpected machine breakdown is highly expensive. This project utilizes sensor data (air temperature, process temperature, speed, torque, and tool wear) to predict machine failure with **99% accuracy**. It includes an end-to-end Machine Learning pipeline and an interactive **Teal & Emerald** styled Streamlit dashboard.

### Key Features
* **Full Data Pipeline**: Scripts to clean data, engineer new features, encode variables, train models, and save them.
* **Explainable AI (SHAP)**: Explains the exact reason *why* the model predicts a machine will fail.
* **Failure Analysis**: Pinpoints issues like High Tool Wear, low speed, or high torque.
* **Model Comparison**: Automatically evaluates and compares Logistic Regression, Decision Trees, Random Forest, and XGBoost.
* **Sleek UI/UX Dashboard**: Custom dark-mode UI with high-contrast Teal & Emerald styling.

---

## 📂 Folder Structure

```text
predictive-maintenance/
├── data/                      # Raw datasets & pipeline output CSVs
├── models/                    # Saved models (PKL files) & Scalers
├── notebooks/                 # Exploratory Data Analysis (EDA) notebook
├── src/                       # Python Source Files
│   ├── app.py                 # Streamlit App Entry Point
│   ├── style.css              # Custom styling (Teal & Emerald theme)
│   ├── charts.py              # Plotly chart utilities
│   ├── data_cleaning.py       # Data cleaning script
│   ├── data_prep.py           # Exploratory data preparation script
│   ├── encoding.py            # Label encoding script
│   ├── feature_engineering.py  # Calculates new features (Wear Ratio, Power, etc.)
│   ├── model_training.py      # Trains and compares all ML models
│   ├── save_model.py          # Saves the best model (Random Forest)
│   ├── predict.py             # Script to verify model output on sample data
│   ├── evaluation.py          # Evaluates performance metrics (Confusion Matrix, ROC)
│   ├── failure_reason.py      # Rule-based failure diagnostic engine
│   └── shap_helper.py         # SHAP explanation calculations
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Tech Stack & Libraries
* **Language**: Python 3.x
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost, SHAP
* **Visualization**: Plotly, Streamlit

---

## 🚀 Installation & Setup

Follow these steps to run the project locally on your machine:

### 1. Clone the Project
```bash
cd predictive-maintenance
```

### 2. Create and Activate Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Project

### Step 1: Run the ML Pipeline (to preprocess data and train models)
Run the following commands in order to prepare the data, train the models, and save the scaler/classifiers:
```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/encoding.py
python src/model_training.py
python src/save_model.py
```

### Step 2: Launch the Web Dashboard
Start the Streamlit application to open the interactive dashboard in your browser:
```bash
streamlit run src/app.py
```

---

## 🧠 Diagnostic Details
* **Confusion Matrix**: Displays static model accuracy on a separate 2,000 sample test set (showing True Negatives, False Positives, False Negatives, True Positives).
* **SHAP Values**: Shows top features driving the machine's state, computed dynamically using Game Theory approximations.

## 📌 Project Context
* **Project Name**: Industrial Predictive Maintenance System
* **Context**: Technical Evaluation & Interactive Diagnostic Dashboard Implementation

Live project:-https://ai-predictive-maintenance-system-6p3eioxr8ohympddvxvkmx.streamlit.app/ 
