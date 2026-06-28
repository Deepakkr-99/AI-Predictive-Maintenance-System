import pandas as pd
import joblib

# Load Saved Model
model = joblib.load("models/random_forest_model.pkl")

# Load Saved Scaler
scaler = joblib.load("models/scaler.pkl")

# User Input (Sample Data)
input_data = pd.DataFrame([{
    "Type": 0,
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
    "Temp_Difference": 10.5,
    "Power": 66382.8,
    "Wear_Ratio": 0.0
}])

# Scale Input Data
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

# Display Result
if prediction[0] == 0:
    print("Machine is Working Normally")
else:
    print("Machine Failure Detected")

