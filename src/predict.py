import joblib

# Load Saved Model
model = joblib.load("models/random_forest_model.pkl")

# Load Saved Scaler
scaler = joblib.load("models/scaler.pkl")

# User Input (Sample Data)
input_data = [[
    0,          # Type
    298.1,      # Air temperature [K]
    308.6,      # Process temperature [K]
    1551,       # Rotational speed [rpm]
    42.8,       # Torque [Nm]
    0,          # Tool wear [min]
    10.5,       # Temp_Difference
    66382.8,    # Power
    0.0         # Wear_Ratio
]]

# Scale Input Data
input_data = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data)

# Display Result
if prediction[0] == 0:
    print("Machine is Working Normally")
else:
    print("Machine Failure Detected")
