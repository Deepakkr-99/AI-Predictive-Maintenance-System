import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# models folder banao agar nahi hai
os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/encoded_data.csv")

# Input and Output
x = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Final Model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Save Model
joblib.dump(model, "models/random_forest_model.pkl")

# Save Scaler
joblib.dump(scaler, "models/scaler.pkl")

# feature importance csv save
feature_names = x.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})
importance_df = importance_df.sort_values("Importance", ascending=False)
importance_df.to_csv("data/feature_importance.csv", index=False)

print("Model Saved Successfully")
print("feature_importance.csv saved")
