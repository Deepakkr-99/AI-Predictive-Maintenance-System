import pandas as pd
import joblib

# saved model se feature importance csv banao
model = joblib.load("models/random_forest_model.pkl")

df = pd.read_csv("data/encoded_data.csv")
feature_names = df.drop("Machine failure", axis=1).columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values("Importance", ascending=False)
importance_df.to_csv("data/feature_importance.csv", index=False)

print(importance_df)
print("\nfeature_importance.csv saved")
