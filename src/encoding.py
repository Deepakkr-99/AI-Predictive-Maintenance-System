import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

os.makedirs("models", exist_ok=True)

df=pd.read_csv("data/featured_data.csv")
le=LabelEncoder()

df["Type"]=le.fit_transform(df["Type"])

# check krna ki numeric me convert hua
print(df.head())

print("shape of the data",df.shape)

df.to_csv("data/encoded_data.csv",index=False)

# label encoder save karo
joblib.dump(le, "models/label_encoder.pkl")

print("Encoding Completed")
