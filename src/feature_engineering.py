import pandas as pd

df = pd.read_csv("data/cleaned_data.csv")

df["Temp_Difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]

df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

# speed 0 hone par error na aaye
wear_ratio = []
for i in range(len(df)):
    if df["Rotational speed [rpm]"][i] == 0:
        wear_ratio.append(0.0)
    else:
        wear_ratio.append(df["Tool wear [min]"][i] / df["Rotational speed [rpm]"][i])

df["Wear_Ratio"] = wear_ratio

print(df.head())

print("Shape:")
print(df.shape)

# Save new dataset
df.to_csv("data/featured_data.csv", index=False)

print("Feature Engineering completed")
