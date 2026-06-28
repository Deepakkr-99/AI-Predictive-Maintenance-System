import pandas as pd

# ---- Step 1: Dataset Load Karna ----

df=pd.read_csv("data/machine.csv")
print(df)

# ---- Step 2: Pehli 5 rows dekhna ----
print("===== First 5 Rows =====")
print(df.head())

# ---- Step 3: Dataset ka Shape ----
print("===== Dataset Shape =====")
print(df.shape)

# ---- Step 4: Columns ki List ----
print("===== Column Names =====")
print(df.columns.tolist())

# ---- Step 5: Data Types ----
print("===== Data Types =====")
print(df.dtypes)

# ---- Step 6: Missing Values Check ----
print("===== Missing Values =====")
print(df.isnull().sum())

# ---- Step 7: Duplicate Rows Check ----
print("===== Duplicate Rows Count =====")
print(df.duplicated().sum())

# ---- Step 8: Statistical Summary ----
print("===== Statistical Summary =====")
print(df.describe())

# ---- Step 9: Target Column Distribution ----
print("===== Machine Failure Distribution =====")
print(df["Machine failure"].value_counts())

# ---- Step 10: Percentage Distribution ----
print("===== Machine Failure Percentage =====")
print(round(df["Machine failure"].value_counts(normalize=True) * 100, 2))

# ---- Step 11: Failure Reason Columns Check ----
print("===== Failure Reason Breakdown (TWF, HDF, PWF, OSF, RNF) =====")
print(df[["TWF", "HDF", "PWF", "OSF","RNF"]].sum())

# ---- Step 12: Product Type Distribution ----
print("===== Product Type Distribution =====")
print(df["Type"].value_counts())
