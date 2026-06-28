# UDI aur Product ID remove
# Data cleaning
import pandas as pd

df = pd.read_csv("data/machine.csv")
df = df.drop(["UDI", "Product ID"], axis=1)

print(df.shape)
print(df.columns)


# Failure reason columns remove
# data leakage remove

df = df.drop(["TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

print(df.shape)
print(df.columns)

# save new cleaned dataset

df.to_csv("data/cleaned_data.csv", index=False)

print("Cleaned data saved successfully")
