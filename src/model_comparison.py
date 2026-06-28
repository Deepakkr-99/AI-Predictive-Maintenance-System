import pandas as pd

# model training ke baad bani hui csv read karo
df = pd.read_csv("data/model_result.csv")

print("===== Model Comparison =====")
print(df)

print("\nBest Model:", df.loc[df["Accuracy"].idxmax(), "Model"])
