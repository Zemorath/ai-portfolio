import pandas as pd

df = pd.read_csv("iris.csv")
print("First 5 rows:\n", df.head())

print("Summary stats:\n", df.describe())
print("Species counts:\n", df["species"].value_counts())

setosa = df[df["species"] == "setosa"]
print("Setosa petal length mean:", setosa["petal_length"].mean())