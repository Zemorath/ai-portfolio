import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")

plt.figure(figsize=(8,6))
plt.scatter(df["sepal_length"], df["petal_length"], c="blue", label="Data Points")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal vs Petal Length")
plt.legend()
plt.savefig("scatter_matplotlib.png")
plt.close()

plt.figure(figsize=(8,6))
sns.boxplot(x="species", y="sepal_width", data=df)
plt.title("Sepal Width by Species")
plt.savefig("boxplot_seaborn.png")
plt.close()