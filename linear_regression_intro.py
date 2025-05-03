import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X = df[["MedInc", "HouseAge"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

print("Coefficients:", dict(zip(X.columns, model.coef_)))
print(f"Intercept: {model.intercept_:.2f}")

example = pd.DataFrame([[8.0, 25.0]], columns=["MedInc", "HouseAge"])
pred_price = model.predict(example)[0]
print(f"Predicted price for MedInc=8, HouseAge=25: ${pred_price*100000:.2f}")
