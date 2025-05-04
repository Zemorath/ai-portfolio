import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

df = pd.read_csv("telco_churn.csv")

features = ["tenure", "MonthlyCharges", "Contract", "InternetService"]
df = df[features + ["Churn"]]

df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce")
df = df.fillna(df.mean(numeric_only=True))

le_contract = LabelEncoder()
le_internet = LabelEncoder()
le_churn = LabelEncoder()
df["Contract"] = le_contract.fit_transform(df["Contract"])
df["InternetService"] = le_internet.fit_transform(df["InternetService"])
df["Churn"] = le_churn.fit_transform(df["Churn"])

X = df[features]
y = df["Churn"]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X[["tenure", "MonthlyCharges"]])
X.loc[:, ["tenure", "MonthlyCharges"]] = scaled_features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(8, 6))
plt.bar(importance["Feature"], importance["Importance"])
plt.title("Feature Importance in Churn Prediction")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.savefig("feature_importance.png")
plt.close()

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_contract, "le_contract.pkl")
joblib.dump(le_internet, "le_internet.pkl")
print("Model and preprocessors saved.")