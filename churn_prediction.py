import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Generate synthetic dataset (mimics telecom churn)
np.random.seed(42)
n_samples = 1000
data = {
    "tenure": np.random.randint(1, 72, n_samples),
    "monthly_charges": np.random.uniform(20, 120, n_samples),
    "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
    "internet_service": np.random.choice(["Fiber optic", "DSL", "No internet"], n_samples),
    "churn": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7])
}
df = pd.DataFrame(data)

# Save to CSV (optional, for inspection)
df.to_csv("churn_data.csv", index=False)

# Preprocess data
# Encode categorical variables with separate LabelEncoders
le_contract = LabelEncoder()
le_internet = LabelEncoder()
le_churn = LabelEncoder()
df["contract_type"] = le_contract.fit_transform(df["contract_type"])
df["internet_service"] = le_internet.fit_transform(df["internet_service"])
df["churn"] = le_churn.fit_transform(df["churn"])  # Yes=1, No=0

# Handle missing values (none in synthetic data, but good practice)
df = df.fillna(df.mean(numeric_only=True))

# Features and target
X = df[["tenure", "monthly_charges", "contract_type", "internet_service"]]
y = df["churn"]

# Scale numerical features using .loc
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X[["tenure", "monthly_charges"]])
X.loc[:, ["tenure", "monthly_charges"]] = scaled_features

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}
results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "cv_mean": cv_scores.mean()
    }
    # Feature importance for Random Forest
    if name == "Random Forest":
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        results[name]["feature_importance"] = importance

# Print results
for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"Cross-Validation Accuracy: {metrics['cv_mean']:.2f}")
    if "feature_importance" in metrics:
        print("Feature Importance:\n", metrics["feature_importance"])

# Example prediction
example = pd.DataFrame(
    [[12, 59.99, "One year", "Fiber optic"]],
    columns=["tenure", "monthly_charges", "contract_type", "internet_service"]
)
# Preprocess example
example["contract_type"] = le_contract.transform([example["contract_type"].iloc[0]])[0]
example["internet_service"] = le_internet.transform([example["internet_service"].iloc[0]])[0]
example.loc[:, ["tenure", "monthly_charges"]] = scaler.transform(example[["tenure", "monthly_charges"]])
# Predict with Random Forest
rf_model = models["Random Forest"]
pred = rf_model.predict(example)[0]
print(f"\nExample Prediction (Random Forest): {'Churn' if pred == 1 else 'No Churn'}")