from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

model = joblib.load("../churn_model.pkl")
scaler = joblib.load("../scaler.pkl")
le_contract = joblib.load("../le_contract.pkl")
le_internet = joblib.load("../le_internet.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            tenure = float(request.form["tenure"])
            monthly_charges = float(request.form["monthly_charges"])
            contract = request.form["contract"]
            internet = request.form["internet_service"]

            if tenure < 0 or monthly_charges < 0:
                return render_template("index.html", error="Tenure and charges must be non-negative")
                
            if contract not in le_contract.classes_:
                return render_template("index.html", error="Invalid contract type")
            if internet not in le_internet.classes_:
                return render_template("index.html", error="Invalid internet service")
            
            data = pd.DataFrame(
                [[tenure, monthly_charges, contract, internet]],
                columns=["tenure", "MonthlyCharges", "Contract", "InternetService"]
            )

            data["Contract"] = le_contract.transform([data["Contract"].iloc[0]])[0]
            data["InternetService"] = le_internet.transform([data["InternetService"].iloc[0]])[0]
            data.loc[:, ["tenure", "MonthlyCharges"]] = scaler.transform(data[["tenure", "MonthlyCharges"]])

            pred = model.predict(data)[0]
            result = "Churn" if pred ==1 else "No Churn"

            return render_template("index.html", result=result)
        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)