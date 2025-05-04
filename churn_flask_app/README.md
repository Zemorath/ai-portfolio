# Customer Churn Prediction Web App

A Flask web app that predicts customer churn using a Random Forest model trained on the Telco Customer Churn dataset. Built as part of an AI curriculum to showcase machine learning and web development skills.

## Features
- Predicts churn based on tenure, monthly charges, contract type, and internet service.
- Uses Random Forest with class weighting to handle imbalanced data.
- Visualizes feature importance to identify key churn drivers.
- Interactive web interface with input validation.
- Metrics: accuracy, precision, recall, confusion matrix.

## Tech Stack
- **ML**: Scikit-learn (Random Forest, StandardScaler, LabelEncoder)
- **Backend**: Flask, Pandas, Joblib
- **Frontend**: HTML, CSS, Jinja2
- **Tools**: Python 3.10, Ubuntu VM, VS Code, GitHub

## Setup
1. Clone the repository:
   ```bash
   git clone git@github.com:yourusername/ai-portfolio.git
   cd churn_flask_app