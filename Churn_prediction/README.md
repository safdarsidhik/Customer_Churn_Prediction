# Customer Churn Prediction App

A Flask-based web application that predicts customer churn using a LightGBM model trained on the Telco Customer Churn dataset.

## What It Does

- Accepts 19 customer features via a modern web UI
- Encodes categorical features and runs them through a pre-trained LightGBM model
- Returns a churn prediction, churn probability, confidence score, and risk level

## Folder Structure

```
Churn_prediction/
├── app.py               ← Flask backend
├── requirements.txt     ← Python dependencies
├── README.md
├── models/
│   └── lgbm_model.pkl   ← Place your trained model here
└── templates/
    └── index.html       ← Frontend UI
```

## How to Run

```bash
cd Churn_prediction
pip install -r requirements.txt
# Place lgbm_model.pkl inside the models/ folder
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## API

**POST /predict**

Accepts a JSON body with the 19 customer feature fields and returns:

```json
{
  "prediction": "Churn",
  "probability": 0.82,
  "confidence": "64.0%",
  "risk_level": "High Risk"
}
```
