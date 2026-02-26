# Customer Churn Prediction

Predict customer churn in the telecom industry using machine learning.

Customer churn — when a customer stops using a service — is a major challenge for subscription-based businesses. This project analyzes historical telecom customer data and builds a model to estimate the probability that a customer will churn. It also highlights the most influential factors affecting churn.

---

## 🔍 Project Overview

Customer churn prediction helps businesses identify users who are likely to leave the service so that proactive retention strategies can be implemented. This project uses machine learning techniques to build and evaluate a binary classification model that predicts whether a customer will churn or not.

---

## 🧠 Approach

1. **Exploratory Data Analysis (EDA)**  
   - Explore patterns, distributions, and missing values.
   - Understand relationships between features and churn.

2. **Data Preprocessing & Encoding**  
   - Convert categorical features (e.g., gender, contract type) into numeric form.
   - Scale numerical values where necessary.

3. **Model Training**  
   - Train a LightGBM classifier on the processed dataset.


---

## 📁 Dataset

The project uses the **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) containing customer details like:

- Demographics
- Account information
- Services subscribed
- Charges and tenure

---

## 🚀 Features

This project includes:

 Data preprocessing & feature encoding  
 LightGBM churn prediction model  
 Flask REST API for predictions  
 Risk level and confidence scoring

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/safdarsidhik/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Run Flask App**
```
python app.py
```
