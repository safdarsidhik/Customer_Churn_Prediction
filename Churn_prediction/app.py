from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load model using absolute path relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lgbm_model.pkl')
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


FEATURE_NAMES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]


def encode_features(data):
    """Encode categorical features to numeric."""
    gender_map = {'Male': 1, 'Female': 0}
    yes_no_map = {'Yes': 1, 'No': 0}
    internet_service_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    multiple_lines_map = {'No phone service': 0, 'No': 1, 'Yes': 2}
    online_map = {'No internet service': 0, 'No': 1, 'Yes': 2}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_map = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }

    features = [
        gender_map.get(data.get('gender', 'Male'), 1),
        yes_no_map.get(data.get('SeniorCitizen', 'No'), 0),
        yes_no_map.get(data.get('Partner', 'No'), 0),
        yes_no_map.get(data.get('Dependents', 'No'), 0),
        float(data.get('tenure', 12)),
        yes_no_map.get(data.get('PhoneService', 'Yes'), 1),
        multiple_lines_map.get(data.get('MultipleLines', 'No'), 1),
        internet_service_map.get(data.get('InternetService', 'DSL'), 0),
        online_map.get(data.get('OnlineSecurity', 'No'), 1),
        online_map.get(data.get('OnlineBackup', 'No'), 1),
        online_map.get(data.get('DeviceProtection', 'No'), 1),
        online_map.get(data.get('TechSupport', 'No'), 1),
        online_map.get(data.get('StreamingTV', 'No'), 1),
        online_map.get(data.get('StreamingMovies', 'No'), 1),
        contract_map.get(data.get('Contract', 'Month-to-month'), 0),
        yes_no_map.get(data.get('PaperlessBilling', 'No'), 0),
        payment_map.get(data.get('PaymentMethod', 'Electronic check'), 0),
        float(data.get('MonthlyCharges', 70.0)),
        float(data.get('TotalCharges', 1000.0)),
    ]
    return features


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure lgbm_model.pkl is placed in the models/ folder.'}), 503

    try:
        data = request.get_json()
        features = encode_features(data)
        features_array = np.array([features])

        prob = float(model.predict_proba(features_array)[0][1])

        prediction = 'Churn' if prob > 0.5 else 'No Churn'
        confidence = f'{round(abs(prob - 0.5) * 200, 1)}%'

        if prob > 0.7:
            risk_level = 'High Risk'
        elif prob >= 0.4:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Low Risk'

        return jsonify({
            'prediction': prediction,
            'probability': round(prob, 4),
            'confidence': confidence,
            'risk_level': risk_level,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
