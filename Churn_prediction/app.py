from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import os
import joblib
app = Flask(__name__)
'''
# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'lgbm_model.pkl')
model = None

def load_model():
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
load_model()
'''
# Load model
try:
    model = joblib.load("models/lgbm_model.pkl")
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

def get_shap_values(features_array):
    """Get SHAP-like feature contributions using model's predict with perturbation."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(features_array)
        if isinstance(shap_vals, list):
            return shap_vals[1][0].tolist(), float(explainer.expected_value[1])
        return shap_vals[0].tolist(), float(explainer.expected_value)
    except Exception:
        # Fallback: approximate contributions
        base_pred = model.predict_proba(features_array)[0][1]
        contributions = []
        for i in range(len(features_array[0])):
            perturbed = features_array.copy()
            perturbed[0][i] = 0
            perturbed_pred = model.predict_proba(perturbed)[0][1]
            contributions.append(float(base_pred - perturbed_pred))
        return contributions, 0.5


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = encode_features(data)
        features_array = np.array([features])

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        prob = model.predict_proba(features_array)[0][1]
        churn_prob = float(prob) * 100

        if churn_prob < 30:
            risk_level = 'Low'
            risk_color = '#22c55e'
        elif churn_prob < 60:
            risk_level = 'Medium'
            risk_color = '#f59e0b'
        else:
            risk_level = 'High'
            risk_color = '#ef4444'

        confidence = abs(churn_prob - 50) * 2

        # SHAP values
        try:
            shap_contributions, base_value = get_shap_values(features_array)
        except Exception:
            shap_contributions = [0.0] * len(FEATURE_NAMES)
            base_value = 0.5

        feature_importance = [
            {'feature': FEATURE_NAMES[i], 'value': float(features[i]), 'contribution': shap_contributions[i]}
            for i in range(len(FEATURE_NAMES))
        ]
        feature_importance.sort(key=lambda x: abs(x['contribution']), reverse=True)

        return jsonify({
            'churn_probability': round(churn_prob, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': round(confidence, 1),
            'feature_importance': feature_importance[:10],
            'base_value': base_value,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
