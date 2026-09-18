from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the exported model and scaler
try:
    model = joblib.load('svc_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Expected features in the correct order:
        # ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
            data['fbs'], data['restecg'], data['thalach'], data['exang'],
            data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]
        
        # Convert to numpy array and reshape for prediction (1 sample)
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = int(model.predict(scaled_features)[0])
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'interpretation': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        })
        
    except KeyError as ke:
        return jsonify({'status': 'error', 'message': f'Missing feature parameter: {str(ke)}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)
