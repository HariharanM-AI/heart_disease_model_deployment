from flask import Flask, request, jsonify, render_template_string
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

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7f9;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            padding: 30px;
            max-width: 600px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #333333;
            margin-bottom: 25px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-weight: 600;
            margin-bottom: 5px;
            color: #555555;
            font-size: 14px;
        }
        input, select {
            padding: 10px;
            border: 1px solid #cccccc;
            border-radius: 5px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            border-color: #4a90e2;
            outline: none;
        }
        button {
            background-color: #4a90e2;
            color: white;
            border: none;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
            transition: background-color 0.3s;
            width: 100%;
        }
        button:hover {
            background-color: #357abd;
        }
        #result {
            margin-top: 25px;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-weight: 600;
            display: none;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .danger {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Heart Disease Predictor</h1>
        <form id="predictorForm">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" value="55" required min="1" max="120">
                </div>
                <div class="form-group">
                    <label for="sex">Sex</label>
                    <select id="sex" name="sex" required>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="cp">Chest Pain Type (cp)</label>
                    <select id="cp" name="cp" required>
                        <option value="0">Typical Angina (0)</option>
                        <option value="1">Atypical Angina (1)</option>
                        <option value="2">Non-anginal Pain (2)</option>
                        <option value="3">Asymptomatic (3)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="trestbps">Resting Blood Pressure (mm Hg)</label>
                    <input type="number" id="trestbps" name="trestbps" value="130" required>
                </div>
                <div class="form-group">
                    <label for="chol">Serum Cholesterol (mg/dl)</label>
                    <input type="number" id="chol" name="chol" value="250" required>
                </div>
                <div class="form-group">
                    <label for="fbs">Fasting Blood Sugar > 120 mg/dl</label>
                    <select id="fbs" name="fbs" required>
                        <option value="0">False (0)</option>
                        <option value="1">True (1)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="restecg">Resting ECG Results</label>
                    <select id="restecg" name="restecg" required>
                        <option value="0">Normal (0)</option>
                        <option value="1">ST-T Wave Abnormality (1)</option>
                        <option value="2">Left Ventricular Hypertrophy (2)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="thalach">Max Heart Rate Achieved</label>
                    <input type="number" id="thalach" name="thalach" value="150" required>
                </div>
                <div class="form-group">
                    <label for="exang">Exercise Induced Angina</label>
                    <select id="exang" name="exang" required>
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="oldpeak">ST Depression (oldpeak)</label>
                    <input type="number" step="0.1" id="oldpeak" name="oldpeak" value="1.0" required>
                </div>
                <div class="form-group">
                    <label for="slope">Slope of Peak Exercise ST</label>
                    <select id="slope" name="slope" required>
                        <option value="0">Upsloping (0)</option>
                        <option value="1">Flat (1)</option>
                        <option value="2">Downsloping (2)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="ca">Major Vessels Colored (0-3)</label>
                    <input type="number" id="ca" name="ca" value="0" min="0" max="3" required>
                </div>
                <div class="form-group full-width">
                    <label for="thal">Thalassemia (thal)</label>
                    <select id="thal" name="thal" required>
                        <option value="1">Normal (1)</option>
                        <option value="2">Fixed Defect (2)</option>
                        <option value="3">Reversable Defect (3)</option>
                    </select>
                </div>
            </div>
            <button type="submit">Predict</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictorForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'none';
            
            const data = {
                age: parseInt(document.getElementById('age').value),
                sex: parseInt(document.getElementById('sex').value),
                cp: parseInt(document.getElementById('cp').value),
                trestbps: parseInt(document.getElementById('trestbps').value),
                chol: parseInt(document.getElementById('chol').value),
                fbs: parseInt(document.getElementById('fbs').value),
                restecg: parseInt(document.getElementById('restecg').value),
                thalach: parseInt(document.getElementById('thalach').value),
                exang: parseInt(document.getElementById('exang').value),
                oldpeak: parseFloat(document.getElementById('oldpeak').value),
                slope: parseInt(document.getElementById('slope').value),
                ca: parseInt(document.getElementById('ca').value),
                thal: parseInt(document.getElementById('thal').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    resultDiv.style.display = 'block';
                    resultDiv.textContent = result.interpretation;
                    if (result.prediction === 1) {
                        resultDiv.className = 'danger';
                    } else {
                        resultDiv.className = 'success';
                    }
                } else {
                    resultDiv.style.display = 'block';
                    resultDiv.className = 'danger';
                    resultDiv.textContent = 'Error: ' + result.message;
                }
            } catch (err) {
                resultDiv.style.display = 'block';
                resultDiv.className = 'danger';
                resultDiv.textContent = 'Failed to connect to the server.';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)

        # Expected features in the correct order:
        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
            data['fbs'], data['restecg'], data['thalach'], data['exang'],
            data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]

        # Convert to numpy array and reshape for prediction
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
