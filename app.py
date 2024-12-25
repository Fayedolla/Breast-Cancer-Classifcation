import os
from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import json

# Directory to save uploaded files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load a machine learning model from a file
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load machine learning models
svm_model = load_model(r'D:\University\Level - 3\Semester - 1\Machine Learning\Project\SVM')
dt_model = load_model(r'D:\University\Level - 3\Semester - 1\Machine Learning\Project\Decision_Tree')
lr_model = load_model(r'D:\University\Level - 3\Semester - 1\Machine Learning\Project\Logistic_Regression')
nb_model = load_model(r'D:\University\Level - 3\Semester - 1\Machine Learning\Project\Naive_Bayes')
kn_model = load_model(r'D:\University\Level - 3\Semester - 1\Machine Learning\Project\KNN')

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to process feature ranges (averages min and max values)
def process_feature_range(feature_range):
    return (feature_range[0] + feature_range[1]) / 2

# Home route
@app.route('/')
def home():
    return '''
    <h1>Welcome to the Machine Learning API!</h1>
    <p>Use <code>/predict</code> for manual input or <code>/predict_file</code> for JSON file uploads.</p>
    <h3>Supported Models:</h3>
    <ul>
        <li>svm</li>
        <li>decision_tree</li>
        <li>logistic_regression</li>
        <li>naive_bayes</li>
        <li>knn</li>
    </ul>
    '''

# Endpoint for predictions using manual input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'model' not in data or 'features' not in data:
            return jsonify({'error': 'Invalid input. Provide "model" and "features".'}), 400

        model_type = data['model'].lower()
        features = np.array(data['features']).reshape(1, -1)

        # Select the model and make a prediction
        if model_type == 'svm':
            prediction = svm_model.predict(features)
            model_name = 'SVM'
        elif model_type == 'decision_tree':
            prediction = dt_model.predict(features)
            model_name = 'Decision Tree'
        elif model_type == 'logistic_regression':
            prediction = lr_model.predict(features)
            model_name = 'Logistic Regression'
        elif model_type == 'naive_bayes':
            prediction = nb_model.predict(features)
            model_name = 'Naive Bayes'
        elif model_type == 'knn':
            prediction = kn_model.predict(features)
            model_name = 'KNN'
        else:
            return jsonify({'error': 'Invalid model type.'}), 400

        # Convert prediction to label
        prediction_label = 'M' if prediction[0] == 1 else 'B'
        return jsonify({
            'model': model_name,
            'prediction': prediction_label,
            'message': 'Prediction successful'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for predictions using a JSON file
@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and file.filename.endswith('.json'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            with open(file_path, 'r') as f:
                data = json.load(f)

            if not all(k in data for k in ['model', 'features']):
                return jsonify({'error': 'Invalid JSON structure. Ensure "model" and "features" are present.'}), 400

            model_type = data['model'].lower()
            features = data['features']

            if not isinstance(features, dict) or len(features) == 0:
                return jsonify({'error': 'Features should be a non-empty dictionary.'}), 400

            processed_features = np.array([process_feature_range(features[feature]) for feature in sorted(features)]).reshape(1, -1)

            if model_type == 'svm':
                prediction = svm_model.predict(processed_features)
                model_name = 'SVM'
            elif model_type == 'decision_tree':
                prediction = dt_model.predict(processed_features)
                model_name = 'Decision Tree'
            elif model_type == 'logistic_regression':
                prediction = lr_model.predict(processed_features)
                model_name = 'Logistic Regression'
            elif model_type == 'naive_bayes':
                prediction = nb_model.predict(processed_features)
                model_name = 'Naive Bayes'
            elif model_type == 'knn':
                prediction = kn_model.predict(processed_features)
                model_name = 'KNN'
            else:
                return jsonify({'error': 'Invalid model type.'}), 400

            prediction_label = 'M' if prediction[0] == 1 else 'B'
            return jsonify({
                'model': model_name,
                'prediction': prediction_label,
                'message': 'Prediction successful'
            })
        else:
            return jsonify({'error': 'File must be a JSON file.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
