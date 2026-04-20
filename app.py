from flask import Flask, request, jsonify
import joblib
import threading

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('logistic_regression_attack_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'description' not in data:
        return jsonify({'error': 'Please provide a "description" field in your JSON.'}), 400
    
    description = data['description']
    # The pipeline handles both TF-IDF vectorization and Logistic Regression prediction
    prediction = model.predict([description])[0]
    
    return jsonify({
        'input_description': description,
        'predicted_category': prediction
    })

def run_app():
    # Running on port 5000
    app.run(port=5000)

# Start Flask in a background thread so it doesn't block the notebook
flask_thread = threading.Thread(target=run_app)
flask_thread.start()
