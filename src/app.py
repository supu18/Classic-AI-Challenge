from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from prediction import predict

app = Flask(__name__)

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'alcohol_accidents_model.pkl'))



@app.route('/predict', methods=['POST'])

def predict_route():
    """
    Endpoint to predict the number of alcohol-related accidents for a given year and month.
    
    Expects a JSON payload with 'year' and 'month'.
    Returns a JSON response with the prediction.
    """
    data = request.get_json()
    
    year = data.get('year', 2021)
    month = data.get('month', 1)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    result = predict(year, month, model_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

