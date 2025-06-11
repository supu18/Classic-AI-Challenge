from prophet.serialize import model_from_json, model_to_json
import pandas as pd
import joblib
import sys
import json
import os
def predict(year,month, model_path):
    """
    Predict the number of alcohol-related accidents for a given year and month.
    
    Parameters:
    year (int): Year for which to make the prediction.
    month (int): Month for which to make the prediction.
    
    Returns:
    int: Predicted number of alcohol-related accidents.
    """
    # Load the trained model
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    
    # Create a DataFrame for the future date
    future = pd.DataFrame({'ds': [pd.to_datetime(f"{year}-{month:02d}-01")]})
    
    # Make the prediction
    forecast = model.predict(future)
    # Predicted value
    predicted_value = forecast['yhat'].values[0]
    result= {
        'Category': 'Alkoholunfälle',
        'Type': 'insgesamt',
        'year': year,
        'month': f'{month:02d}',
        'predicted_value': predicted_value

    }
    # Return the predicted value
    return result
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'alcohol_accidents_model.pkl'))

if len(sys.argv)<3:
    year = 2021
    month = 1
else:
    year = int(sys.argv[1])
    month = int(sys.argv[2])
result = predict(year, month,model_path)

print(f"Prediction for {year}-{month:02d}: {result:}")