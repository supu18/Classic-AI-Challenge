from prophet import Prophet
import pandas as pd
import joblib
import os

def train_model(df):
    """
    Train a Prophet model on the provided data.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the training data with 'ds' and 'y' columns.
    
    Returns:
    Prophet: Trained Prophet model.
    """
    # Rename columns for Prophet
    df = df.rename(columns={
        'time_series': 'ds',
        'WERT': 'y'
    })
    
    # Initialize the Prophet model
    model = Prophet()
    
    # Fit the model to the data
    model.fit(df)
    
    return model

def save_model(model, model_dir):
    """
    Save the trained model to a file.
    
    Parameters:
    model (Prophet): Trained Prophet model.
    model_path (str): Path to save the model.
    """
    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))

    model_path = os.path.join(model_dir, 'alcohol_accidents_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


alcohol_data = pd.read_csv('../data/preprocessed/alcohol_accidents_preprocessed.csv')
model = train_model(alcohol_data)
model_dir = '../models'
save_model(model, model_dir)