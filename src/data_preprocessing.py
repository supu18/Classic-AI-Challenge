import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the data to prepare data for model training and evaluation.
    Creates three datasets:
    1. Complete data (all categories ≤2020) - for visualization
    2. Multi-category training data (all categories ≤2020) - for training prediction model
    3. Alcohol accidents data (≤2020) - for target analysis and simple model fallback
    Parameters:
    data (pd.DataFrame): Input data to preprocess.
    Returns:
    pd.DataFrame: Preprocessed data.
    """
    # Remove records after 2020
    data_filtered = data[data['JAHR'] <= 2020].copy()
    print(f"After filtering ≤2020: {len(data_filtered)} records")
    
    # Remove 'Summe' entries in the MONAT column
    data_filtered = data_filtered[data_filtered['MONAT'] != 'Summe'].copy()
    print(f"After removing 'Summe' entries: {len(data_filtered)} records")
    
    # Convert MONAT to proper datetime format
    data_filtered['date'] = pd.to_datetime(data_filtered['MONAT'], format='%Y%m')
    
    # Handle missing values 
    data_filtered.fillna(0, inplace=True)
    
    # Sort by date for proper time series analysis
    data_filtered = data_filtered.sort_values(['MONATSZAHL', 'AUSPRAEGUNG', 'date']).reset_index(drop=True)

    #complete dataset (all categories) for visualization
    complete_data = data_filtered.copy()

    # Focus on Alkoholunfälle category with insgesamt type
    alcohol_data = data_filtered[
        (data_filtered['MONATSZAHL'] == 'Alkoholunfälle') & 
        (data_filtered['AUSPRAEGUNG'] == 'insgesamt')
    ].copy()

    print(f"Filtered data contains {len(alcohol_data)} records.")

    alcohol_data = alcohol_data[alcohol_data['MONAT'] != 'Summe'].copy()
    # Then convert to datetime
    alcohol_data['time_series'] = pd.to_datetime(alcohol_data['MONAT'], format='%Y%m')
    print(f"Converted 'MONAT' to datetime format.", alcohol_data['time_series'].head())

    # Handle missing values
    alcohol_data.fillna(0, inplace=True)
 # Sort by date
    alcohol_data = alcohol_data.sort_values('time_series').reset_index(drop=True)
    print(f"Preprocessed data contains {len(alcohol_data)} records.")

    return alcohol_data , complete_data
def save_preprocessed_data(alcohol_data , complete_data, output_dir):
    """
    Save the preprocessed data to a CSV file.
    Parameters:
    data (pd.DataFrame): Preprocessed data to save.
    output_path (str): Path to save the preprocessed data.
    """
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    complete_path = os.path.join(output_dir, 'complete_accidents_data.csv')
    complete_data.to_csv(complete_path, index=False)
    print(f"Preprocessed data saved to {complete_path}")
    alcohol_path = os.path.join(output_dir, 'alcohol_accidents_preprocessed.csv')
    alcohol_data.to_csv(alcohol_path, index=False)
    print(f"Alcohol accidents data saved to {alcohol_path}")



input_file = '../data/raw/monatszahlen2505_verkehrsunfaelle_06_06_25.csv'
output_dir = '../data/preprocessed/'

# Load data
data = load_data(input_file)

# Preprocess data
alcohol_data, complete_data = preprocess_data(data)

# Save preprocessed data
save_preprocessed_data(alcohol_data, complete_data, output_dir)

