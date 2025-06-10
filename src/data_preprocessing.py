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
    Parameters:
    data (pd.DataFrame): Input data to preprocess.
    Returns:
    pd.DataFrame: Preprocessed data.
    """
    # Remove records after 2020 
    if 'JAHR' in data.columns:
        data_filtered = data[data['JAHR'] <= 2020]
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

    return alcohol_data
def save_preprocessed_data(data, output_path):
    """
    Save the preprocessed data to a CSV file.
    Parameters:
    data (pd.DataFrame): Preprocessed data to save.
    output_path (str): Path to save the preprocessed data.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")



input_file = '../data/raw/monatszahlen2505_verkehrsunfaelle_06_06_25.csv'
output_file = '../data/preprocessed/alcohol_accidents_preprocessed.csv'

# Load data
data = load_data(input_file)

# Preprocess data
preprocessed_data = preprocess_data(data)

# Save preprocessed data
save_preprocessed_data(preprocessed_data, output_file)

