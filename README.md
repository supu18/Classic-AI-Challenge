# Classic-AI-Challenge
Digital Product School Challenge

This repository contains the code for the Classic AI Challenge, a project developed as part of the Digital Product School program. The challenge focuses on implementing classic AI algorithms and techniques to solve various problems.

# Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```cd
   git clone <repository-url>
2. Navigate to the project directory:
   ```cd
   cd Classic-AI-Challenge
   ```
3. Install the required dependencies:
   ```cd
   pip install -r requirements.txt
   ```
4. Run the application:
   ```cd
   python app.py
   ```
# Usage

The app exposes a REST API endpoint to make predictions.

# API Endpoints
## Predict Alcohol Accidents
### Endpoint: `https://accident-predictor.onrender.com/predict`
### Method: `POST`
### Request Body:
```json
{
  "year": 2021,
  "month": 1
}
```
### Response:
```json
{
  "Category": "Alkoholunfälle",
  "Type": "insgesamt",
  "year": 2021,
  "month": "01",
  "predicted_value": <predicted_value>
}
```

# Data
The project uses a dataset of traffic accidents, specifically focusing on alcohol-related accidents. The data is preprocessed to create training datasets.

# Project Structure
```
Classic-AI-Challenge/
├── src/
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── prediction.py
│   ├── train_model.py
├── data/
│   ├── preprocessed/
│   │   ├── alcohol_accidents_preprocessed.csv
│   │   ├── complete_accidents_data.csv
│   ├── raw/
│   ├── monatszahlen2505_verkehrsunfaelle_06_06_25.csv
├── models/
│   ├── alcohol_accidents_model.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_visualization.ipynb
├── results/
│   ├── interactive_accident_analysis.html
│   ├── interactive_alcohol_accidents.html
├── requirements.txt
├── LICENSE
├── README.md
```
# Visualization
The project includes Jupyter notebooks for data exploration and visualization. These notebooks provide insights into the dataset and the results of the predictions.

# Data Preprocessing
The data preprocessing script handles the following tasks:
- Loading the dataset
- Filtering records to include only those up to the year 2020
- Removing 'Summe' entries in the month column
- Converting the month column to a proper datetime format
- Handling missing values
- Sorting the data by date for time series analysis
# Model Training
The model training script uses the Prophet library to train a time series forecasting model on the alcohol-related accidents data. The trained model is saved for later use in predictions.
# Model Prediction
The prediction script loads the trained model and makes predictions based on the provided year and month. It returns the predicted number of alcohol-related accidents for that period.
# License
This project is licensed under the MIT License. See the LICENSE file for details.



