# White Wine Quality Prediction Project

## Project Overview
This project predicts the quality of white wine using machine learning models based on physicochemical properties. The project includes data preprocessing, model training, hyperparameter tuning, and a FastAPI-based web service for real-time predictions.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Technologies Used](#technologies-used)

## Dataset
The dataset used is the **White Wine Quality** dataset from Kaggle. It contains 4,898 samples with 11 physicochemical features:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

**Target Variable**: Quality (reclassified into categories: Bad, Average, Good, Best)

## Project Structure
```
wine-quality-prediction/
│
├── modelling_white_wine.ipynb    # Main notebook with data analysis and modeling
├── wineapp.py                     # FastAPI application
├── best_model.pkl                 # Saved trained Random Forest model
├── scaler.pkl                     # Saved MinMaxScaler
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup Instructions

1. **Clone the repository**
```bash
git clone <https://github.com/Olasquare043/Practice_white_wine_dataset>
cd wine-quality-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries
```
numpy
pandas
matplotlib
scikit-learn
fastapi
uvicorn
pydantic
joblib
```

## Usage

### Training the Model

1. Open and run the Jupyter notebook:
```bash
jupyter notebook modelling_white_wine.ipynb
```

2. The notebook performs:
   - Data loading and exploration
   - Data preprocessing (handling missing values, scaling)
   - Feature engineering (quality reclassification)
   - Model training (6 different algorithms)
   - Hyperparameter tuning using RandomizedSearchCV
   - Model evaluation and saving

### Running the API

1. **Start the FastAPI server**
```bash
uvicorn wineapp:app --reload
```

2. **Access the API**
   - API will be available at: `http://127.0.0.1:8000`
   - Interactive documentation: `http://127.0.0.1:8000/docs`
   - Alternative documentation: `http://127.0.0.1:8000/redoc`

### Making Predictions

#### Using cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "fixed_acidity": 7.0,
           "volatile_acidity": 0.27,
           "citric_acid": 0.36,
           "residual_sugar": 20.7,
           "chlorides": 0.045,
           "free_sulfur_dioxide": 45.0,
           "total_sulfur_dioxide": 170.0,
           "density": 1.0,
           "pH": 3.0,
           "sulphates": 0.45,
           "alcohol": 8.8
         }'
```

#### Using Python Requests
```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 20.7,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 1.0,
    "pH": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8
}

response = requests.post(url, json=data)
print(response.json())
```

## Model Performance

### Models Evaluated
| Model | Accuracy |
|-------|----------|
| Random Forest | 80.0% |
| Decision Tree | 72.2% |
| SVM | 71.7% |
| K Nearest Neighbors | 71.0% |
| Logistic Regression | 70.5% |
| Naive Bayes | 62.9% |

### Best Model: Random Forest
- **After Hyperparameter Tuning**:
  - Cross-validation Score: 77%
  - Test Accuracy: 81%
  
- **Best Parameters**:
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 2
  - min_samples_leaf: 1

### Classification Report (Random Forest)
```
              precision    recall  f1-score   support

     Average       0.78      0.69      0.73       291
         Bad       0.64      0.19      0.29        37
        Best       1.00      0.42      0.59        36
        Good       0.83      0.93      0.87       616

    accuracy                           0.81       980
```

## API Documentation

### Endpoints

#### 1. Home Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**:
```json
{
  "message": "Welcome to the White Wine Quality Prediction API"
}
```

#### 2. Prediction Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
```json
{
  "fixed_acidity": 7.0,
  "volatile_acidity": 0.27,
  "citric_acid": 0.36,
  "residual_sugar": 20.7,
  "chlorides": 0.045,
  "free_sulfur_dioxide": 45.0,
  "total_sulfur_dioxide": 170.0,
  "density": 1.0,
  "pH": 3.0,
  "sulphates": 0.45,
  "alcohol": 8.8
}
```
- **Response**:
```json
{
  "predicted_quality": "Good"
}
```

## Technologies Used

### Machine Learning
- **Scikit-learn**: Model training and evaluation
- **Pandas & NumPy**: Data manipulation
- **Matplotlib**: Data visualization

### Web Framework
- **FastAPI**: RESTful API development
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### Model Persistence
- **Joblib**: Model and scaler serialization


---

**Note**: Make sure to update the dataset path in the notebook if you're using a different location for the wine dataset.