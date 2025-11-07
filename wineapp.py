# importting necessary libraries
import numpy as np
import joblib
from pydantic import BaseModel, Field
from fastapi import FastAPI


# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input data model
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., example="7.0")
    volatile_acidity: float = Field(..., example="0.27")
    citric_acid: float = Field(..., example="0.36")
    residual_sugar: float = Field(..., example="20.7")
    chlorides: float = Field(..., example="0.045")
    free_sulfur_dioxide: float = Field(..., example="45.0")
    total_sulfur_dioxide: float = Field(..., example="170.0")
    density: float = Field(..., example="1.0")
    pH: float = Field(..., example="3.0")
    sulphates: float = Field(..., example="0.45")
    alcohol: float = Field(..., example="8.8")

# let's define a home endpoint
@app.get("/")
def home_page():
    return {"message": "Welcome to the White Wine Quality Prediction API"}

# Define a prediction endpoint
@app.post("/predict")
def predict_wine_quality(features: WineFeatures):
    # # Convert input data to 2d Array
    features = np.array([[features.fixed_acidity,
                          features.volatile_acidity,
                          features.citric_acid,
                          features.residual_sugar,
                          features.chlorides,
                          features.free_sulfur_dioxide,
                          features.total_sulfur_dioxide,
                          features.density,
                          features.pH,
                          features.sulphates,
                          features.alcohol]])

    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return the prediction result
    return {"predicted_quality": str(prediction[0])}

# To run the app with Uvicorn, use the command:
# uvicorn wineapp:app --reload
