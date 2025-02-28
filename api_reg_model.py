from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Create instance of FastAPI
app = FastAPI()

# Create a class for the request body
class request_body(BaseModel):
  irrigation_hours : int
  
# Load the model
irrigation_model = joblib.load('./reg_model.pkl')
  
@app.post('/predict')
def predict(data : request_body):
  # Prepare data for prediction
  input_feature = [[data.irrigation_hours]]
  
  # Make prediction
  y_pred = irrigation_model.predict(input_feature)[0].astype(float)

  return {'area_irrigated_by_angle' : y_pred.tolist()}