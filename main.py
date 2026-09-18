from fastapi import FastAPI
import joblib
from pydantic import BaseModel , Field
import pandas as pd


app = FastAPI()
model = joblib.load("Models/Ecommerece_model.pkl")    

class EcommerceInput(BaseModel):
    avg_session_length: float = Field(alias="Avg. Session Length")
    time_on_app: float = Field(alias="Time on App")
    time_on_website: float = Field(alias="Time on Website")
    length_of_membership: float = Field(alias="Length of Membership")

    model_config = {
        "populate_by_name": True
    }

@app.post("/predict")
def predict(data: EcommerceInput):

    input_df = pd.DataFrame([{
        "Avg. Session Length": data.avg_session_length,
        "Time on App": data.time_on_app,
        "Time on Website": data.time_on_website,
        "Length of Membership": data.length_of_membership
    }])

    prediction = model.predict(input_df)[0]

    return {"prediction": prediction}


@app.get('/health')
def health_check():
    return {"status":"ok"}

@app.get("/")
def home():
    return {"message": "Welcome to the Ecommerce Prediction API"}   