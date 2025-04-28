from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model_handler import ModelHandler

app = FastAPI()

# Load model at startup
model_handler = ModelHandler(model_path="model.xgb")

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the frontend at the root path "/"
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("static/index.html")

# Define the data model for prediction
class UserFeatures(BaseModel):
    Gender: str
    Education_Level: str
    Income_Category: str
    Total_Relationship_Count: float
    Months_Inactive_12_mon: float
    Contacts_Count_12_mon: float
    Total_Revolving_Bal: float
    Total_Trans_Ct: float

# API endpoint for churn prediction
@app.post("/predict")
def predict_churn(features: UserFeatures):
    features_dict = features.dict()
    result = model_handler.predict(features_dict)
    return result
