import pickle
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

# request
class Lead(BaseModel):
    lead_source: str
    industry: str
    employment_status: str
    location: str
    number_of_courses_viewed: int
    annual_income: float
    interaction_count: int
    lead_score: float

app = FastAPI(title='churn-prediction')

# response
class PredictResponse(BaseModel):
    conversion_probability: float
    conversion: bool

with open('pipeline.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)
    
def predict_single(lead):
    result = pipeline.predict_proba(lead)[0][1]
    return float(result)
    
@app.post("/predict")
def predict(lead: Lead) -> PredictResponse:
    prob = predict_single(lead.model_dump())
    
    return PredictResponse(
        conversion_probability = prob,
        conversion = bool(prob >= 0.5),
    )
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 9696)