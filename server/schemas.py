from pydantic import BaseModel, Field
from typing import Dict


class PredictionRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="The comment text to classify")


class PredictionResponse(BaseModel):
    comment: str
    predictions: Dict[str, float]
    risk_level: str
    highest_risk: str


class ExampleComment(BaseModel):
    text: str
    category: str
