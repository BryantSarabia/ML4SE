from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import ToxicityPredictor
from schemas import (ExampleComment, PredictionRequest,
                     PredictionResponse)

app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API for classifying toxic comments using ML/DL models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ToxicityPredictor(model_dir='../models')


@app.get("/")
async def root():
    return {
        "message": "Toxic Comment Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Classify a comment",
            "/examples": "GET - Get example comments",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = predictor.predict(request.comment)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/examples", response_model=List[ExampleComment])
async def get_examples():
    return [
        ExampleComment(
            text="This is a great article! Very informative and well-written.",
            category="clean"
        ),
        ExampleComment(
            text="You are completely stupid and don't know what you're talking about!",
            category="toxic"
        ),
        ExampleComment(
            text="I will find you and make you pay for this!",
            category="threat"
        ),
        ExampleComment(
            text="What an absolute idiot, go back to school!",
            category="insult"
        ),
        ExampleComment(
            text="This contains really offensive and vulgar language that shouldn't be here.",
            category="obscene"
        )
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
