from fastapi import FastAPI
from src.prediction import initialize, predict
from app.schemas import ListingInput
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    yield


app = FastAPI(
    title="Airbnb Price Prediction API",
    description="API for predicting Airbnb listing prices",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "Airbnb Price Prediction API is running"
    }


@app.post("/predict")
def prediction(data: ListingInput):
    predicted_price = predict(data.model_dump())

    return {
        "predicted_price": round(float(predicted_price), 2)
    }