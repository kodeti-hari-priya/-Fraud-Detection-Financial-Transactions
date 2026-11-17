
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------------------
# 1. INITIALIZE APPLICATION
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "fraud_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

app = FastAPI(title="Financial Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------
# 2. LOAD TRAINED MODEL
# -----------------------------------------

model = None
metadata = None


def load_model():
    global model, metadata

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Trained model files are missing from artifacts/"
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)


# -----------------------------------------
# 3. INPUT DATA STRUCTURE
# -----------------------------------------

class Transaction(BaseModel):
    amount: float = Field(ge=0)
    timestamp: datetime
    customer_location: str
    payment_method: str


# -----------------------------------------
# 4. HEALTH CHECK
# -----------------------------------------

@app.get("/")
def home():
    return {
        "message": "Financial Fraud Detection API is running",
        "status": "online"
    }


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_loaded": model is not None
    }


# -----------------------------------------
# 5. FRAUD PREDICTION
# -----------------------------------------

@app.post("/api/predict")
def predict(transaction: Transaction):

    global model, metadata

    if model is None or metadata is None:
        try:
            load_model()
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=str(error)
            )

    try:
        timestamp = transaction.timestamp

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        timestamp = timestamp.astimezone(timezone.utc)

        # Prepare transaction features
        input_data = {
            "amount": transaction.amount,
            "customer_location": transaction.customer_location,
            "payment_method": transaction.payment_method,
            "transaction_hour": timestamp.hour,
            "transaction_day": timestamp.day,
            "transaction_month": timestamp.month,
            "transaction_weekday": timestamp.weekday()
        }

        # Match the exact training feature order
        feature_columns = metadata["feature_columns"]

        missing_features = [
            column for column in feature_columns
            if column not in input_data
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Input features do not match training data",
                    "missing_features": missing_features
                }
            )

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        prediction = int(model.predict(input_df)[0])

        result = (
            "FRAUD" if prediction == 1
            else "LEGITIMATE"
        )

        return {
            "success": True,
            "prediction": prediction,
            "result": result,
            "message": (
                "Suspicious transaction detected"
                if prediction == 1
                else "Transaction classified as legitimate"
            ),
            "model": metadata.get("best_model", "Trained model")
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
