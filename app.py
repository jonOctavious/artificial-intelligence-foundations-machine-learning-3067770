# app.py
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

# Load trained model and training columns
model = joblib.load("xgb_tuned_model.pkl")
train_columns = joblib.load("train_columns.pkl")

app = FastAPI(title="Adult Income Prediction API")

# Allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or put ["https://nintex-support.skuidsite.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for one prediction
class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.post("/predict")
def predict(data: Union[Person, List[Person]]):
    # Handle both single and list inputs
    if isinstance(data, Person):
        data = [data.dict()]
    else:
        data = [item.dict() for item in data]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # One-hot encode
    df_encoded = pd.get_dummies(df)

    # Align with training columns
    df_encoded = df_encoded.reindex(columns=train_columns, fill_value=0)

    # Predict
    predictions = model.predict(df_encoded)
    probabilities = model.predict_proba(df_encoded)[:, 1]

    # Build response
    results = []
    for i, row in enumerate(data):
        results.append({
            "input": {
                "sex": row["sex"],
                "age": row["age"],
                "education": row["education"],
                "marital_status": row["marital_status"]
            },
            "prediction": int(predictions[i]),
            "probability": float(probabilities[i])
        })

    return {"results": results}
