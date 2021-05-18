import logging
import os
import sys
import joblib
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse

from entities import DiagnosisRequest, DiagnosisResponse
from build_features import process_features

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


TRANSFORMER_PATH = "models/transformer.pkl"
MODEL_PATH = "models/model.pkl"


def load_object(path: str) -> dict:
    with open(path, "rb") as file:
        model = joblib.load(file)
        return model


model: Optional[dict] = None


def make_predict(
    data: List,
    features: List,
    model,
) -> List[DiagnosisResponse]:

    data = pd.DataFrame(data, columns=features)
    ids = data["id"]
    features = data.drop(["id"], axis=1)

    transformer = joblib.load(TRANSFORMER_PATH)
    features = process_features(transformer, features)

    preds = model.predict(features)
    logger.info(f"preds info: {preds.shape}, {preds}")

    return [
        DiagnosisResponse(id=id, diagnosis=int(diagnosis))
        for id, diagnosis in zip(ids, preds)
    ]


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def main():
    return "It is entry point of predictor"


@app.on_event("startup")
def load_model():
    logger.info(f"Loading model...")
    global model

    if MODEL_PATH is None:
        err = f"PATH_TO_MODEL {MODEL_PATH} is None"
        logger.error(err)  # ok
        raise RuntimeError(err)
    model = load_object(MODEL_PATH)

    logger.info("Model is ready...")


@app.get("/healthz")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[DiagnosisResponse])
def predict(request: DiagnosisRequest):
    logger.info(f"Predict request info: {request.features}, {request.data}")
    logger.info(f"Predict model info: {model}")
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
