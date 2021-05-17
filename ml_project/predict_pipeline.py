import logging
import pandas as pd
import sys
import joblib
import argparse

from src.features.build_features import process_features

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict(
    data_path: str,
    model_path: str = "models/model.pkl",
    transformer_path: str = "models/transformer.pkl",
) -> pd.Series:

    data = pd.read_csv(data_path)
    transformer = joblib.load(transformer_path)
    features = process_features(transformer, data)

    model = joblib.load(model_path)
    preds = pd.Series(model.predict(features))
    logger.debug(f"preds shape: {preds.shape}")
    preds.to_csv("data/preds.csv")

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, default="data/raw/heart.csv")
    parser.add_argument("--model", required=True, default="models/model.pkl")
    parser.add_argument(
        "--transformer", required=True, default="models/transformer.pkl"
    )
    args = parser.parse_args()
    predict(args.data, args.model, args.transformer)
